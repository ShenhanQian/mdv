from pathlib import Path
from dataclasses import dataclass, field
from typing import Annotated, List
import tyro
from tyro.conf import Positional
import numpy as np
from PIL import Image
import h5py
import cv2
import pyrender
import trimesh
import multiprocessing as mp
import threading
import dearpygui.dearpygui as dpg
from .utils.camera import OrbitCamera
from mdv.utils.flow import flow_to_image


@dataclass
class MultiDimensionViewerConfig:
    root_folder: Positional[Path]
    """Root folder of the data"""
    width: int = 720
    """Width of the GUI"""
    height: int = 720
    """Height of the GUI"""
    scale: Annotated[float, tyro.conf.arg(aliases=["-s"])] = 1.0
    """Scale of the GUI"""
    exclude_suffixes: Annotated[List[str], tyro.conf.arg(aliases=["-e"])] = field(default_factory=lambda: [])
    """Exclude files with these suffixes"""
    include_suffixes: Annotated[List[str], tyro.conf.arg(aliases=["-i"])] = field(default_factory=lambda: [])
    """Include files with these suffixes"""
    rescale_depth_map: bool = True
    """Rescale depth map for visualization"""
    verbose: Annotated[bool, tyro.conf.arg(aliases=["-v"])] = False
    """Verbose mode"""
    cam_radius: float = 1.0
    """Radius of the camera orbit"""
    cam_fovy: float = 45
    """Field of view of the camera"""
    cam_convention: str = "opengl"
    """Camera convention"""
    light_intensity: float = 10
    """Directional light intensity of the 3D viewer"""
    ambient_light: float = 0.3
    """Ambient light of the 3D viewer"""


class MultiDimensionViewer(object):
    def __init__(self, cfg: MultiDimensionViewerConfig):
        self.root_folder = cfg.root_folder
        self.scale = cfg.scale
        self.width = int(cfg.width * self.scale)
        self.height = int(cfg.height * self.scale)
        self.nav_pos = [int(self.width-(262+30)*self.scale), 10*self.scale]
        self.rescale_depth_map = cfg.rescale_depth_map
        self.verbose = cfg.verbose

        # files types
        self.types_image = ['jpg', 'jpeg', 'png', 'flo5', 'dsp5']
        self.types_mesh = ['obj', 'glb', 'ply']
        self.types_container = ['npz']
        # self.types_txt = ['txt', 'json', 'csv', 'sh']
        self.exclude_suffixes = cfg.exclude_suffixes
        self.include_suffixes = cfg.include_suffixes

        # styles
        self.selectable_width = 12 * self.scale

        # database
        self.active_level = 0
        self.items_levels = {}
        assert self.root_folder.exists(), f"Root folder {self.root_folder} does not exist."
        if self.root_folder.is_file():
            raise NotImplementedError("File is not supported yet.")
        elif self.root_folder.is_dir():
            if len(self.include_suffixes) > 0:
                items = sorted([x.name for x in self.root_folder.iterdir() if x.is_dir() or x.suffix[1:].lower() in self.include_suffixes])
            else:
                items = sorted([x.name for x in self.root_folder.iterdir() if x.is_dir() or x.suffix[1:].lower() not in self.exclude_suffixes])
            self.items_levels.update({0: items})
        self.selected_idx_levels = {self.active_level: 0}
        self.update_items_under_level(self.selected_idx_levels, self.items_levels, self.active_level, update_widgets=False)
        
        # prefetch
        self.prefetch_cache = {}
        self.need_prefetch = True
        self.io_busy = False

        # mesh rendering
        self.render_input_queue = mp.Queue()
        self.render_output_queue = mp.Queue()
        self.mesh_render_process = None
        self.scene = None
        self.cam = OrbitCamera(self.width, self.height, r=cfg.cam_radius, fovy=cfg.cam_fovy, convention=cfg.cam_convention)
        self.light_intensity = cfg.light_intensity
        self.ambient_light = cfg.ambient_light

        # buffers for mouse interaction
        self.cursor_x = None
        self.cursor_y = None
        self.cursor_x_prev = None
        self.cursor_y_prev = None
        self.drag_begin_x = None
        self.drag_begin_y = None
        self.drag_button = None

        self.status_pos = [0, self.height]
        self.status_height = int(20 * self.scale)

    def get_absolate_path(self, selected_idx_levels, items_levels, level):  # TODO: added itmers_levels, need update calling functions
        path = self.root_folder
        for i in range(level+1):
            if selected_idx_levels[i] is None:
                return Path('-')
            if len(items_levels[i]) == 0:
                return Path('-')
            path = path / items_levels[i][selected_idx_levels[i]]
        return path
    
    def update_items_under_level(self, selected_idx_levels, items_levels, level, update_widgets=True):
        base_path = self.get_absolate_path(selected_idx_levels, items_levels, level)
        while base_path.is_dir():
            level += 1

            if len(self.include_suffixes) > 0:
                items = sorted([x.name for x in base_path.iterdir() if x.is_dir() or x.suffix[1:].lower() in self.include_suffixes])
            else:
                items = sorted([x.name for x in base_path.iterdir() if x.is_dir() or x.suffix[1:].lower() not in self.exclude_suffixes])

            if level in selected_idx_levels:
                try:
                    if selected_idx_levels[level] < len(items_levels[level]):
                        selected_idx = items.index(items_levels[level][selected_idx_levels[level]])
                    else:
                        selected_idx = 0
                except ValueError:
                    selected_idx = 0
            else:
                selected_idx = 0
            selected_idx_levels.update({level: selected_idx})
            items_levels.update({level: items})
            selected = items[selected_idx] if len(items) > 0 else '-'

            if update_widgets:
                # add if not exist
                if not dpg.does_item_exist(f'combo_level_{level}'):
                    # check if slider exists
                    if dpg.does_item_exist(f'slider_level'):
                        dpg.delete_item(f'slider_level')

                    with dpg.group(horizontal=True, parent='navigator_tag', tag=f'group_level_{level}'):
                        dpg.add_combo(items, default_value=selected, height_mode=dpg.mvComboHeight_Large, callback=lambda sender, data: self.set_item(sender, data), tag=f'combo_level_{level}')
                        
                        dpg.add_button(label="<", callback=lambda sender, data: self.prev_item(sender, data), tag=f'button_left_level_{level}')
                        dpg.add_button(label=">", callback=lambda sender, data: self.next_item(sender, data), tag=f'button_right_level_{level}')

                        dpg.add_selectable(width=self.selectable_width, default_value=level==self.active_level, tag=f'selectable_level_{level}', callback=lambda sender, data: self.set_level(sender, data))
                        dpg.bind_item_theme(f'selectable_level_{level}', self.selectable_theme)
                else:
                    dpg.configure_item(f'combo_level_{level}', items=items, default_value=selected)
                self.update_button_states(level)
           
            if self.verbose:
                print(f"Update level {level} with item: {selected}")
            base_path = self.get_absolate_path(selected_idx_levels, items_levels, level)
        
        # remove outdated deeper levels
        for l in range(level+1, len(items_levels)):
            if l in items_levels:
                items_levels.pop(l)
            if l in selected_idx_levels:
                selected_idx_levels.pop(l)
            if update_widgets:
                dpg.delete_item(f'group_level_{l}')
        if self.active_level not in items_levels:
            self.set_level(f'selectable_level_{level}', None)
        
        if update_widgets:
            if not dpg.does_item_exist(f'slider_level'):
                dpg.add_slider_int(default_value=0, min_value=0, max_value=len(items_levels[self.active_level])-1, tag=f'slider_level', callback=lambda sender, data: self.set_item(f"slider_{self.active_level}", items_levels[self.active_level][data]), parent='navigator_tag')

    def define_gui(self):
        dpg.create_context()

        # theme
        with dpg.theme() as theme_no_padding:
            with dpg.theme_component(dpg.mvAll):
                # set all padding to 0 to avoid scroll bar
                dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 0, 0, category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 0, 0, category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_CellPadding, 0, 0, category=dpg.mvThemeCat_Core)

        with dpg.theme() as self.selectable_theme:
            with dpg.theme_component(dpg.mvSelectable):
                color = [227, 179, 65]  # light yellow
                # color = [15, 86, 135]  # blue
                color_hovered = [int(x*200/255) for x in color]
                color_active = [int(x*150/255) for x in color]
                dpg.add_theme_color(dpg.mvThemeCol_Header, color, category=dpg.mvThemeCat_Core)  # Change color here
                dpg.add_theme_color(dpg.mvThemeCol_HeaderHovered, color_hovered, category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_HeaderActive, color_active, category=dpg.mvThemeCat_Core)
        
        with dpg.theme() as self.button_theme:
            with dpg.theme_component(dpg.mvButton):
                color = [37, 37, 38]  # dark grey
                color_hovered = [int(x*200/255) for x in color]
                color_active = [int(x*150/255) for x in color]
                dpg.add_theme_color(dpg.mvThemeCol_Button, color, category=dpg.mvThemeCat_Core)  # Change color here
                # dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, color_hovered, category=dpg.mvThemeCat_Core)
                # dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, color_active, category=dpg.mvThemeCat_Core)

        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(width=self.width, height=self.height, default_value=np.zeros([self.height, self.width, 4]), format=dpg.mvFormat_Float_rgba, tag="texture_tag")

        # viewer window
        with dpg.window(label="Viewer", pos=[0, 0], tag='viewer_tag', width=self.width, height=self.height, no_title_bar=True, no_move=True, no_bring_to_front_on_focus=True):
            dpg.add_image("texture_tag", tag='image_tag', width=self.width, height=self.height, show=True)
            dpg.add_text("", tag="text_field_tag", wrap=0, show=False)
        dpg.bind_item_theme("viewer_tag", theme_no_padding)

        # navigator window
        with dpg.window(label="Navigator", tag='navigator_tag', pos=self.nav_pos, autosize=True, no_close=True):
            for level, items in self.items_levels.items():
                with dpg.group(horizontal=True, tag=f'group_level_{level}'):
                    if self.selected_idx_levels[level] is None or len(self.items_levels[level]) == 0:
                        default_value = '-'
                    else:
                        default_value = self.items_levels[level][self.selected_idx_levels[level]]
                    dpg.add_combo(items, default_value=default_value, height_mode=dpg.mvComboHeight_Regular, callback=lambda sender, data: self.set_item(sender, data), tag=f'combo_level_{level}')

                    dpg.add_button(label="<", callback=lambda sender, data: self.prev_item(sender, data), tag=f'button_left_level_{level}')
                    dpg.add_button(label=">", callback=lambda sender, data: self.next_item(sender, data), tag=f'button_right_level_{level}')
                    self.update_button_states(level)

                    dpg.add_selectable(width=self.selectable_width, default_value=level==self.active_level, tag=f'selectable_level_{level}', callback=lambda sender, data: self.set_level(sender, data))
                    dpg.bind_item_theme(f'selectable_level_{level}', self.selectable_theme)
            dpg.add_slider_int(default_value=0, min_value=0, max_value=len(self.items_levels[self.active_level])-1, tag=f'slider_level', callback=lambda sender, data: self.set_item(f"slider_{self.active_level}", self.items_levels[self.active_level][data]))

        # status bar window
        with dpg.window(label="Status", pos=self.status_pos, tag='status_tag', width=self.width, height=self.status_height, no_title_bar=True, no_move=True, no_resize=True):
            dpg.add_text("", tag='status_text_tag')
        dpg.bind_item_theme("status_tag", theme_no_padding)

        # key press handlers
        with dpg.handler_registry():
            dpg.add_key_press_handler(dpg.mvKey_Up, callback=self.prev_level)
            dpg.add_key_press_handler(dpg.mvKey_Down, callback=self.next_level)
            dpg.add_key_press_handler(dpg.mvKey_Left, callback=self.prev_item)
            dpg.add_key_press_handler(dpg.mvKey_Right, callback=self.next_item)
            dpg.add_key_press_handler(dpg.mvKey_Home, callback=self.set_item)
            dpg.add_key_press_handler(dpg.mvKey_End, callback=self.set_item)
            dpg.add_key_press_handler(dpg.mvKey_Escape, callback=lambda: dpg.stop_dearpygui())

            dpg.add_mouse_release_handler(callback=self.callback_mouse_release)
            # dpg.add_mouse_drag_handler(callback=callback_mouse_drag)  # not using the drag callback, since it does not return the starting point
            dpg.add_mouse_move_handler(callback=self.callback_mouse_move)
            dpg.add_mouse_down_handler(callback=self.callback_mouse_button_down)
            dpg.add_mouse_wheel_handler(callback=self.callback_mouse_wheel)

            dpg.add_key_press_handler(dpg.mvKey_W, callback=self.callback_key_press, tag='_mvKey_W')
            dpg.add_key_press_handler(dpg.mvKey_S, callback=self.callback_key_press, tag='_mvKey_S')
            dpg.add_key_press_handler(dpg.mvKey_A, callback=self.callback_key_press, tag='_mvKey_A')
            dpg.add_key_press_handler(dpg.mvKey_D, callback=self.callback_key_press, tag='_mvKey_D')
            dpg.add_key_press_handler(dpg.mvKey_E, callback=self.callback_key_press, tag='_mvKey_E')
            dpg.add_key_press_handler(dpg.mvKey_Q, callback=self.callback_key_press, tag='_mvKey_Q')
            dpg.add_key_press_handler(dpg.mvKey_R, callback=self.callback_key_press, tag='_mvKey_R')
            dpg.add_key_press_handler(dpg.mvKey_Open_Brace, callback=self.callback_key_press, tag='_mvKey_Open_Brace')
            dpg.add_key_press_handler(dpg.mvKey_Close_Brace, callback=self.callback_key_press, tag='_mvKey_Close_Brace')
    
    def set_level(self, sender, data):
        self.active_level = int(sender.split('_')[-1])
        if self.verbose:
            print(f"sender: {sender}, data: {data}, active_level: {self.active_level}")
        for level in self.items_levels.keys():
            dpg.set_value(f'selectable_level_{level}', level==self.active_level)
        self.update_slider()

        threading.Thread(target=self.prefetch_thread).start()
    
    def update_slider(self):
        if len(self.items_levels[self.active_level]) > 0:
            max_value = len(self.items_levels[self.active_level]) - 1
            slider_value = self.selected_idx_levels[self.active_level]
        else:
            max_value = 0
            slider_value = 0
        dpg.configure_item(f'slider_level', max_value=max_value)
        dpg.set_value(f'slider_level', slider_value)
        if self.verbose:
            print(f"Update slider with max value: {len(self.items_levels[self.active_level])-1}")

    def prev_level(self):
        if self.active_level > 0:
            self.set_level(f'selectable_level_{self.active_level-1}', None)

    def next_level(self):
        if self.active_level < len(self.items_levels) - 1:
            self.set_level(f'selectable_level_{self.active_level+1}', None)
    
    def set_item(self, sender, data):
        self.io_busy = True  # prevent prefetching while loading thec current file
        if self.verbose:
            print(f"sender: {sender}, data: {data}")

        if data == dpg.mvKey_Home:
            level = self.active_level
            data = self.items_levels[level][0]
        elif data == dpg.mvKey_End:
            level = self.active_level
            data = self.items_levels[level][-1]
        else:
            level = int(sender.split('_')[-1])
        idx = self.items_levels[level].index(data)
        self.selected_idx_levels[level] = idx
        dpg.set_value(f'combo_level_{level}', data)
        if level == self.active_level:
            dpg.set_value(f'slider_level', idx)
        self.need_prefetch = True
        if self.verbose:
            print(f"Set level {level} with item: {data}")

        self.update_items_under_level(self.selected_idx_levels, self.items_levels, level)
        self.update_button_states(level)
        self.scene = None
        self.need_update = True

    def prev_item(self, sender, data):
        if data == dpg.mvKey_Left:
            level = self.active_level
            sender = f'button_left_level_{level}'
        else:
            level = int(sender.split('_')[-1])
            self.set_level(f'selectable_level_{level}', None)

        idx = self.selected_idx_levels[level]
        if idx > 0:
            self.set_item(sender, self.items_levels[level][idx-1])
            self.update_slider()

    def next_item(self, sender, data):
        if data == dpg.mvKey_Right:
            level = self.active_level
            sender = f'button_right_level_{level}'
        else:
            level = int(sender.split('_')[-1])
            self.set_level(f'selectable_level_{level}', None)

        idx = self.selected_idx_levels[level]
        if idx < len(self.items_levels[level]) - 1:
            self.set_item(sender, self.items_levels[level][idx+1])
            self.update_slider()

    def resize_windows(self):
        dpg.configure_item('viewer_tag', width=self.width, height=self.height)
        dpg.configure_item('navigator_tag', pos=self.nav_pos)

        dpg.delete_item('texture_tag')
        dpg.delete_item('image_tag')
        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(width=self.width, height=self.height, default_value=np.zeros([self.height, self.width, 4]), format=dpg.mvFormat_Float_rgba, tag="texture_tag")
        dpg.add_image("texture_tag", tag='image_tag', parent='viewer_tag')
        self.prefetch_cache = {}
        self.need_update = True

        self.status_pos = [0, self.height]
        dpg.configure_item('status_tag', width=self.width, pos=self.status_pos)

        # restart thread to update rendering resolution
        if self.mesh_render_process is not None:
            self.render_input_queue.put(None)
            self.mesh_render_process.join()
            self.mesh_render_process = mp.Process(target=self.mesh_render_loop)
            self.mesh_render_process.start()
    
    def callback_mouse_move(self, sender, app_data):
        self.cursor_x, self.cursor_y = app_data

        if self.scene is None:
            return

        # drag
        if self.drag_begin_x is not None or self.drag_begin_y is not None:
            if self.cursor_x_prev is None or self.cursor_y_prev is None:
                cursor_x_prev = self.drag_begin_x
                cursor_y_prev = self.drag_begin_y
            else:
                cursor_x_prev = self.cursor_x_prev
                cursor_y_prev = self.cursor_y_prev
            
            # drag with left button
            if self.drag_button is dpg.mvMouseButton_Left:
                cx = self.width // 2
                cy = self.height // 2
                r = min(cx, cy) * 0.9
                # rotate with trackball: https://raw.org/code/trackball-rotation-using-quaternions/
                if (self.drag_begin_x - cx)**2 + (self.drag_begin_y - cy)**2 < r**2:
                    px, py = -(self.drag_begin_x - cx)/r, (self.drag_begin_y - cy)/r
                    px2y2 = px**2 + py**2
                    # p = np.array([px, py, np.sqrt(max(1 - px2y2, 0))])
                    p = np.array([px, py, np.sqrt(max(1 - px2y2, 0.25/px2y2))])
                    p /= np.linalg.norm(p)

                    qx, qy = -(self.cursor_x - cx)/r, (self.cursor_y - cy)/r
                    qx2y2 = qx**2 + qy**2
                    # q = np.array([qx, qy, np.sqrt(max(1 - qx2y2, 0))])
                    q = np.array([qx, qy, np.sqrt(max(1 - qx2y2, 0.25/qx2y2))])
                    q /= np.linalg.norm(q)

                    if self.verbose:
                        print(f"Trackball from {p} to {q}")
                    self.cam.trackball(p, q, rot_begin=self.cam_rot_begin)

                # rotate around Z axis
                else:
                    xy_begin = np.array([cursor_x_prev - cx, cursor_y_prev - cy])
                    xy_end = np.array([self.cursor_x - cx, self.cursor_y - cy])
                    angle_z = np.arctan2(xy_end[1], xy_end[0]) - np.arctan2(xy_begin[1], xy_begin[0])
                    self.cam.orbit_z(angle_z)
            
            # drag with middle button
            elif self.drag_button in [dpg.mvMouseButton_Middle, dpg.mvMouseButton_Right]:
                # Pan in X-Y plane
                self.cam.pan(dx=self.cursor_x - cursor_x_prev, dy=self.cursor_y - cursor_y_prev)
            self.need_update = True
        
        self.cursor_x_prev = self.cursor_x
        self.cursor_y_prev = self.cursor_y

    def callback_mouse_button_down(self, sender, app_data):
        if not dpg.is_item_hovered("viewer_tag") or self.scene is None:
            return
        if self.drag_button != app_data[0]:
            self.drag_begin_x = self.cursor_x
            self.drag_begin_y = self.cursor_y
            self.drag_button = app_data[0]
            self.cam_rot_begin = self.cam.rot
    
    def callback_mouse_release(self, sender, app_data):
        self.drag_begin_x = None
        self.drag_begin_y = None
        self.drag_button = None
        self.cursor_x_prev = None
        self.cursor_y_prev = None
        self.cam_rot_begin = None

    def callback_mouse_wheel(self, sender, app_data):
        if dpg.is_item_hovered("viewer_tag") and self.scene is not None:
            self.cam.scale(app_data)
            self.need_update = True
        else:
            if dpg.is_item_hovered(f'slider_level'):
                if app_data > 0:
                    self.prev_item(f'button_left_level_{self.active_level}', None)
                else:
                    self.next_item(f'button_right_level_{self.active_level}', None)

            for level in self.items_levels.keys():
                if dpg.is_item_hovered(f'combo_level_{level}'):
                    self.set_level(f'selectable_level_{level}', None)
            
                    if app_data > 0:
                        self.prev_item(f'button_left_level_{self.active_level}', None)
                    else:
                        self.next_item(f'button_right_level_{self.active_level}', None)

    def callback_key_press(self, sender, app_data):
        if self.scene is None:
            return
        step = 30
        if sender == '_mvKey_W':
            self.cam.pan(dz=step)
        elif sender == '_mvKey_S':
            self.cam.pan(dz=-step)
        elif sender == '_mvKey_A':
            self.cam.pan(dx=step)
        elif sender == '_mvKey_D':
            self.cam.pan(dx=-step)
        elif sender == '_mvKey_E':
            self.cam.pan(dy=step)
        elif sender == '_mvKey_Q':
            self.cam.pan(dy=-step)
        elif sender == '_mvKey_R':
            self.cam.reset()
        elif sender == '_mvKey_Open_Brace':
            self.light_intensity /= 2
        elif sender == '_mvKey_Close_Brace':
            self.light_intensity *= 2

        self.need_update = True

    def run(self):
        self.define_gui()
        dpg.set_global_font_scale(self.scale)
        dpg.create_viewport(title='Multi-Dimension Viewer', width=self.width, height=self.height + self.status_height, resizable=True)
        dpg.setup_dearpygui()
        dpg.show_viewport()
        self.need_update = True

        while dpg.is_dearpygui_running():
            if self.width != dpg.get_viewport_width() or self.height != dpg.get_viewport_height():
                self.width = dpg.get_viewport_width()
                self.height = dpg.get_viewport_height()
                self.resize_windows()

            self.update_viewer()
            dpg.render_dearpygui_frame()
        dpg.destroy_context()
        if self.mesh_render_process is not None:
            self.render_input_queue.put(None)
            self.mesh_render_process.join()
    
    def update_viewer(self):
        if not self.need_update:
            return
        try:
            path = self.get_absolate_path(self.selected_idx_levels, self.items_levels, len(self.selected_idx_levels)-1)
            if path.name == '-':
                dpg.set_value("texture_tag", np.zeros([self.height, self.width, 4]))
                dpg.configure_item("image_tag", show=True)
                dpg.configure_item("text_field_tag", show=False)
                self.need_update = False
                return
            
            suffix = path.suffix[1:].lower()
            if suffix in self.types_image:
                self.update_status_text('image')
                dpg.configure_item("image_tag", show=True)
                dpg.configure_item("text_field_tag", show=False)
                if path in self.prefetch_cache:
                    img = self.prefetch_cache[path]
                else:
                    img = self.load_image(path)
            elif suffix in self.types_mesh:
                self.update_status_text('mesh')
                dpg.configure_item("image_tag", show=True)
                dpg.configure_item("text_field_tag", show=False)
                if self.mesh_render_process is None:
                    self.mesh_render_process = mp.Process(target=self.mesh_render_loop)
                    self.mesh_render_process.start()

                if self.scene is None:
                    if path in self.prefetch_cache:
                        self.scene = self.prefetch_cache[path]
                    else:
                        self.scene = self.load_scene(path)
                        self.prefetch_cache[path] = self.scene
                    self.camera_node = [node for node in self.scene.nodes if isinstance(node.camera, pyrender.camera.Camera)][0]
                    self.light_node = [node for node in self.scene.nodes if isinstance(node.light, pyrender.light.Light)][0]

                self.scene.set_pose(self.camera_node, self.cam.pose)
                self.scene.set_pose(self.light_node, self.cam.pose)
                self.light_node.light.intensity = self.light_intensity

                if self.verbose:
                    print(f"Render scene with camera pose:")
                    print(self.cam.pose)

                self.render_input_queue.put(self.scene)
                color, depth = self.render_output_queue.get()
                fg_mask = (np.ones_like(depth) * 255)[..., None]
                img = np.concatenate([color, fg_mask], axis=2)
            elif suffix in self.types_container:
                self.update_status_text('container')
                text = ""
                if suffix == 'npz':
                    with np.load(path) as npz:
                        keys = npz.files
                        if len(keys) > 0:
                            for key in keys:
                                text += f"{key}, {npz[key].shape}, {npz[key].dtype}\n"
                else:
                    text = f"Unsupported container type: {str(path)}"
                dpg.set_value("text_field_tag", text)
                dpg.configure_item("image_tag", show=False)
                dpg.configure_item("text_field_tag", show=True)
                img = np.zeros([self.height, self.width, 4])  # We still need to update the texture
            #  elif suffix in self.types_txt:
            elif self.is_text_file(path):
                self.update_status_text('text')
                with open(path, 'r') as f:
                    text = f.read()
                dpg.set_value("text_field_tag", text)
                dpg.configure_item("image_tag", show=False)
                dpg.configure_item("text_field_tag", show=True)
                img = np.zeros([self.height, self.width, 4])  # We still need to update the texture
            else:
                self.update_status_text(None)
                # show the file path as text
                dpg.set_value("text_field_tag", f"Unsupported file type: {str(path)}")
                dpg.configure_item("image_tag", show=False)
                dpg.configure_item("text_field_tag", show=True)
                img = np.zeros([self.height, self.width, 4])
            
            self.need_update = False
            self.io_busy = False
            if self.need_prefetch:
                threading.Thread(target=self.prefetch_thread).start()
        except Exception as e:
            print(f"Error: {e}")
            img = np.zeros([self.height, self.width, 4])
            self.need_update = True
            self.io_busy = True

        # pad
        img = img.astype(np.float32) / 255
        diff_height = (self.height - img.shape[0])
        pad_top = diff_height // 2
        pad_bottom = diff_height - pad_top
        diff_width = (self.width - img.shape[1])
        pad_left = diff_width // 2
        pad_right = diff_width - pad_left
        img = np.pad(img, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode='constant', constant_values=0)
        assert img.shape[0] == self.height and img.shape[1] == self.width and img.shape[2] == 4, f"Image shape: {img.shape}"
        dpg.set_value("texture_tag", img)
        if self.verbose:
            print(f"Updated texture with image shape: {img.shape}")
    
    def update_status_text(self, type):
        if type == 'mesh':
            dpg.set_value("status_text_tag", " Mesh | Move: WASDQE | Reset: R | Light Intensity: [ ]")
        elif type == 'image':
            dpg.set_value("status_text_tag", " Image |")
        elif type == 'container':
            dpg.set_value("status_text_tag", " Container |")
        elif type == 'Text':
            dpg.set_value("status_text_tag", " Text |")
        else:
            dpg.set_value("status_text_tag", "")

    def prefetch_thread(self):
        if self.verbose:
            print("Prefetch thread started")
        # prevent prefetching while loading thec current file
        if self.io_busy:
            return
        if len(self.items_levels[self.active_level]) == 0:
            self.need_prefetch = False
            return
        idx = self.selected_idx_levels[self.active_level]

        if idx < len(self.items_levels[self.active_level]) - 1:
            items_levels_tmp = self.items_levels.copy()
            selected_idx_levels_tmp = self.selected_idx_levels.copy()
            selected_idx_levels_tmp[self.active_level] = idx + 1
            self.update_items_under_level(selected_idx_levels_tmp, items_levels_tmp, self.active_level, update_widgets=False)

            next_path = self.get_absolate_path(selected_idx_levels_tmp, items_levels_tmp, len(selected_idx_levels_tmp)-1)
            if next_path not in self.prefetch_cache:
                suffix = next_path.suffix[1:].lower()
                if suffix in self.types_image:
                    self.prefetch_cache[next_path] = self.load_image(next_path)
                elif suffix in self.types_mesh:
                    scene = self.load_scene(next_path)
                    self.prefetch_cache[next_path] = scene
                if self.verbose:
                    print(f"Prefetch next path: {next_path}")
    
        if idx > 0:
            items_levels_tmp = self.items_levels.copy()
            selected_idx_levels_tmp = self.selected_idx_levels.copy()
            selected_idx_levels_tmp[self.active_level] = idx - 1
            self.update_items_under_level(selected_idx_levels_tmp, items_levels_tmp, self.active_level, update_widgets=False)

            prev_path = self.get_absolate_path(selected_idx_levels_tmp, items_levels_tmp, len(selected_idx_levels_tmp)-1)
            if prev_path not in self.prefetch_cache:
                suffix = prev_path.suffix[1:].lower()
                if suffix in self.types_image:
                    self.prefetch_cache[prev_path] = self.load_image(prev_path)
                elif suffix in self.types_mesh:
                    scene = self.load_scene(prev_path)
                    self.prefetch_cache[prev_path] = scene
                if self.verbose:
                    print(f"Prefetch previous path: {prev_path}")

        self.need_prefetch = False
    
    def is_text_file(self, file_path, block_size=512):
        """
        Check if a file is a text file by reading a portion of its content.

        :param file_path: Path to the file to be checked.
        :param block_size: Number of bytes to read from the file for checking.
        :return: True if the file is a text file, False otherwise.
        """
        try:
            with open(file_path, 'rb') as file:
                block = file.read(block_size)
                if b'\0' in block:
                    return False
                text_characters = bytearray({7, 8, 9, 10, 12, 13, 27} | set(range(0x20, 0x100)) - {0x7f})
                return all(byte in text_characters for byte in block)
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            return False

    def load_image(self, path, resample=Image.BILINEAR):
        if self.verbose:
            print(f"Load image: {path}")

        suffix = path.suffix[1:].lower()
        if suffix in ['dsp5', 'flo5']:
            if suffix == 'dsp5':
                with h5py.File(path, "r") as f:
                    if "disparity" not in f.keys():
                        raise IOError(f"File {path} does not have a 'disparity' key. Is this a valid dsp5 file?")
                    disparity = f["disparity"][()].astype(np.float32)
                disparity[np.isnan(disparity)] = 0
                img = disparity
            elif suffix == 'flo5':
                with h5py.File(path, "r") as f:
                    if "flow" not in f.keys():
                        raise IOError(f"File {path} does not have a 'flow' key. Is this a valid flo5 file?")
                    flow = f["flow"][()].astype(np.float32)

                # resize here to avoid unnecessary slow cocnversion from flow to image
                flow_h, flow_w = flow.shape[:2]
                scale = min(self.height / flow_h, self.width / flow_w)
                flow = cv2.resize(flow, (int(flow_w * scale), int(flow_h * scale)), interpolation=cv2.INTER_LINEAR)

                img = flow_to_image(flow)
            img = Image.fromarray(img)
        else:
            img = Image.open(path)

        if img.mode == 'RGB':  # RGB
            img = np.array(img.convert('RGBA'))
        elif img.mode == 'I;16':  # 16-bit grayscale
            img = np.array(img, dtype=np.uint16)
            if self.rescale_depth_map:
                img = img / img.max()  # for visualization
            else:
                img = img / 65535.0  # Normalize to [0, 1]
            img = (img * 255).astype(np.uint8)
        elif img.mode == 'L':  # 8-bit grayscale
            img = np.array(img, dtype=np.uint8)
            img = img / img.max() * 255
        elif img.mode == 'F':  # 32-bit float grayscale
            img = np.array(img, dtype=np.float32)
            img = (img / (img.max() + 1e-6) * 255).astype(np.uint8)
        elif img.mode == '1':  # 1-bit binary
            img = np.array(img, dtype=np.uint8) * 255
        else:
            # raise ValueError(f"Unsupported image mode: {img.mode}")
            print(f"Unlisted image mode: {img.mode}")
            img = np.array(img)
        
        img_h, img_w = img.shape[:2]
        scale = min(self.height / img_h, self.width / img_w)
        img = cv2.resize(img, (int(img_w * scale), int(img_h * scale)), interpolation=cv2.INTER_LINEAR)

        if img.ndim == 2:
            img = np.repeat(img[:, :, np.newaxis], 4, axis=2)
        return img

    def mesh_render_loop(self):
        r = pyrender.OffscreenRenderer(self.width, self.height)  # Initialize in the process
        while True:
            scene = self.render_input_queue.get()
            if scene is None:
                break

            # Process the task
            color, depth = r.render(scene)
            self.render_output_queue.put([color, depth])

    def import_trimesh_to_pyrender(self, mesh):
        if isinstance(mesh, trimesh.points.PointCloud):
            return pyrender.Mesh.from_points(mesh.vertices, colors=mesh.colors)
        else:
            return pyrender.Mesh.from_trimesh(mesh)
    
    def load_scene(self, file_path: Path):
        scene = pyrender.Scene(ambient_light=[self.ambient_light]*3)
        mesh = trimesh.load(file_path)
        if Path(file_path).suffix == ".glb":
            for k in mesh.geometry.keys():
                mesh_k = mesh.geometry[k]
                scene.add(self.import_trimesh_to_pyrender(mesh_k))
        else:
            scene.add(self.import_trimesh_to_pyrender(mesh))
        
        camera = pyrender.PerspectiveCamera(yfov=np.radians(self.cam.fovy))
        scene.add(camera, pose=self.cam.pose)
        light = pyrender.DirectionalLight(color=np.ones(3), intensity=self.light_intensity)
        scene.add(light, pose=self.cam.pose)
        return scene

    def update_button_states(self, level):
        if len(self.items_levels[level]) == 0:
            left_enabled = False
            right_enabled = False
        else:
            idx = self.selected_idx_levels[level]
            left_enabled = idx > 0
            right_enabled = idx < len(self.items_levels[level]) - 1
        left_label = "<" if left_enabled else " "
        right_label = ">" if right_enabled else " "
        dpg.configure_item(f'button_left_level_{level}', label=left_label)
        dpg.configure_item(f'button_right_level_{level}', label=right_label)

        dpg.bind_item_theme(f'button_left_level_{level}', self.button_theme)
        dpg.bind_item_theme(f'button_right_level_{level}', self.button_theme)


def main():
    cfg = tyro.cli(MultiDimensionViewerConfig)
    app = MultiDimensionViewer(cfg)
    app.run()

if __name__ == '__main__':
    main()
