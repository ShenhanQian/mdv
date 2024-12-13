from pathlib import Path
from dataclasses import dataclass, field
from typing import Annotated, Literal
import tyro
from tyro.conf import Positional
import numpy as np
from PIL import Image
import pyrender
import trimesh
import multiprocessing as mp
import threading
import dearpygui.dearpygui as dpg
from .utils.camera import OrbitCamera


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
    types: Annotated[Literal["image", "mesh"], tyro.conf.arg(aliases=["-t"])] = field(default_factory=lambda: ["image", "mesh"])
    """Types of files to be displayed"""
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
    light_intensity: float = 10.
    """Light intensity"""
    

class MultiDimensionViewer(object):
    def __init__(self, cfg: MultiDimensionViewerConfig):
        self.root_folder = cfg.root_folder
        self.scale = cfg.scale
        self.width = int(cfg.width * self.scale)
        self.height = int(cfg.height * self.scale)
        self.rescale_depth_map = cfg.rescale_depth_map
        self.verbose = cfg.verbose

        # files types
        self.types_image = ['jpg', 'jpeg', 'png']
        self.types_mesh = ['obj', 'glb']
        self.supported_types = []
        if 'image' in cfg.types:
            self.supported_types += self.types_image
        if 'mesh' in cfg.types:
            self.supported_types += self.types_mesh

        # styles
        self.selectable_width = 12 * self.scale

        # database
        self.active_level = 0
        self.items_levels = {}
        if self.root_folder.is_file():
            raise NotImplementedError("File is not supported yet.")
        elif self.root_folder.is_dir():
            items = sorted([x.name for x in self.root_folder.iterdir() if x.is_dir() or x.suffix[1:] in self.supported_types])
            self.items_levels.update({0: items})
        self.selected_per_level = {self.active_level: self.items_levels[self.active_level][0]}
        self.update_items_under_level(self.selected_per_level, self.items_levels, self.active_level, update_widgets=False)
        
        # prefetch
        self.prefetch_cache = {}
        self.need_prefetch = True
        self.io_busy = False
        # self.prefetch_thread = threading.Thread(target=self.prefetch_loop).start()

        # mesh rendering
        self.render_input_queue = mp.Queue()
        self.render_output_queue = mp.Queue()
        self.mesh_render_process = None
        self.scene = None
        self.cam = OrbitCamera(self.width, self.height, r=cfg.cam_radius, fovy=cfg.cam_fovy, convention=cfg.cam_convention)
        self.light_intensity = cfg.light_intensity

        # buffers for mouse interaction
        self.cursor_x = None
        self.cursor_y = None
        self.cursor_x_prev = None
        self.cursor_y_prev = None
        self.drag_begin_x = None
        self.drag_begin_y = None
        self.drag_button = None

    def get_absolate_path(self, selected_per_level, level):
        path = self.root_folder
        for i in range(level+1):
            if selected_per_level[i] is None:
                return Path('-')
            path = path / selected_per_level[i]
        return path
    
    def update_items_under_level(self, selected_per_level, items_levels, level, update_widgets=True):
        while self.get_absolate_path(selected_per_level, level).is_dir():
            level += 1

            base_path = self.get_absolate_path(selected_per_level, level-1)

            items = sorted([x.name for x in base_path.iterdir() if x.is_dir() or x.suffix[1:] in self.supported_types])
            items_levels.update({level: items})

            if level in selected_per_level and selected_per_level[level] in items:
                selected = selected_per_level[level]
            else:
                if len(items) > 0:
                    selected = items[0]
                else:
                    selected = '-'
            selected_per_level.update({level: selected})
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
        
        for l in range(level+1, len(items_levels)):
            if l in items_levels:
                items_levels.pop(l)
            if l in selected_per_level:
                selected_per_level.pop(l)
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
            dpg.add_image("texture_tag", tag='image_tag', width=self.width, height=self.height)
        dpg.bind_item_theme("viewer_tag", theme_no_padding)

        # navigator window
        with dpg.window(label="Navigator", tag='navigator_tag', pos=[0, 0], autosize=True, no_close=True):
            for level, items in self.items_levels.items():
                with dpg.group(horizontal=True, tag=f'group_level_{level}'):
                    dpg.add_combo(items, default_value=self.selected_per_level[level], height_mode=dpg.mvComboHeight_Regular, callback=lambda sender, data: self.set_item(sender, data), tag=f'combo_level_{level}')

                    dpg.add_button(label="<", callback=lambda sender, data: self.prev_item(sender, data), tag=f'button_left_level_{level}')
                    dpg.add_button(label=">", callback=lambda sender, data: self.next_item(sender, data), tag=f'button_right_level_{level}')
                    self.update_button_states(level)

                    dpg.add_selectable(width=self.selectable_width, default_value=level==self.active_level, tag=f'selectable_level_{level}', callback=lambda sender, data: self.set_level(sender, data))
                    dpg.bind_item_theme(f'selectable_level_{level}', self.selectable_theme)
            dpg.add_slider_int(default_value=0, min_value=0, max_value=len(self.items_levels[self.active_level])-1, tag=f'slider_level', callback=lambda sender, data: self.set_item(f"slider_{self.active_level}", self.items_levels[self.active_level][data]))

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

        if len(self.items_levels[self.active_level]) > 0:
            max_value = len(self.items_levels[self.active_level]) - 1
            slider_value = self.items_levels[self.active_level].index(self.selected_per_level[self.active_level])
        else:
            max_value = 0
            slider_value = 0
        dpg.configure_item(f'slider_level', max_value=max_value)
        dpg.set_value(f'slider_level', slider_value)
        if self.verbose:
            print(f"Update slider with max value: {len(self.items_levels[self.active_level])-1}")

        self.need_prefetch = True
        if self.need_prefetch:
            threading.Thread(target=self.prefetch_loop).start()

    def prev_level(self):
        if self.active_level > 0:
            self.set_level(f'selectable_level_{self.active_level-1}', None)

    def next_level(self):
        if self.active_level < len(self.items_levels) - 1:
            self.set_level(f'selectable_level_{self.active_level+1}', None)
    
    def set_item(self, sender, data):
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
        self.selected_per_level[level] = data
        dpg.set_value(f'combo_level_{level}', data)
        self.need_prefetch = True
        if self.verbose:
            print(f"Set level {level} with item: {data}")

        self.update_items_under_level(self.selected_per_level, self.items_levels, level)
        self.update_button_states(level)
        self.scene = None
        self.need_update = True

    def prev_item(self, sender, data):
        if data == dpg.mvKey_Left:
            level = self.active_level
            sender = f'button_left_level_{level}'
        else:
            level = int(sender.split('_')[-1])

        idx = self.items_levels[level].index(self.selected_per_level[level])
        if idx > 0:
            self.set_item(sender, self.items_levels[level][idx-1])
            if level == self.active_level:
                dpg.set_value(f'slider_level', idx-1)

    def next_item(self, sender, data):
        if data == dpg.mvKey_Right:
            level = self.active_level
            sender = f'button_right_level_{level}'
        else:
            level = int(sender.split('_')[-1])

        idx = self.items_levels[level].index(self.selected_per_level[level])
        if idx < len(self.items_levels[level]) - 1:
            self.set_item(sender, self.items_levels[level][idx+1])
            if level == self.active_level:
                dpg.set_value(f'slider_level', idx+1)

    def resize_windows(self):
        dpg.configure_item('viewer_tag', width=self.width, height=self.height)
        dpg.configure_item('navigator_tag', pos=[0, 0])

        dpg.delete_item('texture_tag')
        dpg.delete_item('image_tag')
        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(width=self.width, height=self.height, default_value=np.zeros([self.height, self.width, 4]), format=dpg.mvFormat_Float_rgba, tag="texture_tag")
        dpg.add_image("texture_tag", tag='image_tag', parent='viewer_tag')
        self.prefetch_cache = {}
        self.need_update = True

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
            for level in self.items_levels.keys():
                if dpg.is_item_hovered(f'combo_level_{level}'):
                    if app_data > 0:
                        self.prev_item(f'button_left_level_{level}', None)
                    else:
                        self.next_item(f'button_right_level_{level}', None)

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
        dpg.create_viewport(title='Multi-Dimension Viewer', width=self.width, height=self.height, resizable=True)
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
        # try:
        path = self.get_absolate_path(self.selected_per_level, len(self.selected_per_level)-1)
        if path.name == '-':
            dpg.set_value("texture_tag", np.zeros([self.height, self.width, 4]))
            self.need_update = False
            return
        
        self.io_busy = True  # prevent prefetching while loading thec current file
        suffix = path.suffix[1:].lower()
        if suffix in self.types_image:
            if path in self.prefetch_cache:
                img = self.prefetch_cache[path]
            else:
                img = self.load_image(path)
            self.io_busy = False
        elif suffix in self.types_mesh:
            if self.scene is None:
                if path in self.prefetch_cache:
                    self.scene = self.prefetch_cache[path]
                else:
                    self.scene = self.load_scene(path)

                camera = pyrender.PerspectiveCamera(yfov=np.radians(self.cam.fovy))
                self.node_camera = self.scene.add(camera, pose=self.cam.pose)
                self.light = pyrender.DirectionalLight(color=np.ones(3), intensity=self.light_intensity)
                self.node_light = self.scene.add(self.light, pose=self.cam.pose)

                if self.mesh_render_process is None:
                    self.mesh_render_process = mp.Process(target=self.mesh_render_loop)
                    self.mesh_render_process.start()
            else:
                self.scene.set_pose(self.node_camera, self.cam.pose)
                self.scene.set_pose(self.node_light, self.cam.pose)
                self.light.intensity = self.light_intensity

            if self.verbose:
                print(f"Render scene with camera pose:")
                print(self.cam.pose)

            self.render_input_queue.put(self.scene)
            self.io_busy = False
            img = self.render_output_queue.get()
            img = np.concatenate([img, np.ones([img.shape[0], img.shape[1], 1]) * 255], axis=2)
        else:
            img = np.zeros([self.height, self.width, 4])
            if self.verbose:
                raise TypeError(f"Unsupported file type: {path}")
        
        if self.need_prefetch:
            threading.Thread(target=self.prefetch_loop).start()
            
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

        self.need_update = False

    def prefetch_loop(self):
        # prevent prefetching while loading thec current file
        if self.io_busy:
            return
        if len(self.items_levels[self.active_level]) == 0:
            self.need_prefetch = False
            return
        idx = self.items_levels[self.active_level].index(self.selected_per_level[self.active_level])

        if idx < len(self.items_levels[self.active_level]) - 1:
            items_levels_tmp = self.items_levels.copy()
            selected_per_level_tmp = self.selected_per_level.copy()
            selected_per_level_tmp[self.active_level] = self.items_levels[self.active_level][idx + 1]
            self.update_items_under_level(selected_per_level_tmp, items_levels_tmp, self.active_level, update_widgets=False)

            next_path = self.get_absolate_path(selected_per_level_tmp, len(selected_per_level_tmp)-1)
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
            selected_per_level_tmp = self.selected_per_level.copy()
            selected_per_level_tmp[self.active_level] = self.items_levels[self.active_level][idx - 1]
            self.update_items_under_level(selected_per_level_tmp, items_levels_tmp, self.active_level, update_widgets=False)

            prev_path = self.get_absolate_path(selected_per_level_tmp, len(selected_per_level_tmp)-1)
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

    def load_image(self, path, resample=Image.BILINEAR):
        img = Image.open(path)
        if self.verbose:
            print(f"Load image: {path}, size: {img.size}, mode: {img.mode}")
        
        # turn RGBA to RGB if needed
        if img.mode == 'RGB':
            img = img.convert('RGBA')
        # Handle 16-bit depth images
        if img.mode == 'I;16':
            img = np.array(img, dtype=np.uint16)
            if self.rescale_depth_map:
                img = img / img.max()  # for visualization
            else:
                img = img / 65535.0  # Normalize to [0, 1]
            img = (img * 255).astype(np.uint8)  # Convert to 8-bit
            img = Image.fromarray(img)

        scale = min(self.height / img.height, self.width / img.width)
        img = img.resize((int(img.width * scale), int(img.height * scale)), resample)
        img = np.asarray(img)
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
            self.render_output_queue.put(color)

    @staticmethod
    def load_scene(file_path: Path):
        scene = pyrender.Scene()
        mesh = trimesh.load(file_path)
        if Path(file_path).suffix == ".glb":
            for k in mesh.geometry.keys():
                mesh_k = mesh.geometry[k]
                scene.add(pyrender.Mesh.from_trimesh(mesh_k))
        else:
            scene.add(pyrender.Mesh.from_trimesh(mesh))
        return scene

    def update_button_states(self, level):
        if self.selected_per_level[level] == '-':
            left_enabled = False
            right_enabled = False
        else:
            idx = self.items_levels[level].index(self.selected_per_level[level])
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
