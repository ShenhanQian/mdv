from pathlib import Path
import numpy as np
from PIL import Image
from dataclasses import dataclass
import tyro
from tyro.conf import Positional
import dearpygui.dearpygui as dpg


@dataclass
class MultiDimensionViewerConfig:
    root_folder: Positional[Path]
    width: int = 1000
    height: int = 1000
    scale: float = 1.0

class MultiDimensionViewer(object):
    def __init__(self, cfg: MultiDimensionViewerConfig):
        self.root_folder = cfg.root_folder
        self.scale = cfg.scale
        self.width = int(cfg.width * self.scale)
        self.height = int(cfg.height * self.scale)

        self.need_update = False

        self.rows = sorted([x.name for x in self.root_folder.iterdir() if x.is_dir()])
        self.selected_row = '-'

        self.cols = []
        self.selected_col_idx = 0
        self.selected_col = '-'

    def define_gui(self):
        dpg.create_context()

        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(width=self.width, height=self.height, default_value=np.zeros([self.height, self.width, 3]), format=dpg.mvFormat_Float_rgb, tag="texture_tag")

        # viewer window
        with dpg.window(label="Viewer", pos=[0, 0], tag='viewer_tag', width=self.width, height=self.height, no_title_bar=True, no_move=True, no_bring_to_front_on_focus=True):
            dpg.add_image("texture_tag", tag='image_tag', width=self.width, height=self.height)

        # navigator window
        with dpg.window(label="Navigator", tag='navigator_tag', pos=[0, 0], autosize=True):

            # row switch
            with dpg.group(horizontal=True):
                dpg.add_combo(['-'] + self.rows, default_value=self.selected_row, label="rows", height_mode=dpg.mvComboHeight_Large, callback=self.set_row, tag='combo_row')
                dpg.add_button(label="Button", callback=self.prev_row, arrow=True, direction=dpg.mvDir_Up)
                dpg.add_button(label="Button", callback=self.next_row, arrow=True, direction=dpg.mvDir_Down)
            
            # column switch
            with dpg.group(horizontal=True):
                dpg.add_slider_int(label="cols", max_value=max(len(self.cols)-1, 0), callback=self.set_col, tag='slider_col')
                dpg.add_button(label="Button", callback=self.prev_col, arrow=True, direction=dpg.mvDir_Left)
                dpg.add_button(label="Button", callback=self.next_col, arrow=True, direction=dpg.mvDir_Right)
        
        # key press handlers
        with dpg.handler_registry():
            dpg.add_key_press_handler(dpg.mvKey_Up, callback=self.prev_row)
            dpg.add_key_press_handler(dpg.mvKey_Down, callback=self.next_row)
            dpg.add_key_press_handler(dpg.mvKey_Left, callback=self.prev_col)
            dpg.add_key_press_handler(dpg.mvKey_Right, callback=self.next_col)
            dpg.add_key_press_handler(dpg.mvKey_Home, callback=self.set_row)
            dpg.add_key_press_handler(dpg.mvKey_End, callback=self.set_row)
            dpg.add_key_press_handler(dpg.mvKey_Escape, callback=lambda: dpg.stop_dearpygui())
            dpg.add_mouse_wheel_handler(callback=self.on_mouse_wheel)

        # theme
        with dpg.theme() as theme_no_padding:
            with dpg.theme_component(dpg.mvAll):
                # set all padding to 0 to avoid scroll bar
                dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 0, 0, category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 0, 0, category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_CellPadding, 0, 0, category=dpg.mvThemeCat_Core)
        dpg.bind_item_theme("viewer_tag", theme_no_padding)

    def on_mouse_wheel(self, sender, app_data):
        if dpg.is_item_hovered("combo_row"):
            if app_data > 0:
                self.prev_row(None, None)
            else:
                self.next_row(None, None)
        elif dpg.is_item_hovered("slider_col"):
            if app_data > 0:
                self.prev_col(None, None)
            else:
                self.next_col(None, None)

    def set_row(self, sender, data):
        if len(self.rows) < 1:
            return

        if data == dpg.mvKey_Home:
            data = self.rows[0]
        elif data == dpg.mvKey_End:   
            data = self.rows[-1]

        self.selected_row = data
        if data == '-':
            self.cols = []
        else:
            self.cols = sorted([x.name for x in (self.root_folder / self.selected_row).iterdir()])
        self.selected_col_idx = 0
        self.selected_col = self.cols[self.selected_col_idx] if len(self.cols) > 0 else '-'

        dpg.set_value("combo_row", value=self.selected_row)
        dpg.configure_item("slider_col", max_value=max(len(self.cols)-1, 0), default_value=self.selected_col_idx)
        self.need_update = True

    def prev_row(self, sender, data):
        if self.selected_row == '-':
            self.selected_row = self.rows[-1]
        else:
            idx = self.rows.index(self.selected_row)
            if idx > 0:
                self.selected_row = self.rows[idx-1]
            # else:
            #     self.selected_row = self.rows[-1]
        dpg.set_value("combo_row", value=self.selected_row)
        self.set_row(None, self.selected_row)

    def next_row(self, sender, data):
        if self.selected_row == '-':
            self.selected_row = self.rows[0]
        else:
            idx = self.rows.index(self.selected_row)
            if idx < len(self.rows)-1:
                self.selected_row = self.rows[idx+1]
            # else:
            #     self.selected_row = self.rows[0]
        dpg.set_value("combo_row", value=self.selected_row)
        self.set_row(None, self.selected_row)

    def set_col(self, sender, data):
        if len(self.cols) < 1:
            return

        self.selected_col_idx = data
        dpg.set_value("slider_col", value=self.selected_col_idx)
        self.selected_col = self.cols[self.selected_col_idx]
        self.need_update = True
                
    def prev_col(self, sender, data):
        if len(self.cols) > 1:
            if self.selected_col_idx > 0:
                self.selected_col_idx -= 1
            self.selected_col = self.cols[self.selected_col_idx]
            dpg.set_value("slider_col", value=self.selected_col_idx)
            self.set_col(None, self.selected_col_idx)

    def next_col(self, sender, data):
        if len(self.cols) > 1:
            if self.selected_col_idx < len(self.cols)-1:
                self.selected_col_idx += 1
            self.selected_col = self.cols[self.selected_col_idx]
            dpg.set_value("slider_col", value=self.selected_col_idx)
            self.set_col(None, self.selected_col_idx)

    def resize_windows(self):
        dpg.configure_item('viewer_tag', width=self.width, height=self.height)
        dpg.configure_item('navigator_tag', pos=[0, 0])

        dpg.delete_item('texture_tag')
        dpg.delete_item('image_tag')
        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(width=self.width, height=self.height, default_value=np.zeros([self.height, self.width, 3]), format=dpg.mvFormat_Float_rgb, tag="texture_tag")
        dpg.add_image("texture_tag", tag='image_tag', parent='viewer_tag')
        self.need_update = True

    def run(self):
        self.define_gui()
        dpg.set_global_font_scale(self.scale)
        dpg.create_viewport(title='Multi-Dimension Viewer', width=self.width, height=self.height, resizable=True)
        dpg.setup_dearpygui()
        dpg.show_viewport()

        while dpg.is_dearpygui_running():
            if self.width != dpg.get_viewport_width() or self.height != dpg.get_viewport_height():
                self.width = dpg.get_viewport_width()
                self.height = dpg.get_viewport_height()
                self.resize_windows()

            if self.need_update:
                self.update_viewer()
                self.need_update = False
            dpg.render_dearpygui_frame()
        dpg.destroy_context()
    
    def update_viewer(self):
        if self.selected_row != '-' and self.selected_col != '-':
            path = self.root_folder / self.selected_row / self.selected_col
            if 'jpg' not in str(path).lower() and 'png' not in str(path).lower():
                return
          
            # directly load as float32
            img = self.load_image(path)
            img = img.astype(np.float32) / 255
            diff_height = (self.height - img.shape[0])
            pad_top = diff_height // 2
            pad_bottom = diff_height - pad_top
            diff_width = (self.width - img.shape[1])
            pad_left = diff_width // 2
            pad_right = diff_width - pad_left
            img = np.pad(img, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode='constant', constant_values=0)

            dpg.set_value("texture_tag", img)
        else:
            dpg.set_value("texture_tag", np.zeros([self.height, self.width, 3]))

    def load_image(self, path, resample=Image.BILINEAR):
        img = Image.open(path)
        scale = min(self.height / img.height, self.width / img.width)
        img = img.resize((int(img.width * scale), int(img.height * scale)), resample)
        img = np.asarray(img)
        if img.ndim == 2:
            img = np.repeat(img[:, :, np.newaxis], 3, axis=2)
        return img


def main():
    cfg = tyro.cli(MultiDimensionViewerConfig)
    app = MultiDimensionViewer(cfg)
    app.run()

if __name__ == '__main__':
    main()
