import tkinter.filedialog as fd
from threading import Thread, Event

import numpy as np
import open3d as o3d
from open3d.visualization import gui
import time


class AppWindow:
    MENU_OPEN = 1
    MENU_QUIT = 3

    def __init__(self):
        # Create window
        self.window = gui.Application.instance.create_window("Scene Visualization", 1024, 768)
        w = self.window

        # Menu
        if gui.Application.instance.menubar is None:
            file_menu = gui.Menu()
            file_menu.add_item("Open...", AppWindow.MENU_OPEN)
            file_menu.add_separator()
            file_menu.add_item("Quit", AppWindow.MENU_QUIT)
            menu = gui.Menu()
            menu.add_menu("File", file_menu)
            gui.Application.instance.menubar = menu

        w.set_on_menu_item_activated(AppWindow.MENU_OPEN, self._on_menu_open)
        w.set_on_key(self._on_key)

        # Scene
        self.scene_widget = gui.SceneWidget()
        self.window.add_child(self.scene_widget)
        self.scene_widget.scene = o3d.visualization.rendering.Open3DScene(self.window.renderer)
        self.scene_widget.scene.scene.set_sun_light([-1, -1, -1], [1, 1, 1], 20000)
        self.scene_widget.scene.scene.enable_sun_light(True)

        # Rotation control
        self.is_spinning = False
        self.spin_event = Event()

    def _on_key(self, event):
        if event.key == gui.KeyName.S and event.type == gui.KeyEvent.Type.UP:
            if self.is_spinning:
                self.is_spinning = False
                self.spin_event.clear()
            else:
                self.is_spinning = True
                self.spin_event.set()
                Thread(target=self._spin_camera, daemon=True).start()

    def _spin_camera(self):
        radius = 0.01
        speed = 0.2
        center = np.array([-1.5, 0, 0])
        up = [0, 0, 1]

        while self.spin_event.is_set():
            for angle in np.arange(0, 360, speed):
                if not self.spin_event.is_set():
                    return
                theta = np.radians(angle)
                eye = [radius * np.sin(theta), radius * np.cos(theta), 0] + center
                self.scene_widget.scene.camera.look_at(center, eye, up)
                gui.Application.instance.post_to_main_thread(self.window, lambda: self.scene_widget.force_redraw())
                time.sleep(0.01)

    def _on_menu_open(self):
        def get_model_path():
            filename = fd.askopenfilename(title="Choose a file",
                                          filetypes=[("3D Models", "*.ply;*.stl;*.obj;*.glb"),
                                                     ("All files", "*.*")])
            if filename:
                self.load(filename)

        Thread(target=get_model_path, daemon=True).start()

    def load(self, filename):
        self.scene_widget.scene.clear_geometry()
        model = o3d.io.read_triangle_model(filename, True)

        for mesh in model.meshes:
            mat = model.materials[mesh.material_idx]
            mat.shader = "defaultUnlit"
            self.scene_widget.scene.add_geometry(mesh.mesh_name, mesh.mesh, mat)

        bbox = o3d.geometry.AxisAlignedBoundingBox([-1, -1, -1], [1, 1, 1])
        self.scene_widget.setup_camera(90, bbox, [0, 0, 0])
        gui.Application.instance.post_to_main_thread(self.window, lambda: self.scene_widget.force_redraw())

if __name__ == "__main__":
    gui.Application.instance.initialize()
    AppWindow()
    gui.Application.instance.run()
