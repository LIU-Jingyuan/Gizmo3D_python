import numpy as np
import pyvista as pv

def _init_scene(self):
    # Set centers. Add bunny and ground.

    # Load the bunny from local OBJ file
    bunny = pv.read("./data/bunny.obj")
    self.obj_actor = self.plotter.add_mesh(
        bunny,
        color="white",
        smooth_shading=True,
        name="bunny",
    )
    self.obj_center = np.array(bunny.center)

    # Create a ground plane at the bunny's lowest y
    # bounds = (xmin, xmax, ymin, ymax, zmin, zmax)
    b = bunny.bounds
    y_min = b[2]
    x_span = b[1] - b[0]
    z_span = b[5] - b[4]
    pad = max(x_span, z_span) * 5.0

    ground = pv.Plane(
        center=(bunny.center[0], y_min, bunny.center[2]),
        direction=(0, 1, 0),
        i_size=pad,
        j_size=pad,
        i_resolution=25,
        j_resolution=25,
    )

    self.ground_actor = self.plotter.add_mesh(
        ground,
        color="lightgrey",
        show_edges=True,
        name="ground",
    )
    self.ground_center = np.array(ground.center)

    # Camera
    self.plotter.camera_position = "xy"
    self.plotter.reset_camera()
    self.plotter.camera.elevation = 20
    self.plotter.camera.azimuth = -30
    self.plotter.camera.zoom(1.5)
