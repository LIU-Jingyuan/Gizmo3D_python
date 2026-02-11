import numpy as np
import pyvista as pv
from PyQt5.QtCore import QObject, pyqtSignal
from vtkmodules.util.numpy_support import vtk_to_numpy

from transgizmo.arrowTranslationHandle import arrowTranslationHandle
from transgizmo.planeTranslationHandle import planeTranslationHandle
from transgizmo.rotationHandle import rotationHandle
from transgizmo.geometry import (
    generate_ray_from_screen,
    initialize_translation_plane,
    initialize_rotation_plane,
    apply_translation_constraint,
    quaternion_rotation_to,
    quaternion_to_rotation_matrix,
)


class Gizmo(QObject):

    handle_pressed  = pyqtSignal(str, int, int)
    handle_dragged  = pyqtSignal(str, int, int)
    handle_released = pyqtSignal(str)

    def __init__(self, plotter, position=(0, 0, 0), show_debug_vis=True,
                 epsilon=2.5e-8):
        super().__init__()
        self.plotter = plotter
        self.position = np.asarray(position, dtype=float)
        self.epsilon = epsilon

        # Interaction state
        self._active_handle       = None
        self._axis_constraint     = None
        self._click_ray           = None
        self._constraint_plane    = None
        self._intersection_point  = None
        self._target_actor = None

        # Rotation-specific
        self._last_rotation_dir   = None
        self._cumulative_rotation = np.eye(3)   # Cumulative orientation of the object (local → world)

        self.show_debug_vis = show_debug_vis

        # Arrow translation handles
        self.XTrans_handle  = arrowTranslationHandle(plotter, position, "XTrans",  gizmo=self)
        self.YTrans_handle  = arrowTranslationHandle(plotter, position, "YTrans",  gizmo=self)
        self.ZTrans_handle  = arrowTranslationHandle(plotter, position, "ZTrans",  gizmo=self)

        # Plane translation handles
        self.YZTrans_handle = planeTranslationHandle(plotter, position, "YZTrans", gizmo=self)
        self.XZTrans_handle = planeTranslationHandle(plotter, position, "XZTrans", gizmo=self)
        self.XYTrans_handle = planeTranslationHandle(plotter, position, "XYTrans", gizmo=self)

        # Rotation handles
        self.XRot_handle    = rotationHandle(plotter, position, "XRot", gizmo=self)
        self.YRot_handle    = rotationHandle(plotter, position, "YRot", gizmo=self)
        self.ZRot_handle    = rotationHandle(plotter, position, "ZRot", gizmo=self)

        # Flat list for batch operations (translate / rotate all handles)
        self._all_handles = [
            self.XTrans_handle,  self.YTrans_handle,  self.ZTrans_handle,
            self.YZTrans_handle, self.XZTrans_handle, self.XYTrans_handle,
            self.XRot_handle,    self.YRot_handle,    self.ZRot_handle,
        ]

    ### Public helpers
    def set_target(self, actor):
        # whose mesh this gizmo transforms
        self._target_actor = actor

    ### Interaction callbacks (called by handles)
    def _on_handle_pressed(self, handle_name, x, y):
        """ When a handle is clicked on
        build ray, constraint plane, and
        initialise state for translation or rotation dragging."""
        self._active_handle   = handle_name
        self._axis_constraint = handle_name

        # Disable camera interaction
        style = self.plotter.iren.interactor.GetInteractorStyle()
        if style is not None:
            style.OnLeftButtonUp()
            style.SetEnabled(0)

        # screen --> world ray
        self._click_ray = generate_ray_from_screen(
            self.plotter.renderer, x, y, epsilon=self.epsilon,
        )

        # build constraint plane
        if "Rot" in handle_name:
            self._constraint_plane = initialize_rotation_plane(
                self.position, handle_name,
                orientation=self._cumulative_rotation,
                epsilon=self.epsilon,
            )
        else:
            self._constraint_plane = initialize_translation_plane(
                self._click_ray, self.position, handle_name,
                orientation=self._cumulative_rotation,
                epsilon=self.epsilon,
            )

        # ray / plane intersection
        code, pt = self._click_ray.intersects(self._constraint_plane)
        self._intersection_point = pt if code == 1 else None

        # Mode-specific initialisation
        if code == 1:
            if "Rot" in handle_name:
                direction = pt - self.position
                norm = np.linalg.norm(direction)
                self._last_rotation_dir = (
                    direction / norm if norm > self.epsilon else None
                )
            else:
                self._constraint_plane.position = pt.copy()

        if self.show_debug_vis:
            self._add_debug_actors()

        self.handle_pressed.emit(handle_name, x, y)

    def _on_handle_dragged(self, handle_name, x, y):
        """ When a handle is being dragged — cast a new ray each frame,
        intersect it with the stored constraint plane, and move /
        rotate the target + handles accordingly."""
        ray = generate_ray_from_screen(self.plotter.renderer, x, y,
                                       epsilon=self.epsilon)
        code, intersection = ray.intersects(self._constraint_plane)

        if code != 1:
            self.handle_dragged.emit(handle_name, x, y)
            return

        # Store the current ray (used by debug vis)
        self._click_ray = ray

        # Apply the transformation (no render yet)
        if "Rot" in handle_name:
            self._update_rotation(intersection)
        else:
            self._update_translation(intersection)

        # Re-intersect with the (possibly shifted) plane so the debug
        # lime sphere sits exactly on the updated yellow quad.
        _, pt = ray.intersects(self._constraint_plane)
        self._intersection_point = pt

        # Refresh debug overlay (includes render), or just render
        if self.show_debug_vis:
            self._add_debug_actors()
        else:
            self.plotter.render()

        self.handle_dragged.emit(handle_name, x, y)

    def _on_handle_released(self, handle_name):
        """A handle was released — clean up state and re-enable camera."""
        self._active_handle     = None
        self._axis_constraint   = None
        self._last_rotation_dir = None

        if self.show_debug_vis:
            self._remove_debug_actors()

        style = self.plotter.iren.interactor.GetInteractorStyle()
        if style is not None:
            style.SetEnabled(1)

        self.handle_released.emit(handle_name)

    def _update_translation(self, intersection):
        # Compute constrained displacement, move target + handles.
        constrained = apply_translation_constraint(
            self._constraint_plane.position,
            intersection,
            self._axis_constraint,
            orientation=self._cumulative_rotation,
        )
        displacement = constrained - self._constraint_plane.position

        # Move the target mesh
        if self._target_actor is not None:
            self._translate_actor_mesh(self._target_actor, displacement)

        # Move every handle mesh
        for handle in self._all_handles:
            self._translate_actor_mesh(handle.actor, displacement)

        # Keep bookkeeping in sync
        self.position += displacement
        self._constraint_plane.position = constrained.copy()

    def _update_rotation(self, intersection):
        # Compute incremental rotation, rotate target + handles.
        direction = intersection - self.position
        norm = np.linalg.norm(direction)
        if norm < self.epsilon or self._last_rotation_dir is None:
            return
        current_dir = direction / norm

        if np.dot(self._last_rotation_dir, current_dir) > 1.0 - self.epsilon:
            return

        # Incremental quaternion --> 3×3 rotation matrix
        q = quaternion_rotation_to(self._last_rotation_dir, current_dir, epsilon=self.epsilon)
        rot_3x3 = quaternion_to_rotation_matrix(q)
        center = self.position

        # Rotate the target mesh
        if self._target_actor is not None:
            self._rotate_actor_mesh(self._target_actor, rot_3x3, center)

        # Rotate every handle mesh
        for handle in self._all_handles:
            self._rotate_actor_mesh(handle.actor, rot_3x3, center)

        # Accumulate the rotation
        self._cumulative_rotation = rot_3x3 @ self._cumulative_rotation

        self._last_rotation_dir = current_dir.copy()

    @staticmethod
    def _translate_actor_mesh(actor, displacement):
        # Translate the rendered mesh of actor by displacement
        # in-place (directly modifies the VTK point array).
        vtk_data = actor.GetMapper().GetInput()
        vtk_points = vtk_data.GetPoints()
        pts = vtk_to_numpy(vtk_points.GetData())

        pts += displacement

        vtk_points.Modified()
        vtk_data.Modified()

    @staticmethod
    def _rotate_actor_mesh(actor, rot_3x3, center):
        # Rotate the rendered mesh of actor around center
        # in-place (points and normals).
        vtk_data = actor.GetMapper().GetInput()
        vtk_points = vtk_data.GetPoints()
        pts = vtk_to_numpy(vtk_points.GetData())

        centered = pts - center
        pts[:] = (rot_3x3 @ centered.T).T + center
        vtk_points.Modified()

        pn = vtk_data.GetPointData().GetNormals()
        if pn is not None:
            normals = vtk_to_numpy(pn)
            normals[:] = (rot_3x3 @ normals.T).T
            pn.Modified()

        cn = vtk_data.GetCellData().GetNormals()
        if cn is not None:
            normals = vtk_to_numpy(cn)
            normals[:] = (rot_3x3 @ normals.T).T
            cn.Modified()

        vtk_data.Modified()


    def _add_debug_actors(self):
        ray = self._click_ray
        plane = self._constraint_plane
        if ray is None or plane is None:
            return

        # Ray line
        cam_pos = np.array(self.plotter.camera.position)
        dist = np.linalg.norm(plane.position - cam_pos)
        end_pt = cam_pos + ray.direction * max(dist * 3.0, 1.0)
        self.plotter.add_mesh(
            pv.Line(cam_pos, end_pt),
            color="white", line_width=3,
            name="_debug_ray", pickable=False,
        )

        # Small sphere at camera position
        self.plotter.add_mesh(
            pv.Sphere(radius=0.005, center=cam_pos),
            color="white",
            name="_debug_ray_origin", pickable=False,
        )

        # Constraint plane
        self.plotter.add_mesh(
            pv.Plane(
                center=self.position, direction=plane.normal,
                i_size=0.4, j_size=0.4,
                i_resolution=4, j_resolution=4,
            ),
            color="yellow", opacity=0.25,
            show_edges=True, edge_color="yellow",
            name="_debug_plane", pickable=False,
        )

        # Intersection sphere
        if self._intersection_point is not None:
            self.plotter.add_mesh(
                pv.Sphere(radius=0.008, center=self._intersection_point),
                color="lime",
                name="_debug_intersection", pickable=False,
            )

        self.plotter.render()

    def _remove_debug_actors(self):
        # Remove all debug visualisation actors.
        for name in ("_debug_ray", "_debug_ray_origin", "_debug_plane", "_debug_intersection"):
            try:
                self.plotter.remove_actor(name)
            except Exception:
                pass
        self.plotter.render()
