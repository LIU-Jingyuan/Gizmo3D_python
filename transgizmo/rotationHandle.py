import numpy as np
import pyvista as pv
from vtkmodules.vtkRenderingCore import vtkCellPicker


class rotationHandle:
    HOVER_COLOR = "yellow"
    _RING_RADIUS = 0.15            # major radius of the torus
    _CROSS_SECTION_RADIUS = 0.002  # minor radius (tube thickness)
    _RESOLUTION = 100              # rings & slices

    def __init__(self, plotter, position, label, gizmo=None):
        self.plotter = plotter
        self.position = np.asarray(position, dtype=float)
        self.label = label
        self.gizmo = gizmo
        self._hovered = False
        self._dragging = False

        # axis colour & orientation
        if label == "XRot":
            self.color = "red"
            rotation_axis = (0, 1, 0)
            rotation_angle = 90.0
        elif label == "YRot":
            self.color = "green"
            rotation_axis = (1, 0, 0)
            rotation_angle = 90.0
        elif label == "ZRot":
            self.color = "blue"
            rotation_axis = None       # default torus already lies in XY
            rotation_angle = 0.0

        self.original_color = self.color

        # build torus mesh (centred at origin, XY plane by default)
        self.torus_mesh = pv.ParametricTorus(
            ringradius=self._RING_RADIUS,
            crosssectionradius=self._CROSS_SECTION_RADIUS,
        )

        # orient the torus to the correct plane
        if rotation_axis is not None and rotation_angle != 0.0:
            self.torus_mesh.rotate_vector(
                rotation_axis, rotation_angle, inplace=True,
            )

        # translate to gizmo position
        self.torus_mesh.translate(self.position, inplace=True)

        # add to plotter
        self.actor = self.plotter.add_mesh(
            self.torus_mesh,
            color=self.color,
            name=self.label,
            pickable=True,
            smooth_shading=True,
        )

        # hover detection via VTK cell picker
        self._picker = vtkCellPicker()
        self._picker.SetTolerance(0.005)
        self._observer_tag = self.plotter.iren.add_observer(
            "MouseMoveEvent", self._on_mouse_move,
        )

        # click / release observers for drag interaction
        self.plotter.iren.add_observer(
            "LeftButtonPressEvent", self._on_left_button_press,
        )
        self.plotter.iren.add_observer(
            "LeftButtonReleaseEvent", self._on_left_button_release,
        )

    def _on_mouse_move(self, caller, event):
        # Pick-test on every mouse move and toggle the highlight.
        # During a drag, forward the position to the gizmo instead.
        x, y = caller.GetEventPosition()

        # If this handle is being dragged, just forward the position
        if self._dragging and self.gizmo is not None:
            self.gizmo._on_handle_dragged(self.label, x, y)
            return

        # Skip hover testing while any other handle is being dragged
        if self.gizmo is not None and self.gizmo._active_handle is not None:
            return

        self._picker.Pick(x, y, 0, self.plotter.renderer)
        picked_actor = self._picker.GetActor()
        self._set_highlight(picked_actor == self.actor)

    def _set_highlight(self, highlighted):
        # Set colour to HOVER_COLOR while highlighted, revert otherwise.
        if highlighted and not self._hovered:
            self._hovered = True
            self.actor.prop.color = pv.Color(self.HOVER_COLOR)
            self.plotter.render()
        elif not highlighted and self._hovered:
            self._hovered = False
            self.actor.prop.color = pv.Color(self.original_color)
            self.plotter.render()

    def _on_left_button_press(self, caller, event):
        # If this handle is hovered, begin a drag and notify the gizmo.
        if (self._hovered
                and self.gizmo is not None
                and self.gizmo._active_handle is None):
            self._dragging = True
            x, y = caller.GetEventPosition()
            self.gizmo._on_handle_pressed(self.label, x, y)

    def _on_left_button_release(self, caller, event):
        # End the drag (if active) and notify the gizmo.
        if self._dragging and self.gizmo is not None:
            self._dragging = False
            self.gizmo._on_handle_released(self.label)

    def set_color(self, color):
        # Change the handle's base colour (also updates the restore colour).
        self.color = color
        self.original_color = color
        if not self._hovered:
            self.actor.prop.color = pv.Color(color)
            self.plotter.render()
