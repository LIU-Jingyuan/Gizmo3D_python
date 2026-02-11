import numpy as np
import pyvista as pv
from vtkmodules.vtkRenderingCore import vtkCellPicker


class planeTranslationHandle:
    HOVER_COLOR = "yellow"
    _PLANE_WIDTH = 0.03
    _PLANE_HEIGHT = 0.03
    _PLANE_OFFSET = 0.06          # offset from gizmo centre along each axis

    def __init__(self, plotter, position, label, gizmo=None):
        self.plotter = plotter
        self.position = np.asarray(position, dtype=float)
        self.label = label
        self.gizmo = gizmo
        self._hovered = False
        self._dragging = False

        # plane orientation, colour (mix of the two axis colours), and offset
        if label == "YZTrans":
            self.color = "cyan"               # green + blue
            self.normal = (1, 0, 0)
            offset = np.array([0.0, self._PLANE_OFFSET, self._PLANE_OFFSET])
        elif label == "XZTrans":
            self.color = "magenta"            # red + blue
            self.normal = (0, 1, 0)
            offset = np.array([self._PLANE_OFFSET, 0.0, self._PLANE_OFFSET])
        elif label == "XYTrans":
            self.color = "orange"             # red + green
            self.normal = (0, 0, 1)
            offset = np.array([self._PLANE_OFFSET, self._PLANE_OFFSET, 0.0])

        self.original_color = self.color
        centre = self.position + offset

        # plane mesh
        self.plane_mesh = pv.Plane(
            center=centre,
            direction=self.normal,
            i_size=self._PLANE_WIDTH,
            j_size=self._PLANE_HEIGHT,
            i_resolution=1,
            j_resolution=1,
        )

        self.actor = self.plotter.add_mesh(
            self.plane_mesh,
            color=self.color,
            name=self.label,
            pickable=True,
            show_edges=False,
            opacity=0.6,
        )

        # disable backface culling so the plane is visible from both sides
        self.actor.prop.SetBackfaceCulling(False)

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
