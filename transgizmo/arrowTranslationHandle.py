import numpy as np
import pyvista as pv
from vtkmodules.vtkRenderingCore import vtkCellPicker


class arrowTranslationHandle:
    HOVER_COLOR = "yellow"
    _ARROW_SCALE = 0.2
    _TIP_LENGTH = 0.1              # fraction of total length
    _TIP_RADIUS = 0.03
    _SHAFT_RADIUS = 0.01
    _RESOLUTION = 20

    def __init__(self, plotter, position, label, gizmo=None):
        self.plotter = plotter
        self.position = np.asarray(position, dtype=float)
        self.label = label
        self.gizmo = gizmo
        self._hovered = False
        self._dragging = False

        # axis colour & direction
        if label == "XTrans":
            self.color = "red"
            self.direction = (1, 0, 0)
        elif label == "YTrans":
            self.color = "green"
            self.direction = (0, 1, 0)
        elif label == "ZTrans":
            self.color = "blue"
            self.direction = (0, 0, 1)

        self.original_color = self.color

        # appearance: build the arrow proxy mesh
        self.arrow_mesh = pv.Arrow(
            start=self.position,
            direction=self.direction,
            tip_length=self._TIP_LENGTH,
            tip_radius=self._TIP_RADIUS,
            shaft_radius=self._SHAFT_RADIUS,
            shaft_resolution=self._RESOLUTION,
            tip_resolution=self._RESOLUTION,
            scale=self._ARROW_SCALE,
        )

        # add to plotter and set pickable
        self.actor = self.plotter.add_mesh(
            self.arrow_mesh,
            color=self.color,
            name=self.label,
            pickable=True,
            smooth_shading=True,
        )

        # hover detection
        self._picker = vtkCellPicker()
        self._picker.SetTolerance(0.005)
        self._observer_tag = self.plotter.iren.add_observer(
            "MouseMoveEvent", self._on_mouse_move,
        )

        # click / release observers
        self.plotter.iren.add_observer(
            "LeftButtonPressEvent", self._on_left_button_press,
        )
        self.plotter.iren.add_observer(
            "LeftButtonReleaseEvent", self._on_left_button_release,
        )


    def _on_mouse_move(self, caller, event):
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
        # End the drag and notify the gizmo.
        if self._dragging and self.gizmo is not None:
            self._dragging = False
            self.gizmo._on_handle_released(self.label)

    def set_color(self, color):
        # Change the handle's base colour.
        self.color = color
        self.original_color = color
        if not self._hovered:
            self.actor.prop.color = pv.Color(color)
            self.plotter.render()
