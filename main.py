import sys
import argparse
from pyvistaqt import QtInteractor
from PyQt5 import QtWidgets, uic, QtCore
from transgizmo.gizmo import Gizmo

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, args):
        super().__init__()

        uic.loadUi("./UI/MainWindow.ui", self)

        self.verbose = args.verbose

        # Replace QWidget placeholder with QtInteractor
        self.plotter = QtInteractor(self.canvas3D)
        layout = QtWidgets.QVBoxLayout(self.canvas3D)
        layout.addWidget(self.plotter.interactor)
        self.plotter.add_axes()

        self._init_scene()

        self.gizmo = Gizmo(self.plotter, position=self.obj_center,
                           show_debug_vis=self.verbose,
                           epsilon=args.epsilon)
        self.gizmo.set_target(self.obj_actor)

        # Connect gizmo signals (only print when verbose)
        if self.verbose:
            self.gizmo.handle_pressed.connect(
                lambda name, x, y: print(f"[Gizmo] Pressed:  {name}  at ({x}, {y})")
            )
            self.gizmo.handle_dragged.connect(
                lambda name, x, y: print(f"[Gizmo] Dragging: {name}  at ({x}, {y})")
            )
            self.gizmo.handle_released.connect(
                lambda name: print(f"[Gizmo] Released: {name}")
            )

    from UI.scene import _init_scene


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=True, description="UI for testing 3D widget.")
    parser.add_argument('--verbose', action='store_true',
                        help='Show visualization of ray, constraint plane, and mouse movements when set.')
    parser.add_argument('--epsilon', type=float, default=2.5e-8,
                        help='Floating-point tolerance used for near-zero checks in ray/plane intersection and quaternion comparisons')
    args = parser.parse_args()

    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow(args)
    window.show()
    sys.exit(app.exec_())