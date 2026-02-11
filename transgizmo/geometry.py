import numpy as np


class Ray:
    # A ray (or line segment) defined by start and end in world space.
    def __init__(self, start, end, epsilon):
        self.start = np.asarray(start, dtype=float)
        self.end = np.asarray(end, dtype=float)
        self.epsilon = epsilon

    @property
    def direction(self):
        # Unit direction vector from start toward end.
        d = self.end - self.start
        length = np.linalg.norm(d)
        return d / length if length > self.epsilon else np.zeros(3)

    def intersects(self, plane):
        """
        Returns
        (code, point) : tuple[int, np.ndarray]
            * code 0 — no intersection (ray is parallel to the plane)
            * code 1 — single intersection at point
            * code 2 — ray lies entirely in the plane
        """
        u = self.start - self.end
        w = self.start - plane.position

        D = np.dot(plane.normal, u)
        N = -np.dot(plane.normal, w)

        if abs(D) < self.epsilon:
            # Ray is parallel to the plane
            if abs(N) < self.epsilon:
                return (2, self.start.copy())   # lies in plane
            return (0, np.zeros(3))             # no intersection

        s = N / D
        return (1, self.start + u * s)


class Plane:
    # An infinite plane defined by a position on it and a unit normal
    def __init__(self, position, normal, epsilon):
        self.position = np.asarray(position, dtype=float)
        n = np.asarray(normal, dtype=float)
        length = np.linalg.norm(n)
        self.normal = n / length if length > epsilon else np.array([0.0, 1.0, 0.0])


### Screen --> world ray
def generate_ray_from_screen(renderer, x, y, epsilon):
    # Unproject display coordinates *(x, y)* into a world-space :class:`Ray`.

    # Near plane (z = 0)
    renderer.SetDisplayPoint(float(x), float(y), 0.0)
    renderer.DisplayToWorld()
    wp = renderer.GetWorldPoint()
    near = np.array(wp[:3]) / wp[3]

    # Far plane (z = 1)
    renderer.SetDisplayPoint(float(x), float(y), 1.0)
    renderer.DisplayToWorld()
    wp = renderer.GetWorldPoint()
    far = np.array(wp[:3]) / wp[3]

    return Ray(near, far, epsilon=epsilon)


def compute_plane_normal(ray, axis_constraint, orientation=None):
    # Compute the plane normal for a translation constraint.
    view = ray.start - ray.end                      # rough view direction

    if orientation is None:
        orientation = np.eye(3)

    local_x = orientation @ np.array([1.0, 0.0, 0.0])
    local_y = orientation @ np.array([0.0, 1.0, 0.0])
    local_z = orientation @ np.array([0.0, 0.0, 1.0])

    # Single-axis: remove the component along the constrained axis
    if axis_constraint == "XTrans":
        normal = view - np.dot(view, local_x) * local_x
    elif axis_constraint == "YTrans":
        normal = view - np.dot(view, local_y) * local_y
    elif axis_constraint == "ZTrans":
        normal = view - np.dot(view, local_z) * local_z
    # Plane constraints: normal IS the perpendicular local axis
    elif axis_constraint == "XZTrans":
        normal = np.dot(view, local_y) * local_y
    elif axis_constraint == "YZTrans":
        normal = np.dot(view, local_x) * local_x
    else:  # XYTrans
        normal = np.dot(view, local_z) * local_z

    return normal


def initialize_translation_plane(click_ray, position, axis_constraint,
                                 orientation=None, epsilon=None):
    # Create the constraint class `Plane` for a translation handle.
    normal = compute_plane_normal(click_ray, axis_constraint, orientation)
    return Plane(np.asarray(position, dtype=float), normal, epsilon=epsilon)


def initialize_rotation_plane(position, axis_constraint, orientation=None,
                              epsilon=None):
    # Create the constraint class `Plane` for a rotation handle.
    normals = {
        "XRot": np.array([1.0, 0.0, 0.0]),
        "YRot": np.array([0.0, 1.0, 0.0]),
        "ZRot": np.array([0.0, 0.0, 1.0]),
    }
    normal = normals.get(axis_constraint, np.array([0.0, 1.0, 0.0]))
    if orientation is not None:
        normal = orientation @ normal
    return Plane(np.asarray(position, dtype=float), normal, epsilon=epsilon)


def apply_translation_constraint(plane_position, intersection_position,
                                 axis_constraint, orientation=None):
    # Project an intersection point onto the constrained axis / plane.
    plane_position = np.asarray(plane_position, dtype=float)
    intersection_position = np.asarray(intersection_position, dtype=float)

    if axis_constraint in ("XTrans", "YTrans", "ZTrans"):
        if orientation is None:
            orientation = np.eye(3)

        axis_map = {
            "XTrans": np.array([1.0, 0.0, 0.0]),
            "YTrans": np.array([0.0, 1.0, 0.0]),
            "ZTrans": np.array([0.0, 0.0, 1.0]),
        }
        local_axis = orientation @ axis_map[axis_constraint]
        disp = intersection_position - plane_position
        return plane_position + np.dot(disp, local_axis) * local_axis

    # Plane constraints (XY / XZ / YZ) — use the full intersection
    return intersection_position.copy()

def quaternion_rotation_to(v_from, v_to, epsilon):
    # Return the quaternion (w, x, y, z) that rotates v_from to v_to.
    v_from = np.asarray(v_from, dtype=float)
    v_to = np.asarray(v_to, dtype=float)

    d = np.dot(v_from, v_to)

    # Nearly identical — identity quaternion
    if d >= 1.0 - epsilon:
        return np.array([1.0, 0.0, 0.0, 0.0])

    # Nearly opposite — 180° rotation around an arbitrary perpendicular
    if d <= -1.0 + epsilon:
        axis = np.cross(np.array([1.0, 0.0, 0.0]), v_from)
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(np.array([0.0, 1.0, 0.0]), v_from)
        axis = axis / np.linalg.norm(axis)
        return np.array([0.0, axis[0], axis[1], axis[2]])

    cross = np.cross(v_from, v_to)
    w = 1.0 + d
    q = np.array([w, cross[0], cross[1], cross[2]])
    return q / np.linalg.norm(q)


def quaternion_to_rotation_matrix(q):
    # Convert a (w, x, y, z) unit quaternion to a 3×3 rotation matrix.
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)    ],
        [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)    ],
        [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)],
    ])
