"""Analytic obstacle signed-distance fields for fabric collision avoidance.

This is the CPU/numpy analogue of the world-mesh collision query in NVIDIA's
``fabrics_sim`` (``body_sphere_3d_repulsion.py`` launches a Warp kernel that
calls ``wp.mesh_query_point`` against arbitrary meshes). Here we only need
axis-aligned / oriented boxes, so instead of a mesh BVH we use closed-form box
SDFs. Each obstacle answers two queries for a query point ``p``:

* ``signed_distance(p)`` -- distance from ``p`` to the box surface. Negative
  when ``p`` is inside the box (penetrating), positive outside.
* ``direction(p)`` -- unit vector pointing from ``p`` toward the closest point
  on the box surface (i.e. the direction the collision would push *into*). This
  mirrors ``n = normalize(closest_point - sphere_center)`` in the Warp kernel.

The fabric repulsion term shaves off the sphere radius from the signed distance
and builds a rank-1 metric along ``direction``; see ``fabric_terms.py``.
"""

import json

import numpy as np


def _wxyz_to_matrix(wxyz):
    """Convert a (w, x, y, z) quaternion to a 3x3 rotation matrix."""
    w, x, y, z = np.asarray(wxyz, dtype=float)
    n = np.sqrt(w * w + x * x + y * y + z * z)
    if n < 1e-12:
        return np.eye(3)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def _matrix_to_wxyz(R):
    """Convert a 3x3 rotation matrix to a (w, x, y, z) quaternion."""
    R = np.asarray(R, dtype=float)
    trace = np.trace(R)
    if trace > 0.0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    else:
        i = int(np.argmax([R[0, 0], R[1, 1], R[2, 2]]))
        if i == 0:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif i == 1:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


class BoxObstacle:
    """An oriented box obstacle defined by a center, full-extent size, and rotation.

    Parameters
    ----------
    center : (3,) array
        World position of the box center.
    size : (3,) array
        Full side lengths (x, y, z) of the box. This matches the ``scaling``
        field in ``fabrics_sim`` world YAMLs, which are full extents.
    rotation : (3, 3) array, optional
        World-from-box rotation matrix. Defaults to identity (axis aligned).
    name : str, optional
        Human-readable label, handy for debugging / visualization.
    """

    def __init__(self, center, size, rotation=None, name=""):
        self.center = np.asarray(center, dtype=float).reshape(3)
        self.half_extents = 0.5 * np.asarray(size, dtype=float).reshape(3)
        self.rotation = np.eye(3) if rotation is None else np.asarray(rotation, dtype=float).reshape(3, 3)
        self.name = name

    @property
    def size(self):
        return 2.0 * self.half_extents

    @property
    def wxyz(self):
        """Orientation as a (w, x, y, z) quaternion (viser's convention)."""
        return _matrix_to_wxyz(self.rotation)

    @classmethod
    def from_pose(cls, center, size, wxyz=None, name=""):
        """Build from a center, size, and (w, x, y, z) quaternion."""
        rotation = None if wxyz is None else _wxyz_to_matrix(wxyz)
        return cls(center=center, size=size, rotation=rotation, name=name)

    def to_dict(self):
        """Serialize to a plain dict (JSON-friendly), pose as quaternion."""
        return {
            "name": self.name,
            "center": self.center.tolist(),
            "size": self.size.tolist(),
            "wxyz": self.wxyz.tolist(),
        }

    @classmethod
    def from_dict(cls, d):
        """Inverse of ``to_dict``. Rotation may be given as ``wxyz`` or omitted."""
        return cls.from_pose(
            center=d["center"], size=d["size"],
            wxyz=d.get("wxyz"), name=d.get("name", ""))

    def _to_local(self, points):
        """Transform world points into the box frame. Accepts (3,) or (N, 3)."""
        pts = np.atleast_2d(np.asarray(points, dtype=float))
        return (pts - self.center) @ self.rotation  # R^T (p - c)

    def signed_distance(self, points):
        """Exact signed distance to the box surface (negative inside).

        Uses the standard box SDF: outside distance is the norm of the positive
        part of ``|local| - half_extents``; inside distance is the (negative)
        largest component of that quantity.
        """
        pts_local = self._to_local(points)
        q = np.abs(pts_local) - self.half_extents
        outside = np.linalg.norm(np.maximum(q, 0.0), axis=1)
        inside = np.minimum(np.max(q, axis=1), 0.0)
        dist = outside + inside
        return dist if dist.shape[0] > 1 else dist[0]

    def closest_point(self, points):
        """Closest point on the (solid) box to each query point, in world frame."""
        pts_local = self._to_local(points)
        # Clamp to the box interior gives the closest point for exterior queries.
        clamped = np.clip(pts_local, -self.half_extents, self.half_extents)
        return clamped @ self.rotation.T + self.center

    def direction(self, points):
        """Unit vector from each query point toward the box (collision normal).

        For exterior points this points from the sphere center toward the
        closest surface point (repulsion acts opposite to this). For interior
        points we fall back to the nearest-face normal so the direction is still
        well defined while penetrating.
        """
        pts = np.atleast_2d(np.asarray(points, dtype=float))
        pts_local = self._to_local(pts)
        q = np.abs(pts_local) - self.half_extents
        dirs = np.zeros_like(pts_local)

        inside_mask = np.all(q < 0.0, axis=1)

        # Exterior points: direction toward the clamped (closest) point.
        if np.any(~inside_mask):
            ext = ~inside_mask
            clamped = np.clip(pts_local[ext], -self.half_extents, self.half_extents)
            delta = clamped - pts_local[ext]
            norms = np.linalg.norm(delta, axis=1, keepdims=True)
            dirs[ext] = np.divide(delta, norms, out=np.zeros_like(delta), where=norms > 1e-12)

        # Interior points: the clamp-based closest point degenerates to the
        # query point itself, so fall back to the nearest-face normal. ``n``
        # points from the center toward the nearest face (same sign as the
        # local coordinate on the axis of least penetration).
        if np.any(inside_mask):
            ins = inside_mask
            # q < 0 inside; the axis with q closest to 0 is the nearest face.
            nearest_axis = np.argmax(q[ins], axis=1)
            local_dir = np.zeros((int(ins.sum()), 3))
            rows = np.arange(int(ins.sum()))
            sign = np.sign(pts_local[ins][rows, nearest_axis])
            sign[sign == 0.0] = 1.0  # exactly-centered => arbitrary +face
            local_dir[rows, nearest_axis] = sign
            dirs[ins] = local_dir

        world_dirs = dirs @ self.rotation.T
        return world_dirs if world_dirs.shape[0] > 1 else world_dirs[0]


class ObstacleSet:
    """A collection of obstacles queried together.

    For a batch of query points, returns for each point the signed distance and
    collision direction of the *closest* obstacle. This is the numpy analogue of
    the Warp kernel loop over all world meshes that keeps the minimum distance.
    """

    def __init__(self, obstacles=None):
        self.obstacles = list(obstacles) if obstacles is not None else []

    def add(self, obstacle):
        self.obstacles.append(obstacle)
        return obstacle

    def add_box(self, center, size, rotation=None, name="", wxyz=None):
        if wxyz is not None:
            return self.add(BoxObstacle.from_pose(center, size, wxyz=wxyz, name=name))
        return self.add(BoxObstacle(center, size, rotation=rotation, name=name))

    def remove(self, obstacle):
        self.obstacles.remove(obstacle)

    def __len__(self):
        return len(self.obstacles)

    # -- serialization ------------------------------------------------------
    def to_dict(self):
        """Serialize the whole set to a JSON-friendly dict."""
        return {"version": 1, "frame": "world",
                "boxes": [o.to_dict() for o in self.obstacles]}

    @classmethod
    def from_dict(cls, d):
        return cls([BoxObstacle.from_dict(b) for b in d.get("boxes", [])])

    def save_json(self, path):
        """Write the obstacle set to ``path`` as JSON."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_json(cls, path):
        """Load an obstacle set previously written by ``save_json``."""
        with open(path, "r") as f:
            return cls.from_dict(json.load(f))

    def query(self, points):
        """Closest-obstacle signed distance and direction for each query point.

        Parameters
        ----------
        points : (N, 3) array
            Query points (e.g. robot collision-sphere centers).

        Returns
        -------
        signed_distance : (N,) array
            Distance to the closest obstacle surface, negative if penetrating.
        direction : (N, 3) array
            Unit vector from each point toward the closest obstacle surface.
        """
        pts = np.atleast_2d(np.asarray(points, dtype=float))
        n = pts.shape[0]
        if not self.obstacles:
            return np.full(n, np.inf), np.zeros((n, 3))

        best_dist = np.full(n, np.inf)
        best_dir = np.zeros((n, 3))
        for obs in self.obstacles:
            d = np.atleast_1d(obs.signed_distance(pts))
            closer = d < best_dist
            if np.any(closer):
                dirs = np.atleast_2d(obs.direction(pts))
                best_dist[closer] = d[closer]
                best_dir[closer] = dirs[closer]
        return best_dist, best_dir
