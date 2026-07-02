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

import numpy as np


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

    def add_box(self, center, size, rotation=None, name=""):
        return self.add(BoxObstacle(center, size, rotation=rotation, name=name))

    def __len__(self):
        return len(self.obstacles)

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
