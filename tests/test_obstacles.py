"""Smoke test: analytic box SDF and obstacle-set queries.

Pure-numpy, no Drake / hardware. Verifies the signed-distance and collision
direction used by the fabric repulsion term.
"""
import numpy as np

from odyssey.fabrics import BoxObstacle, ObstacleSet


def test_box_signed_distance_sign():
    box = BoxObstacle(center=[0, 0, 0], size=[2, 2, 2])  # [-1, 1]^3
    # Outside (+1), on-surface (0), inside (-0.5).
    d = box.signed_distance(np.array([[2, 0, 0], [0, 0, 1.0], [0.5, 0, 0]]))
    assert np.allclose(d, [1.0, 0.0, -0.5])


def test_box_direction_points_toward_surface():
    box = BoxObstacle(center=[0, 0, 0], size=[2, 2, 2])
    # Exterior point at +x: normal (center->closest surface) points -x.
    ext = box.direction(np.array([2.0, 0.0, 0.0]))
    assert np.allclose(ext, [-1.0, 0.0, 0.0])
    # Interior point near +x face: nearest-face normal points +x.
    ins = box.direction(np.array([0.5, 0.0, 0.0]))
    assert np.allclose(ins, [1.0, 0.0, 0.0])
    # Direction is always unit length (or zero).
    assert np.isclose(np.linalg.norm(ext), 1.0)


def test_obstacle_set_returns_closest():
    obs = ObstacleSet()
    obs.add_box(center=[0, 0, 0], size=[2, 2, 2])
    obs.add_box(center=[5, 0, 0], size=[2, 2, 2])
    dist, direction = obs.query(np.array([[2.0, 0.0, 0.0]]))
    # Closest box is the first one; distance to its surface is 1.0.
    assert np.isclose(dist[0], 1.0)
    assert np.allclose(direction[0], [-1.0, 0.0, 0.0])


def test_empty_obstacle_set_is_infinite():
    obs = ObstacleSet()
    dist, direction = obs.query(np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]))
    assert np.all(np.isinf(dist))
    assert np.allclose(direction, 0.0)
