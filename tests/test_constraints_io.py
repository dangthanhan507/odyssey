"""Smoke test: constraint-box serialization and the viser editor.

The obstacle JSON round-trip is the contract between the constraint_editor GUI,
the fabric example, and the real hardware. The editor test constructs a viser
server on an off-default port but never runs its blocking loop.
"""
import numpy as np

from odyssey.fabrics import BoxObstacle, ObstacleSet


def test_box_dict_round_trip():
    box = BoxObstacle([0.4, 0.1, 0.6], [0.3, 0.2, 0.5], name="b")
    d = box.to_dict()
    assert set(d) == {"name", "center", "size", "wxyz"}
    box2 = BoxObstacle.from_dict(d)
    assert np.allclose(box.center, box2.center)
    assert np.allclose(box.size, box2.size)
    assert np.allclose(box.rotation, box2.rotation)


def test_rotated_box_json_round_trip(tmp_path):
    # A tilted box must survive save/load with identical SDF behavior.
    from scipy.spatial.transform import Rotation
    R = Rotation.from_euler("xyz", [0.3, 0.5, -0.2]).as_matrix()
    s = ObstacleSet()
    s.add_box([0.4, 0.1, 0.6], [0.3, 0.2, 0.5], rotation=R, name="tilted")
    s.add_box([0.5, 0.0, 0.02], [0.6, 1.0, 0.06], name="table")

    path = tmp_path / "constraints.json"
    s.save_json(str(path))
    s2 = ObstacleSet.load_json(str(path))

    assert len(s2) == 2
    assert np.allclose(s.obstacles[0].rotation, s2.obstacles[0].rotation, atol=1e-6)
    pts = np.array([[0.4, 0.1, 1.0], [0.5, 0.0, 0.1]])
    d1, _ = s.query(pts)
    d2, _ = s2.query(pts)
    assert np.allclose(d1, d2)


def test_add_box_from_quaternion():
    obs = ObstacleSet()
    obs.add_box([0, 0, 0], [1, 1, 1], wxyz=[1.0, 0.0, 0.0, 0.0], name="ident")
    assert np.allclose(obs.obstacles[0].rotation, np.eye(3))


def test_editor_add_remove_save(tmp_path):
    from odyssey.examples.constraint_editor import ConstraintEditor
    out = tmp_path / "out.json"
    editor = ConstraintEditor(output_path=str(out), port=8099)
    try:
        assert len(editor._entries) == 1  # seed box
        editor._add_default_box()
        editor._add_default_box()
        assert len(editor._entries) == 3
        editor.remove_entry(editor._entries[-1])
        assert len(editor._entries) == 2
        editor.save()
        loaded = ObstacleSet.load_json(str(out))
        assert len(loaded) == 2
    finally:
        editor.server.stop()
