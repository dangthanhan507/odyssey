"""Smoke test: workstation LCM message-building/publishing.

KukaLCM opens a multicast LCM socket in __init__ (no network peer needed) and
the send_* methods just publish to LCM (fine with no listener). We never start
viser here (OdysseyBaseWorkstation is constructed with use_viser=False).
"""
import numpy as np

from odyssey.workstation import KukaLCM, OdysseyBaseWorkstation


def test_kuka_lcm_construct():
    lcm = KukaLCM()
    assert lcm.lcm is not None


def test_send_joint_command():
    lcm = KukaLCM()
    lcm.send_joint_command(np.zeros(7).tolist(), np.zeros(7).tolist())


def test_send_pose_command():
    lcm = KukaLCM()
    lcm.send_pose_command(
        np.array([1.0, 0.0, 0.0, 0.0]),
        np.array([0.5, 0.0, 0.5]),
        np.zeros(7).tolist(),
    )


def test_send_cartesian_velocity_command():
    lcm = KukaLCM()
    lcm.send_cartesian_velocity_command(np.zeros(6), np.zeros(7).tolist())


def test_base_workstation_no_viser():
    """Constructing without viser must not start a server or load a URDF."""
    ws = OdysseyBaseWorkstation(
        robot_urdf_path="unused.urdf",
        robot_description_path="unused/",
        use_viser=False,
    )
    assert ws.use_viser is False
    assert ws.kuka_lcm is not None
    # send_* helpers should delegate to the underlying KukaLCM without error
    ws.send_joint_command(np.zeros(7).tolist())
    ws.send_pose_command(np.array([1.0, 0.0, 0.0, 0.0]), np.array([0.5, 0.0, 0.5]))
    ws.send_cartesian_velocity_command(np.zeros(6))
