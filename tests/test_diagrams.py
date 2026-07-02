"""Smoke test: ControlMode enum and control-diagram builders."""
from pydrake.systems.framework import Diagram

from odyssey.diagrams import (
    ControlMode,
    cartesian_velocity_diagram,
    diffik_pose_diagram,
    joint_control_diagram,
)


def test_control_mode_members():
    names = {m.name for m in ControlMode}
    assert names == {"JOINT", "DIFFIK_POSE", "CARTESIAN_VELOCITY"}


def test_joint_control_diagram_builds(plant):
    diagram = joint_control_diagram(plant, simulated=True)
    assert isinstance(diagram, Diagram)


def test_diffik_pose_diagram_builds(plant):
    diagram = diffik_pose_diagram(plant, simulated=True, ee_frame="iiwa_link_7")
    assert isinstance(diagram, Diagram)


def test_cartesian_velocity_diagram_builds(plant):
    diagram = cartesian_velocity_diagram(
        plant, ee_frame="iiwa_link_7", simulated=True, vel_limit=0.03
    )
    assert isinstance(diagram, Diagram)
