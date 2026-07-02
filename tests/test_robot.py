"""Smoke test: RobotLoopDiagram construction, diagram build, arm pose, and a
tiny finite simulator advance in simulated-hardware mode.
"""
import numpy as np
import pytest
from pydrake.systems.framework import Diagram

from odyssey.robot import ControlMode, RobotLoopDiagram


@pytest.fixture
def loop_diagram(default_config):
    return RobotLoopDiagram(
        config=default_config,
        use_simulated_hardware=True,
        control_mode=ControlMode.JOINT,
    )


def test_construct(loop_diagram):
    assert loop_diagram.control_mode == ControlMode.JOINT
    assert loop_diagram.simulated is True


def test_setup_diagram(loop_diagram):
    diagram = loop_diagram.setup_diagram()
    assert isinstance(diagram, Diagram)


def test_get_arm_pose(loop_diagram):
    quat, pos = loop_diagram.get_arm_pose("iiwa_link_7", np.zeros(7))
    assert quat.shape == (4,)
    assert pos.shape == (3,)
    # quaternion should be (approximately) unit norm
    assert np.isclose(np.linalg.norm(quat), 1.0, atol=1e-6)


def test_tiny_simulator_advance(loop_diagram):
    """Run the simulated loop for a tiny finite duration to smoke-test that the
    diagram advances. This is bounded (0.01s), so it cannot hang; in simulated
    JOINT mode with no LCM commands the robot simply idles, which is fine.
    """
    diagram = loop_diagram.setup_diagram()
    loop_diagram.run_system(diagram, duration=0.01)
