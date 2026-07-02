"""Smoke test: every odyssey module and example imports cleanly.

Example modules are imported as modules (never as __main__), so their
`if __name__ == '__main__':` blocks do not execute.
"""
import importlib

import pytest

ODYSSEY_MODULES = [
    "odyssey.robot",
    "odyssey.diagrams",
    "odyssey.utils",
    "odyssey.workstation",
    "odyssey.robot_lcm",
    "odyssey.msgs.lcm_msgs",
]


@pytest.mark.parametrize("module_name", ODYSSEY_MODULES)
def test_odyssey_module_imports(module_name):
    mod = importlib.import_module(module_name)
    assert mod is not None


def test_lcm_message_classes_available():
    from odyssey.msgs.lcm_msgs import (
        iiwa_commands_t,
        lcmt_iiwa_command,
        lcmt_iiwa_status,
    )
    assert iiwa_commands_t is not None
    assert lcmt_iiwa_command is not None
    assert lcmt_iiwa_status is not None


# Examples that only need libraries known to be installed. pyspacemouse is
# installed, so spacemouse_teleop is expected to import. oculus_reader is NOT
# installed; there is currently no oculus example, but if one is added it
# should be skipped via importorskip below.
EXAMPLE_MODULES = [
    "odyssey.examples.cartesian_drag",
    "odyssey.examples.joint_sliders",
    "odyssey.examples.move_forward",
    "odyssey.examples.spacemouse_teleop",
    # oculus_teleop imports oculus_reader lazily (inside __init__), so the
    # module itself imports fine without the hardware library installed.
    "odyssey.examples.oculus_teleop",
    # gamepad_teleop imports pygame lazily (inside __init__); pygame IS
    # installed, so the module imports fine without a joystick connected.
    "odyssey.examples.gamepad_teleop",
]


@pytest.mark.parametrize("module_name", EXAMPLE_MODULES)
def test_example_module_imports(module_name):
    # oculus_teleop imports oculus_reader lazily (inside __init__), so importing
    # the module does not require the hardware library. Only skip if some future
    # example imports it at module top level and it is not installed.
    mod = importlib.import_module(module_name)
    assert mod is not None
