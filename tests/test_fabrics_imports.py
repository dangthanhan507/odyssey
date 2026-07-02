"""Smoke test: fabrics modules and the example import cleanly.

The example module is imported as a module (never as __main__), so its
``if __name__ == '__main__':`` / ``main()`` block does not execute.
"""
import importlib

import pytest

FABRICS_MODULES = [
    "odyssey.fabrics",
    "odyssey.fabrics.obstacles",
    "odyssey.fabrics.kinematics",
    "odyssey.fabrics.fabric_terms",
    "odyssey.fabrics.fabric",
    "odyssey.examples.fabric_box_avoidance",
    "odyssey.examples.constraint_editor",
]


@pytest.mark.parametrize("module_name", FABRICS_MODULES)
def test_fabrics_module_imports(module_name):
    mod = importlib.import_module(module_name)
    assert mod is not None


def test_public_api_available():
    from odyssey.fabrics import (
        BoxObstacle,
        ObstacleSet,
        RobotKinematics,
        IiwaBoxFabric,
    )
    assert BoxObstacle is not None
    assert ObstacleSet is not None
    assert RobotKinematics is not None
    assert IiwaBoxFabric is not None
