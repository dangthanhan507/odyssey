"""Shared pytest fixtures for odyssey smoke tests.

These tests are fast smoke tests: they verify that modules import and that
core objects construct/build. They do NOT touch real hardware, do NOT start a
viser GUI, and do NOT run infinite Simulator loops.

Config/urdf paths in the repo are relative to the package directory, so we
resolve absolute paths from the package location (via a submodule's __file__,
since `odyssey` is a namespace package with no __init__.py and thus
`odyssey.__file__` is None).
"""
import os

import pytest


def _package_dir():
    # odyssey is a namespace package (no __init__.py), so odyssey.__file__ is
    # None. Resolve the package directory from a concrete submodule instead.
    import odyssey.robot as _robot
    return os.path.dirname(_robot.__file__)


@pytest.fixture(scope="session")
def package_dir():
    return _package_dir()


@pytest.fixture(scope="session")
def default_config(package_dir):
    """Absolute path to the default kuka scenario config."""
    return os.path.join(package_dir, "configs", "kuka_default.yaml")


@pytest.fixture(scope="session")
def obstacle_set():
    """A small box obstacle set (built once for the session)."""
    from odyssey.fabrics import ObstacleSet
    obs = ObstacleSet()
    obs.add_box(center=[0.55, 0.30, 0.60], size=[0.30, 0.20, 0.50], name="front")
    obs.add_box(center=[0.55, 0.00, 0.02], size=[0.60, 1.00, 0.06], name="table")
    return obs


@pytest.fixture(scope="session")
def fabric(obstacle_set):
    """A constructed IiwaBoxFabric (builds the Drake plant once)."""
    from odyssey.fabrics import IiwaBoxFabric
    return IiwaBoxFabric(obstacle_set, use_posture_attractor=True)
