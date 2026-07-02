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
def scenario(default_config):
    from manipulation.station import load_scenario
    return load_scenario(filename=default_config)


@pytest.fixture(scope="session")
def plant(scenario):
    """A finalized MultibodyPlant built the same way robot.py does it."""
    from odyssey.robot import MakeFakeStation
    fake_station = MakeFakeStation(scenario)
    return fake_station.GetSubsystemByName("plant")
