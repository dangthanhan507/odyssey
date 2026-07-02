"""Smoke test: pure-drake helpers in odyssey.utils."""
from pydrake.multibody.inverse_kinematics import (
    DifferentialInverseKinematicsParameters,
)
from pydrake.systems.framework import DiagramBuilder, LeafSystem, System

from odyssey.utils import AddIiwaDifferentialIK, DiffIKParams, VelocityDiffIK


def test_diffik_params(plant):
    params = DiffIKParams(plant)
    assert isinstance(params, DifferentialInverseKinematicsParameters)


def test_velocity_diffik_is_leaf_system(plant):
    system = VelocityDiffIK(plant, frame_name="iiwa_link_7")
    assert isinstance(system, LeafSystem)


def test_add_iiwa_differential_ik(plant):
    builder = DiagramBuilder()
    before = len(builder.GetSystems())
    diffik = AddIiwaDifferentialIK(
        builder, plant, plant.GetFrameByName("iiwa_link_7")
    )
    after = len(builder.GetSystems())
    assert after == before + 1
    assert isinstance(diffik, System)
