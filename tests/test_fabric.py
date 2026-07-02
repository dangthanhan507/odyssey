"""Smoke test: the geometric fabric constructs, steps, and avoids obstacles.

Uses the session-scoped ``fabric`` fixture (builds the Drake plant once). These
run the numpy fabric directly; no viser, no LCM, no hardware.
"""
import numpy as np

from odyssey.fabrics import IiwaBoxFabric, ObstacleSet

HOME_Q = np.array([0.0, 0.4, 0.0, -1.0, 0.0, 0.6, 0.0])


def test_kinematics_shapes(fabric):
    q = HOME_Q
    x, J = fabric.kin.body_points_taskmap(q)
    assert x.shape == (3 * fabric.kin.num_spheres,)
    assert J.shape == (3 * fabric.kin.num_spheres, fabric.num_joints)
    xe, Je = fabric.kin.ee_taskmap(q)
    assert xe.shape == (3,)
    assert Je.shape == (3, fabric.num_joints)


def test_jacobian_matches_finite_difference(fabric):
    q = HOME_Q.copy()
    xe, Je = fabric.kin.ee_taskmap(q)
    eps = 1e-6
    q2 = q.copy()
    q2[0] += eps
    xe2, _ = fabric.kin.ee_taskmap(q2)
    fd = (xe2 - xe) / eps
    assert np.allclose(fd, Je[:, 0], atol=1e-4)


def test_eval_natural_produces_pd_metric(fabric):
    q = HOME_Q
    qd = np.zeros(fabric.num_joints)
    M, f, M_inv = fabric.eval_natural(q, qd)
    assert M.shape == (fabric.num_joints, fabric.num_joints)
    assert f.shape == (fabric.num_joints,)
    # Metric must be symmetric positive-definite (invertible) so q'' is defined.
    assert np.allclose(M, M.T, atol=1e-8)
    assert np.all(np.linalg.eigvalsh(M) > 0.0)
    assert np.allclose(M @ M_inv, np.eye(fabric.num_joints), atol=1e-6)


def test_single_step_runs_and_is_finite(fabric):
    q = HOME_Q
    qd = np.zeros(fabric.num_joints)
    fabric.set_ee_target([0.5, -0.35, 0.55])
    q_new, qd_new, qdd = fabric.step(q, qd, 1.0 / 60.0)
    assert np.all(np.isfinite(q_new))
    assert np.all(np.isfinite(qd_new))
    assert np.all(np.isfinite(qdd))


def test_fabric_reaches_target_in_free_space(fabric):
    fabric.set_posture_target(HOME_Q)
    fabric.set_ee_target([0.55, 0.0, 0.35])
    q = HOME_Q.copy()
    qd = np.zeros(fabric.num_joints)
    dt = 1.0 / 60.0
    for _ in range(600):
        q, qd, qdd = fabric.step(q, qd, dt)
    xe, _ = fabric.kin.ee_taskmap(q)
    # Should converge close to the (reachable, collision-free) target.
    assert np.linalg.norm(xe - np.array([0.55, 0.0, 0.35])) < 0.05


def test_repulsion_reduces_penetration():
    """With the same reach, adding repulsion must not increase penetration."""
    target = np.array([0.5, 0.55, 0.6])
    box = ([0.45, 0.30, 0.6], [0.3, 0.15, 0.5])
    probe = ObstacleSet()
    probe.add_box(*box)

    def run(with_box):
        obs = ObstacleSet()
        if with_box:
            obs.add_box(*box)
        fab = IiwaBoxFabric(obs, use_posture_attractor=True)
        fab.set_posture_target(HOME_Q)
        fab.set_ee_target(target)
        q = HOME_Q.copy()
        qd = np.zeros(fab.num_joints)
        worst = np.inf
        for _ in range(900):
            q, qd, qdd = fab.step(q, qd, 1.0 / 60.0)
            d, _ = probe.query(fab.kin.sphere_positions(q))
            worst = min(worst, float((d - fab.kin.sphere_radii).min()))
        return worst

    worst_off = run(False)
    worst_on = run(True)
    # Repulsion should keep the arm at least as clear as with no obstacle.
    assert worst_on >= worst_off - 1e-3
