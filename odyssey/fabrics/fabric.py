"""Geometric fabric for the iiwa: assembles leaf terms and integrates motion.

Single-robot numpy port of NVIDIA ``fabrics_sim``'s ``BaseFabric.eval_natural``
+ ``DisplacementIntegrator``. The pipeline per control step is:

1. For each task map, evaluate ``(x, J)`` at ``q`` and at a perturbed
   ``q + eps*qd`` to get the curvature force (finite-difference ``J_dot qd``).
2. Each leaf term produces a metric ``M`` and force ``f`` (forcing or geometric).
3. Pull metric/forces to configuration space (``J^T M J``, ``J^T f``) and add
   the curvature correction, summing all task maps into one ``(M, f)``.
4. Energize the geometric force (keeps the fabric a valid, stable geometry),
   add potential forces and cspace damping.
5. Solve ``q'' = -M^{-1} f`` and integrate ``q, q'`` forward.

Behavior is tuned entirely through ``configs/fabric_params.yaml``; the code
mirrors the fabrics_sim structure so the two stay comparable.
"""

import os

import numpy as np
import yaml

from odyssey.fabrics.kinematics import RobotKinematics
from odyssey.fabrics.fabric_terms import (
    EndEffectorAttractor,
    CspacePostureAttractor,
    JointLimitRepulsion,
    BodySphereRepulsion,
    CollisionResponse,
)


def _load_params(path):
    if path is None:
        pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(pkg_dir, "configs", "fabric_params.yaml")
    with open(path, "r") as f:
        return yaml.safe_load(f)["fabric_params"]


class IiwaBoxFabric:
    """Geometric fabric driving the iiwa to an ee target while avoiding boxes.

    Parameters
    ----------
    obstacles : ObstacleSet
        World obstacles the collision spheres repel from.
    params_path : str, optional
        Path to fabric parameter YAML. Defaults to packaged fabric_params.yaml.
    config : str, optional
        Scenario YAML for the Drake plant (passed to RobotKinematics).
    ee_frame : str
        End-effector frame for the attractor.
    use_posture_attractor : bool
        If True, also pull the arm toward a rest posture (nullspace resolution).
    """

    def __init__(self, obstacles, params_path=None, config=None,
                 ee_frame="iiwa_link_7", use_posture_attractor=True):
        self.params = _load_params(params_path)
        self.kin = RobotKinematics(config=config, ee_frame=ee_frame)
        self.num_joints = self.kin.num_joints
        self.obstacles = obstacles

        # Shared obstacle query, then a forcing + geometric repulsion term on the
        # same body-points task map (mirrors add_body_repulsion).
        self.response = CollisionResponse(obstacles, self.kin.sphere_radii,
                                          self.params["body_repulsion"])
        self.repulsion_forcing = BodySphereRepulsion(self.params["body_repulsion"], True, self.response)
        self.repulsion_geom = BodySphereRepulsion(self.params["body_repulsion"], False, self.response)

        # End-effector attractor (forcing) and optional cspace posture attractor.
        self.ee_attractor = EndEffectorAttractor(self.params["ee_attractor"])
        self.posture_attractor = None
        if use_posture_attractor:
            self.posture_attractor = CspacePostureAttractor(self.params["cspace_attractor"])

        # Joint-limit repulsion on upper and lower task maps.
        self.jl_upper = JointLimitRepulsion(self.params["joint_limit_repulsion"])
        self.jl_lower = JointLimitRepulsion(self.params["joint_limit_repulsion"])
        self.q_lower, self.q_upper = self.kin.joint_limits()

        self._last_sphere_positions = None
        self._last_signed_distance = None

    # -- target setters -----------------------------------------------------
    def set_ee_target(self, target):
        self.ee_attractor.set_target(target)

    def set_posture_target(self, q_rest):
        if self.posture_attractor is not None:
            self.posture_attractor.set_target(q_rest)

    # -- diagnostics --------------------------------------------------------
    @property
    def collision_status(self):
        return self.response.in_collision

    @property
    def min_signed_distance(self):
        if self._last_signed_distance is None:
            return np.inf
        finite = self._last_signed_distance[np.isfinite(self._last_signed_distance)]
        return float(finite.min()) if finite.size else np.inf

    # -- core ---------------------------------------------------------------
    def _pullback(self, x, J, x_eps, J_eps, qd, terms):
        """Pull one task map's terms to cspace, returning (M, f_pot, f_geom).

        Follows eval_container: curvature force from finite-difference J_dot,
        forcing terms contribute a potential force plus a curvature correction
        routed into the geometric bucket, geometric terms contribute directly.
        """
        eps = 1e-5
        xd = J @ qd
        jac_dot = (J_eps - J) / eps
        curvature_force = jac_dot @ qd

        dim = x.shape[0]
        M_leaf = np.zeros((dim, dim))
        pot_leaf = np.zeros(dim)
        geom_leaf = np.zeros(dim)
        for term in terms:
            M_leaf = M_leaf + term.metric(x, xd)
            f = term.force(x, xd)
            if term.is_forcing:
                pot_leaf = pot_leaf + f
            else:
                geom_leaf = geom_leaf + f

        Jt = J.T
        root_M = Jt @ M_leaf @ J

        # Geometric force gets the leaf geometric force + curvature correction.
        leaf_geom = geom_leaf + M_leaf @ curvature_force
        root_geom = Jt @ leaf_geom
        # Forcing terms contribute a potential force in cspace.
        root_pot = Jt @ pot_leaf
        return root_M, root_pot, root_geom

    def eval_natural(self, q, qd):
        """Return total (M, f) for M q'' + f = 0 at state (q, qd)."""
        eps = 1e-5
        q = np.asarray(q, dtype=float)
        qd = np.asarray(qd, dtype=float)
        q_eps = q + eps * qd

        nj = self.num_joints
        M = self.params["base_mass"] * np.eye(nj)  # keeps total metric invertible
        f_geom = np.zeros(nj)
        f_pot = np.zeros(nj)

        # --- body-points task map (collision spheres) ---
        x_bp, J_bp = self.kin.body_points_taskmap(q)
        x_bp_eps, J_bp_eps = self.kin.body_points_taskmap(q_eps)
        sphere_vel = (J_bp @ qd).reshape(self.kin.num_spheres, 3)
        self.response.update(x_bp.reshape(self.kin.num_spheres, 3), sphere_vel)
        self._last_sphere_positions = x_bp.reshape(self.kin.num_spheres, 3)
        self._last_signed_distance = self.response.signed_distance
        rm, rp, rg = self._pullback(x_bp, J_bp, x_bp_eps, J_bp_eps, qd,
                                    [self.repulsion_forcing, self.repulsion_geom])
        M += rm; f_pot += rp; f_geom += rg

        # --- end-effector attractor task map ---
        x_ee, J_ee = self.kin.ee_taskmap(q)
        x_ee_eps, J_ee_eps = self.kin.ee_taskmap(q_eps)
        rm, rp, rg = self._pullback(x_ee, J_ee, x_ee_eps, J_ee_eps, qd, [self.ee_attractor])
        M += rm; f_pot += rp; f_geom += rg

        # --- cspace posture attractor (identity task map: x = q, J = I) ---
        if self.posture_attractor is not None:
            I = np.eye(nj)
            rm, rp, rg = self._pullback(q, I, q_eps, I, qd, [self.posture_attractor])
            M += rm; f_pot += rp; f_geom += rg

        # --- joint-limit repulsion (upper: x = upper - q, J = -I;
        #                             lower: x = q - lower, J = +I) ---
        I = np.eye(nj)
        x_up = self.q_upper - q
        rm, rp, rg = self._pullback(x_up, -I, self.q_upper - q_eps, -I, qd, [self.jl_upper])
        M += rm; f_pot += rp; f_geom += rg
        x_lo = q - self.q_lower
        rm, rp, rg = self._pullback(x_lo, I, q_eps - self.q_lower, I, qd, [self.jl_lower])
        M += rm; f_pot += rp; f_geom += rg

        # --- energization (Euclidean energy => energy metric = I, force = 0) ---
        M_inv = np.linalg.inv(M)
        joint_accel = -M_inv @ f_geom
        scaling = 1.0 / (float(qd @ qd) + 1e-6)
        alpha = -scaling * float(qd @ joint_accel)  # energy_metric = I, energy_force = 0
        mass_velocity = M @ qd
        force = f_geom - alpha * mass_velocity

        # Add potential (forcing) force and cspace damping.
        force = force + f_pot
        force = force + self.params["cspace_damping"]["gain"] * mass_velocity

        return M, force, M_inv

    def step(self, q, qd, dt):
        """Displacement-integrate one step; returns (q_new, qd_new, qdd)."""
        q = np.asarray(q, dtype=float)
        qd = np.asarray(qd, dtype=float)
        M, force, M_inv = self.eval_natural(q, qd)
        qdd = -M_inv @ force
        q_new = q + dt * qd + 0.5 * dt ** 2 * qdd
        qd_new = qd + dt * qdd
        return q_new, qd_new, qdd
