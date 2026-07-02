"""Fabric leaf terms: attractor, joint-limit repulsion, body-sphere repulsion.

These are single-robot numpy ports of the corresponding classes in NVIDIA's
``fabrics_sim`` (``fabric_terms/attractor.py``, ``joint_limit_repulsion.py``,
``body_sphere_3d_repulsion.py``). Each term produces, in its own task space, a
metric ``M`` (a priority mass) and a force ``f`` such that ``M x'' + f = 0``
encodes the desired acceleration policy ``x''`` via ``f = -M x''``.

A term is either a *forcing* policy (drives toward a goal, has a potential) or a
*geometric* policy (velocity-homogeneous, speed-independent shape). The fabric
assembly (``fabric.py``) pulls each ``(M, f)`` back to configuration space and
combines them.
"""

import numpy as np


def _normalize(v, eps=1e-12):
    n = np.linalg.norm(v)
    return v / n if n > eps else np.zeros_like(v)


class EndEffectorAttractor:
    """Forcing attractor pulling a task-space point toward a target.

    Port of ``Attractor`` (forcing branch). The metric is an isotropic mass; the
    policy is a conical (tanh-saturated) pull plus distance-gated damping.
    """

    is_forcing = True

    def __init__(self, params):
        self.p = params
        self.target = None  # set via set_target; None disables the term

    def set_target(self, target):
        self.target = None if target is None else np.asarray(target, dtype=float)

    def metric(self, x, xd):
        dim = x.shape[0]
        if self.target is None:
            return np.zeros((dim, dim))
        return self.p["isotropic_mass"] * np.eye(dim)

    def force(self, x, xd):
        dim = x.shape[0]
        if self.target is None:
            return np.zeros(dim)
        err = x - self.target
        norm = np.linalg.norm(err)
        scaling = self.p["conical_gain"] * np.tanh(self.p["conical_sharpness"] * norm)
        xdd = -scaling * _normalize(err)
        # Distance-gated damping: full damping when close to the target.
        gate = 0.5 * np.tanh(-self.p["damping_sharpness"] * (norm - self.p["damping_radius"])) + 0.5
        xdd = xdd - self.p["damping"] * gate * xd
        M = self.metric(x, xd)
        return -M @ xdd


class CspacePostureAttractor:
    """Geometric attractor pulling the configuration toward a rest posture.

    Port of ``CspaceAttractor`` (geometric). Helps resolve the arm's nullspace
    when the end-effector attractor under-constrains the joints. Optional.
    """

    is_forcing = False

    def __init__(self, params):
        self.p = params
        self.target = None

    def set_target(self, q_rest):
        self.target = None if q_rest is None else np.asarray(q_rest, dtype=float)

    def metric(self, x, xd):
        dim = x.shape[0]
        if self.target is None:
            return np.zeros((dim, dim))
        return self.p.get("isotropic_mass", 1.0) * np.eye(dim)

    def force(self, x, xd):
        dim = x.shape[0]
        if self.target is None:
            return np.zeros(dim)
        err = x - self.target
        scaling = self.p["conical_gain"] * np.tanh(self.p["conical_sharpness"] * np.linalg.norm(err))
        xdd_not_hd2 = -scaling * _normalize(err)
        xdd = float(xd @ xd) * xdd_not_hd2  # velocity-homogeneous => geometric
        M = self.metric(x, xd)
        return -M @ xdd


class JointLimitRepulsion:
    """Forcing barrier keeping a coordinate positive (distance-to-limit).

    Port of ``JointLimitRepulsion``. Used on both the upper task map
    ``x = upper - q`` and lower task map ``x = q - lower``, so ``x -> 0`` at the
    limit. The metric explodes as ``x -> 0`` and is velocity-gated.
    """

    is_forcing = True

    def __init__(self, params):
        self.p = params
        self._min_x_delta = (params["metric_scalar"] / params["max_metric"]) ** 0.5

    def metric(self, x, xd):
        dim = x.shape[0]
        x_delta = np.maximum(self._min_x_delta, x - self.p["metric_exploder_offset"])
        diag = self.p["metric_scalar"] / x_delta ** 2
        if self.p["velocity_gate"]:
            active = np.logical_or(xd < self.p["breakaway_velocity"], x < self.p["breakaway_distance"])
            diag = active.astype(float) * diag
        return np.diag(diag)

    def force(self, x, xd):
        dim = x.shape[0]
        damping = (xd <= 0.0).astype(float) * self.p["damping_gain"]
        xdd = self.p["soft_relu_gain"] * np.ones(dim) - damping * xd
        M = self.metric(x, xd)
        return -M @ xdd


class BodySphereRepulsion:
    """Body-sphere obstacle repulsion in the stacked ``(3*n)`` sphere space.

    Port of ``BodySphereRepulsion`` + the ``collision_response`` Warp kernel and
    ``BaseFabricRepulsion``. Both a forcing and a geometric instance are added to
    the same body-points task map (as in ``add_body_repulsion``); construct with
    ``is_forcing`` set accordingly and share one ``CollisionResponse`` between
    them so the (expensive) obstacle query runs once per step.
    """

    def __init__(self, params, is_forcing, response):
        self.p = params
        self.is_forcing = is_forcing
        self.response = response  # shared CollisionResponse

    def metric(self, x, xd):
        r = self.response
        dim = x.shape[0]
        if r.base_metric is None:
            return np.zeros((dim, dim))
        # Barrier scaling: 1/dist^2 per sphere, tiled across x/y/z rows.
        dist = np.clip(r.signed_distance, self.p["rescaled_min_dist"], None)
        expanded = np.repeat(dist, 3)  # (3n,)
        scalar = self.p["forcing_metric_scalar"] if self.is_forcing else self.p["geom_metric_scalar"]
        row_scale = (scalar / expanded ** 2)[:, None]
        return row_scale * r.base_metric

    def force(self, x, xd):
        r = self.response
        dim = x.shape[0]
        if r.accel_dir is None:
            return np.zeros(dim)
        accel_dir = r.accel_dir.reshape(-1)  # (3n,)
        if self.is_forcing:
            xdd = -self.p["constant_accel"] * accel_dir - self.p["damping_gain"] * xd
        else:
            xdd = float(xd @ xd) * (-self.p["constant_accel_geom"] * accel_dir)
        M = self.metric(x, xd)
        return -M @ xdd


class CollisionResponse:
    """Computes per-sphere base metric, acceleration direction, signed distance.

    Numpy analogue of the ``collision_response`` Warp kernel + ``BaseFabricRepulsion``.
    Call ``update(sphere_positions, sphere_velocities)`` once per control step;
    the repulsion terms then read ``base_metric`` / ``accel_dir`` / ``signed_distance``.
    """

    def __init__(self, obstacles, sphere_radii, params):
        self.obstacles = obstacles
        self.radii = np.asarray(sphere_radii, dtype=float)
        self.p = params
        self.n = len(self.radii)
        self.signed_distance = None  # (n,)
        self.accel_dir = None        # (n, 3)
        self.base_metric = None      # (3n, 3n)

    @property
    def in_collision(self):
        return self.signed_distance is not None and bool(np.any(self.signed_distance < 0.0))

    def update(self, sphere_positions, sphere_velocities):
        """Query obstacles for every sphere and build the collision response.

        Parameters
        ----------
        sphere_positions : (n, 3) array
        sphere_velocities : (n, 3) array
        """
        p = self.p
        pos = np.asarray(sphere_positions, dtype=float).reshape(self.n, 3)
        vel = np.asarray(sphere_velocities, dtype=float).reshape(self.n, 3)

        # Closest-obstacle signed distance (to surface) and normal per sphere.
        raw_dist, normals = self.obstacles.query(pos)  # normal points center->surface

        signed = raw_dist - self.radii  # shave off sphere radius
        self.signed_distance = signed

        base_accel = np.zeros((self.n, 3))
        metric = np.zeros((3 * self.n, 3 * self.n))

        for i in range(self.n):
            d_signed = signed[i]
            if not np.isfinite(d_signed) or d_signed > p["engage_depth"]:
                continue
            n = normals[i]
            if np.linalg.norm(n) < 1e-9:
                continue
            d = float(np.clip(d_signed, p["min_depth"], p["max_depth"]))

            # Accelerate away from the obstacle: base_accel points center->surface
            # (the policy negates it to push away). Matches the Warp kernel.
            base_accel[i] += p["metric_scalar"] * (1.0 / d) * n

            # Velocity gate: soften the metric when already moving away.
            switch = 1.0
            if p["velocity_gate"]:
                dir_vel = -float(n @ vel[i])  # >0 => moving away
                switch = 0.5 * (np.tanh(-p["velocity_gate_sharpness"] *
                                        (dir_vel - p["velocity_gate_offset"])) + 1.0)
            block = np.outer(n, n) * p["metric_scalar"] * (1.0 / d) * switch
            metric[3 * i:3 * i + 3, 3 * i:3 * i + 3] += block

        # Normalize base metric (Frobenius) and acceleration direction (per sphere).
        eps = 1e-6
        fro = np.linalg.norm(metric)
        self.base_metric = metric / (fro + eps)
        self.accel_dir = np.array([_normalize(base_accel[i]) for i in range(self.n)])
