"""Drake-backed forward kinematics for fabric task maps.

NVIDIA's ``fabrics_sim`` computes robot-frame origins and Jacobians with Warp FK
kernels on the GPU (``taskmaps/robot_frame_origins_taskmap.py`` +
``prod/kinematics.py``). We don't need the GPU or batching for odyssey, so this
module wraps Drake's ``MultibodyPlant`` to provide the same quantities on the
CPU for a single robot:

* collision-sphere world positions and their translational Jacobians
  (the ``body_points`` task map),
* an end-effector position + Jacobian (the attractor task map),
* joint position limits.

Every task map here returns ``(x, J)`` where ``x`` is the stacked task-space
position and ``J`` is ``d x / d q`` -- exactly what the fabric pullback
(``J^T M J``, ``J^T f``) consumes.
"""

import os

import numpy as np
from pydrake.all import JacobianWrtVariable
from manipulation.station import load_scenario

from odyssey.robot import MakeFakeStation


# Collision spheres approximating the iiwa arm, defined as (link, local offset,
# radius). Offsets are in each link's body frame (meters). Radii are chosen to
# loosely envelope the arm links -- these are the odyssey analogue of the
# ``collision_sphere_frames`` / ``collision_sphere_radii`` YAML lists in
# ``fabrics_sim``'s kuka_allegro_pose_params.yaml.
DEFAULT_IIWA_COLLISION_SPHERES = [
    ("iiwa_link_1", (0.0, 0.0, 0.0), 0.10),
    ("iiwa_link_2", (0.0, 0.0, 0.0), 0.10),
    ("iiwa_link_2", (0.0, 0.06, 0.0), 0.09),
    ("iiwa_link_3", (0.0, 0.0, 0.0), 0.09),
    ("iiwa_link_3", (0.0, 0.0, 0.1), 0.08),
    ("iiwa_link_4", (0.0, 0.0, 0.0), 0.09),
    ("iiwa_link_4", (0.0, 0.06, 0.0), 0.08),
    ("iiwa_link_5", (0.0, 0.0, 0.0), 0.08),
    ("iiwa_link_5", (0.0, 0.0, 0.1), 0.07),
    ("iiwa_link_6", (0.0, 0.0, 0.0), 0.08),
    ("iiwa_link_7", (0.0, 0.0, 0.05), 0.07),
]


class RobotKinematics:
    """Forward kinematics + Jacobians for the iiwa via Drake.

    Parameters
    ----------
    config : str, optional
        Path to the scenario YAML used to build the plant. Defaults to the
        packaged ``configs/kuka_default.yaml``.
    collision_spheres : list of (link_name, offset, radius), optional
        Overrides ``DEFAULT_IIWA_COLLISION_SPHERES``.
    ee_frame : str
        Frame name used for the end-effector attractor task map.
    """

    def __init__(self, config=None, collision_spheres=None, ee_frame="iiwa_link_7"):
        if config is None:
            pkg_dir = os.path.dirname(os.path.abspath(__file__))
            pkg_dir = os.path.dirname(pkg_dir)  # .../odyssey
            config = os.path.join(pkg_dir, "configs", "kuka_default.yaml")

        scenario = load_scenario(filename=config)
        self._diagram = MakeFakeStation(scenario)
        self._plant = self._diagram.GetSubsystemByName("plant")
        self._context = self._plant.CreateDefaultContext()
        self._world = self._plant.world_frame()

        self.num_joints = self._plant.num_positions()

        self._spheres = collision_spheres or DEFAULT_IIWA_COLLISION_SPHERES
        self._sphere_frames = [self._plant.GetFrameByName(name) for name, _, _ in self._spheres]
        self._sphere_offsets = [np.asarray(off, dtype=float) for _, off, _ in self._spheres]
        self.sphere_radii = np.array([r for _, _, r in self._spheres], dtype=float)

        self._ee_frame = self._plant.GetFrameByName(ee_frame)

    @property
    def num_spheres(self):
        return len(self._spheres)

    def _set_q(self, q):
        self._plant.SetPositions(self._context, np.asarray(q, dtype=float))

    def joint_limits(self):
        """Return (lower, upper) joint position limits as (nq,) arrays."""
        return (self._plant.GetPositionLowerLimits(),
                self._plant.GetPositionUpperLimits())

    def body_points_taskmap(self, q):
        """Collision-sphere positions and Jacobian.

        Returns
        -------
        x : (3 * num_spheres,) array
            Stacked world positions of every collision-sphere center.
        J : (3 * num_spheres, nq) array
            Stacked translational Jacobians ``d x / d q``.
        """
        self._set_q(q)
        n = self.num_spheres
        x = np.zeros(3 * n)
        J = np.zeros((3 * n, self.num_joints))
        for i, (frame, offset) in enumerate(zip(self._sphere_frames, self._sphere_offsets)):
            X_WF = frame.CalcPoseInWorld(self._context)
            x[3 * i:3 * i + 3] = X_WF.translation() + X_WF.rotation() @ offset
            J[3 * i:3 * i + 3, :] = self._plant.CalcJacobianTranslationalVelocity(
                self._context, JacobianWrtVariable.kQDot, frame, offset, self._world, self._world)
        return x, J

    def ee_taskmap(self, q):
        """End-effector position and translational Jacobian.

        Returns
        -------
        x : (3,) array
            World position of the end-effector frame origin.
        J : (3, nq) array
            Translational Jacobian ``d x / d q``.
        """
        self._set_q(q)
        X_WE = self._ee_frame.CalcPoseInWorld(self._context)
        x = X_WE.translation()
        J = self._plant.CalcJacobianTranslationalVelocity(
            self._context, JacobianWrtVariable.kQDot, self._ee_frame,
            np.zeros(3), self._world, self._world)
        return x, J

    def sphere_positions(self, q):
        """Convenience: (num_spheres, 3) array of collision-sphere centers."""
        x, _ = self.body_points_taskmap(q)
        return x.reshape(self.num_spheres, 3)
