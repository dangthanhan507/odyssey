import multiprocessing as mp
import time
import numpy as np
from pydrake.all import RigidTransform, RotationMatrix, Quaternion
from odyssey.robot import RobotLoopDiagram, ControlMode
from odyssey.workstation import OdysseyBaseWorkstation


class OculusTeleopWorkstation(OdysseyBaseWorkstation):
    """
    Teleop the kuka end-effector with an Oculus (Quest) VR controller.

    Hold the right controller's grip button to enable motion. While held, the
    relative motion of the controller (measured from where it was when the grip
    was first pressed) is applied to the end-effector pose that the arm had when
    the grip was first pressed. Releasing the grip re-anchors both the base
    controller pose and the base end-effector pose, so motion always resumes
    from the current state (this lets you "clutch" like a mouse lift).

    NOTE: this robot (urdf/med.urdf) has no gripper, so the right trigger
    (which the old repo used for the gripper) is read but currently unused.
    """

    def __init__(self, urdf_path, robot_description_path, initial_wxyz, initial_position):
        OdysseyBaseWorkstation.__init__(self,
                                        robot_urdf_path=urdf_path,
                                        robot_description_path=robot_description_path,
                                        load_meshes=True,
                                        load_collision_meshes=False,
                                        host='127.0.0.1',
                                        port=8080
        )

        # lazily import so that the pose math in this module is importable /
        # testable without oculus_reader (which needs a physical Quest + ADB).
        from oculus_reader.reader import OculusReader
        self.reader = OculusReader()
        time.sleep(0.3)

        # anchor the base ee pose to the arm's current pose
        self.base_ee_pose = RigidTransform(
            RotationMatrix(Quaternion(np.asarray(initial_wxyz))),
            np.asarray(initial_position),
        )
        self.base_controller_pose = None
        self.prev_grip = False
        self.commanded_pose = self.base_ee_pose

    @staticmethod
    def compute_commanded_pose(base_ee_pose: RigidTransform,
                               base_controller_pose: RigidTransform,
                               current_controller_pose: RigidTransform,
                               grip_held: bool,
                               prev_grip: bool):
        """
        Core teleop math (factored out so it can be validated without hardware).

        Returns (commanded_pose, new_base_ee_pose, new_base_controller_pose).

        - grip not held: command == base ee pose, and the base controller pose
          keeps tracking the current controller pose so it is always fresh.
        - grip held: command == base_ee_pose @ (base_controller_pose^-1 @
          current_controller_pose), i.e. the base ee pose plus the relative
          motion of the controller since the grip was pressed. The base ee pose
          and base controller pose stay FIXED during a continuous hold so motion
          does not double-accumulate frame to frame.
        - grip released (prev held, now not): re-anchor the base ee pose to the
          last commanded pose (computed from the relative motion at release) and
          re-anchor the base controller pose to the current controller pose, so
          the next grip press moves relative to the new state without a jump.
        """
        if base_controller_pose is None:
            base_controller_pose = current_controller_pose

        if grip_held:
            rel_controller = base_controller_pose.inverse() @ current_controller_pose
            commanded_pose = base_ee_pose @ rel_controller
            # anchors stay fixed during the hold; command reflects relative motion.
            return commanded_pose, base_ee_pose, base_controller_pose

        if prev_grip and not grip_held:
            # grip released: bake the last relative motion into the base ee pose
            # and re-anchor the controller to the current pose.
            rel_controller = base_controller_pose.inverse() @ current_controller_pose
            commanded_pose = base_ee_pose @ rel_controller
            return commanded_pose, commanded_pose, current_controller_pose

        # grip not held: keep anchor fresh, command == base ee pose
        return base_ee_pose, base_ee_pose, current_controller_pose

    def handle(self):
        OdysseyBaseWorkstation.handle(self, ms=0)

        transform_dict, buttons_dict = self.reader.get_transformations_and_buttons()
        if 'r' not in transform_dict:
            # no controller data yet
            return

        current_controller_pose = RigidTransform(transform_dict['r'])
        grip_held = buttons_dict['rightGrip'][0] > 0.5
        # trigger = buttons_dict['rightTrig'][0] > 0.5  # gripper (no gripper on this robot)

        commanded_pose, self.base_ee_pose, self.base_controller_pose = \
            self.compute_commanded_pose(
                self.base_ee_pose,
                self.base_controller_pose,
                current_controller_pose,
                grip_held,
                self.prev_grip,
            )
        self.commanded_pose = commanded_pose
        self.prev_grip = grip_held

        quat = commanded_pose.rotation().ToQuaternion().wxyz()
        pos = commanded_pose.translation()
        self.send_pose_command(quat=quat, pos=pos)


if __name__ == '__main__':
    # run robot.py
    initial_q = np.array([0.0, np.pi/6, 0.0, -80*np.pi/180, 0.0, np.pi/6, 0.0])
    robotloop = RobotLoopDiagram(config='configs/kuka_default.yaml', use_simulated_hardware=True, use_impedance=True, control_mode=ControlMode.DIFFIK_POSE)
    quat, pos = robotloop.get_arm_pose('iiwa_link_7', initial_q)

    def robot_loop_process():
        robotloop = RobotLoopDiagram(config='configs/kuka_default.yaml', use_simulated_hardware=True, use_impedance=True, control_mode=ControlMode.DIFFIK_POSE)
        robotloopdiagram = robotloop.setup_diagram()
        robotloop.run_system(robotloopdiagram, initial_q=initial_q)

    robot_process = mp.Process(target=robot_loop_process, daemon=True)
    robot_process.start()

    workstation = OculusTeleopWorkstation(
        urdf_path='urdf/med.urdf',
        robot_description_path='urdf/lbr_description/',
        initial_wxyz=quat,
        initial_position=pos
    )

    while True:
        try:
            workstation.handle()
        except KeyboardInterrupt:
            break
    # kill robot process
    robot_process.terminate()
    robot_process.join()
