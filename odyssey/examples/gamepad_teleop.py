import multiprocessing as mp
import numpy as np
from odyssey.robot import RobotLoopDiagram, ControlMode
from odyssey.workstation import OdysseyBaseWorkstation

# Schunk WSG gripper is commanded over LCM (channel SCHUNK_WSG_COMMAND) using
# Drake's lcmt_schunk_wsg_command, and consumed by the separate
# drake-schunk-driver daemon. We publish on odyssey's LCM bus so it shares the
# same multicast provider as the iiwa commands.
import lcm
from drake import lcmt_schunk_wsg_command


def stick_deadzone(x, y, z, deadzone=0.1):
    """Radial deadzone for a 3-axis stick, matching spacemouse_teleop."""
    stick = np.array([x, y, z])
    m = np.linalg.norm(stick)
    if m < deadzone:
        return np.array([0.0, 0.0, 0.0])
    over = (m - deadzone) / (1 - deadzone)
    return stick * over / m


class SchunkGripperLCM:
    """Publishes lcmt_schunk_wsg_command on odyssey's LCM bus.

    The drake-schunk-driver daemon must be listening on the same bus/channel
    (SCHUNK_WSG_COMMAND). target_position_mm is the commanded opening; force is
    the grasp force limit in Newtons.
    """

    def __init__(self, channel='SCHUNK_WSG_COMMAND'):
        # same multicast provider as odyssey's KukaLCM / KukaLoopLCM
        self.lcm = lcm.LCM(provider="udpm://239.241.129.92:20185?ttl=0")
        self.channel = channel

    def send(self, target_position_mm, force):
        msg = lcmt_schunk_wsg_command()
        msg.utime = 0
        msg.target_position_mm = float(target_position_mm)
        msg.force = float(force)
        self.lcm.publish(self.channel, msg.encode())


class GamepadTeleopWorkstation(OdysseyBaseWorkstation):
    """
    Teleop the kuka end-effector with a gamepad, and drive a Schunk WSG gripper
    with the right trigger.

    Arm (cartesian velocity, like spacemouse_teleop):
      - left stick  -> translation in x / y
      - right stick -> translation in z (vertical axis) and one rotation
      - the exact axis indices are gamepad dependent; see AXIS_* below and remap
        for your controller (defaults target an Xbox-style layout via pygame).

    Gripper (proportional):
      - right trigger 0..1 maps linearly to the gripper opening from
        gripper_open_mm (released) to gripper_closed_mm (fully squeezed).

    Like the other teleop examples, the gamepad (pygame) is imported lazily so
    the mapping math in this module is importable / testable without a joystick.
    """

    # pygame axis indices for a typical Xbox-style controller. Remap as needed.
    AXIS_LEFT_X = 0
    AXIS_LEFT_Y = 1
    AXIS_RIGHT_X = 3
    AXIS_RIGHT_Y = 4
    AXIS_RIGHT_TRIGGER = 5  # rest ~= -1.0, fully pressed ~= +1.0 on many pads

    def __init__(self, urdf_path, robot_description_path,
                 velocity_limit=0.1,
                 gripper_open_mm=100.0,
                 gripper_closed_mm=0.0,
                 gripper_force=40.0,
                 deadzone=0.1):
        OdysseyBaseWorkstation.__init__(self,
                                        robot_urdf_path=urdf_path,
                                        robot_description_path=robot_description_path,
                                        load_meshes=True,
                                        load_collision_meshes=False,
                                        host='127.0.0.1',
                                        port=8080
        )
        self.velocity_limit = velocity_limit
        self.gripper_open_mm = gripper_open_mm
        self.gripper_closed_mm = gripper_closed_mm
        self.gripper_force = gripper_force
        self.deadzone = deadzone

        self.gripper = SchunkGripperLCM()

        # lazily import so this module's mapping math imports without a joystick
        import pygame
        pygame.init()
        pygame.joystick.init()
        assert pygame.joystick.get_count() >= 1, "No gamepad detected."
        self._pygame = pygame
        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()

    @staticmethod
    def trigger_to_gripper_mm(trigger_value, open_mm, closed_mm):
        """Map a normalized trigger value in [0, 1] to a gripper opening in mm.

        trigger 0.0 (released) -> open_mm; trigger 1.0 (squeezed) -> closed_mm.
        Values are clipped so out-of-range triggers stay within [closed, open].
        """
        t = float(np.clip(trigger_value, 0.0, 1.0))
        return open_mm + t * (closed_mm - open_mm)

    @staticmethod
    def normalize_trigger(raw_axis_value):
        """Many pygame triggers rest at -1.0 and read +1.0 fully pressed.

        Convert that [-1, 1] range to a [0, 1] normalized trigger.
        """
        return (float(raw_axis_value) + 1.0) / 2.0

    def sticks_to_velocity(self, lx, ly, rx, ry):
        """Map stick axes to an end-effector twist V_WE (angular xyz, linear xyz).

        Mirrors spacemouse_teleop's scaling: linear from the sticks scaled by
        velocity_limit, with a radial deadzone applied first.
        """
        # left stick -> x/y translation, right stick vertical -> z translation
        lin = stick_deadzone(lx, ly, ry, deadzone=self.deadzone)
        # right stick horizontal -> yaw-ish angular term
        ang = stick_deadzone(rx, 0.0, 0.0, deadzone=self.deadzone)

        V_WE = np.zeros(6)
        V_WE[0] = 0.2 * ang[0]                      # angular x
        V_WE[3] = self.velocity_limit * lin[0]      # linear x
        V_WE[4] = -self.velocity_limit * lin[1]     # linear y
        V_WE[5] = self.velocity_limit * lin[2]      # linear z
        return V_WE

    def handle(self):
        OdysseyBaseWorkstation.handle(self, ms=0)

        # pump pygame's event queue so axis reads are fresh
        self._pygame.event.pump()

        lx = self.joystick.get_axis(self.AXIS_LEFT_X)
        ly = self.joystick.get_axis(self.AXIS_LEFT_Y)
        rx = self.joystick.get_axis(self.AXIS_RIGHT_X)
        ry = self.joystick.get_axis(self.AXIS_RIGHT_Y)
        trig_raw = self.joystick.get_axis(self.AXIS_RIGHT_TRIGGER)

        V_WE = self.sticks_to_velocity(lx, ly, rx, ry)
        self.send_cartesian_velocity_command(V_WE=V_WE)

        trigger = self.normalize_trigger(trig_raw)
        target_mm = self.trigger_to_gripper_mm(
            trigger, self.gripper_open_mm, self.gripper_closed_mm
        )
        self.gripper.send(target_position_mm=target_mm, force=self.gripper_force)


if __name__ == '__main__':
    def robot_loop_process():
        robotloop = RobotLoopDiagram(config='configs/kuka_default.yaml', use_simulated_hardware=False, use_impedance=True, control_mode=ControlMode.CARTESIAN_VELOCITY)
        robotloopdiagram = robotloop.setup_diagram()
        robotloop.run_system(robotloopdiagram)

    robot_process = mp.Process(target=robot_loop_process, daemon=True)
    robot_process.start()

    workstation = GamepadTeleopWorkstation(
        urdf_path='urdf/med.urdf',
        robot_description_path='urdf/lbr_description/',
        velocity_limit=0.1
    )

    while True:
        try:
            workstation.handle()
        except KeyboardInterrupt:
            break
    # kill robot process
    robot_process.terminate()
    robot_process.join()
