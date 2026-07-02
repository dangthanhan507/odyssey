"""
Move the robot end-effector forward in the +x direction (world frame) by a
fixed distance over a fixed duration.

This uses CARTESIAN_VELOCITY control: we command a constant world-frame twist
V_WE = [wx, wy, wz, vx, vy, vz] (rotational first, translational last -- same
convention as spacemouse_teleop.py) for exactly `duration` seconds, then command
a zero twist to stop.

    vx = distance / duration

For 0.01 m (1 cm) over 10 s that is vx = 1e-3 m/s, well under the diffik
translational speed limit (0.03 m/s).
"""
import multiprocessing as mp
import time
import numpy as np
from odyssey.robot import RobotLoopDiagram, ControlMode
from odyssey.workstation import OdysseyBaseWorkstation


class MoveForwardWorkstation(OdysseyBaseWorkstation):
    def __init__(self, urdf_path, robot_description_path, distance=0.01, duration=10.0):
        OdysseyBaseWorkstation.__init__(self,
                                        robot_urdf_path=urdf_path,
                                        robot_description_path=robot_description_path,
                                        load_meshes=True,
                                        load_collision_meshes=False,
                                        host='127.0.0.1',
                                        port=8080
        )
        self.distance = distance
        self.duration = duration
        self.vx = distance / duration  # m/s

    def wait_for_hardware_state(self, timeout=10.0):
        """Block until we've received the current measured joint position from
        the robot over LCM. This is the starting point of the move: cartesian
        velocity control integrates from wherever the hardware currently is, so
        we make sure the robot loop is up and reporting state before we command
        any motion."""
        start = time.monotonic()
        while self.kuka_lcm.get_joint_position_measured() is None:
            self.handle()
            if time.monotonic() - start > timeout:
                raise RuntimeError(
                    "Timed out waiting for hardware joint state over LCM. "
                    "Is the robot loop process running?"
                )
        return np.array(self.kuka_lcm.get_joint_position_measured())

    def run(self):
        start_q = self.wait_for_hardware_state()
        np.set_printoptions(precision=4, suppress=True)
        print(f"Starting from current hardware joint position: {start_q}")
        print(f"Moving +x by {self.distance} m over {self.duration} s "
              f"(vx = {self.vx:.6f} m/s). Ctrl-C to abort.")

        # constant world-frame twist: +x translation only
        V_WE = np.array([0.0, 0.0, 0.0, self.vx, 0.0, 0.0])

        start = time.monotonic()
        while time.monotonic() - start < self.duration:
            self.send_cartesian_velocity_command(V_WE=V_WE)
            self.handle()

        # stop the robot: hold zero twist briefly so the command lands
        stop_start = time.monotonic()
        while time.monotonic() - stop_start < 0.5:
            self.send_cartesian_velocity_command(V_WE=np.zeros(6))
            self.handle()

        print("Done.")


if __name__ == '__main__':
    def robot_loop_process():
        robotloop = RobotLoopDiagram(
            config='configs/kuka_default.yaml',
            use_simulated_hardware=False,
            use_impedance=True,
            control_mode=ControlMode.CARTESIAN_VELOCITY,
        )
        robotloopdiagram = robotloop.setup_diagram()
        robotloop.run_system(robotloopdiagram)

    robot_process = mp.Process(target=robot_loop_process, daemon=True)
    robot_process.start()

    workstation = MoveForwardWorkstation(
        urdf_path='urdf/med.urdf',
        robot_description_path='urdf/lbr_description/',
        distance=0.01,   # 1 cm
        duration=10.0,   # seconds
    )

    try:
        workstation.run()
    except KeyboardInterrupt:
        # best-effort stop on abort
        for _ in range(10):
            workstation.send_cartesian_velocity_command(V_WE=np.zeros(6))
            workstation.handle()

    # kill robot process
    robot_process.terminate()
    robot_process.join()
