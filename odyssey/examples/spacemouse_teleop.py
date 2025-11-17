import multiprocessing as mp
import numpy as np
from odyssey.robot import RobotLoopDiagram, ControlMode
from odyssey.workstation import OdysseyBaseWorkstation
import pyspacemouse
import time

def StickDeadzone(x, y, z, deadzone = 0.1):
    stick = np.array([x,y,z])
    m = np.linalg.norm(stick)
    if m < deadzone:
        return np.array([0, 0, 0])
    over = (m - deadzone) / (1 - deadzone)
    return stick * over / m

class SpacemouseTeleopWorkstation(OdysseyBaseWorkstation):
    def __init__(self, urdf_path, robot_description_path, velocity_limit=0.1):
        OdysseyBaseWorkstation.__init__(self,
                                robot_urdf_path=urdf_path,
                                robot_description_path=robot_description_path,
                                load_meshes=True,
                                load_collision_meshes=False,
                                host='127.0.0.1',
                                port=8080
        )
        self.velocity_limit = velocity_limit
        assert pyspacemouse.open(), "Failed to open Spacemouse"
        self.stick_x = 0.0
        self.stick_y = 0.0
        self.stick_z = 0.0
        self.stick_wx = 0.0
        self.stick_wy = 0.0
        self.stick_wz = 0.0
        
        self.history_V_WE = np.zeros((50,6))
        
        # for EMA
        # self.prev_V_WE = np.zeros((6,))
    
    def handle(self):
        OdysseyBaseWorkstation.handle(self, ms=0)
        state = pyspacemouse.read()
        command_xyz = StickDeadzone(state.x, state.y, state.z)
        command_rpy = StickDeadzone(state.roll, state.pitch, state.yaw)
        self.stick_x = command_xyz[0]
        self.stick_y = command_xyz[1]
        self.stick_z = command_xyz[2]
        self.stick_wx = command_rpy[0]
        self.stick_wy = command_rpy[1]
        self.stick_wz = command_rpy[2]
        V_WE_desired = np.zeros((6,))
        V_WE_desired[0] = 0.2 * self.stick_wx
        V_WE_desired[1] = 0.2 * self.stick_wy
        V_WE_desired[2] = 0.2 * self.stick_wz
        V_WE_desired[3] = self.velocity_limit * self.stick_x
        V_WE_desired[4] = -self.velocity_limit * self.stick_y
        V_WE_desired[5] = self.velocity_limit * self.stick_z
        
        # smoothing factor of exponential moving average
        # st = alpha * xt + (1-alpha) * st-1
        # alpha = 0.3
        # V_WE_desired = V_WE_desired * alpha + (1 - alpha) * self.prev_V_WE if np.max(np.abs(V_WE_desired)) > 1e-4 else np.zeros((6,))
        # self.prev_V_WE = V_WE_desired

        # self.history_V_WE[:-1, :] = self.history_V_WE[1:, :]
        # self.history_V_WE[-1, :] = V_WE_desired
        # V_WE_desired = np.mean(self.history_V_WE, axis=0)
        
        self.send_cartesian_velocity_command(
            V_WE=V_WE_desired,
        )
        
        # time.sleep(1.0 / 1000.0)

if __name__ == '__main__':
    # run robot.py
    # initial_q = np.array([0.0, np.pi/6, 0.0, -80*np.pi/180, 0.0, np.pi/6, 0.0])
    # robotloop = RobotLoopDiagram(config='configs/kuka_default.yaml', use_simulated_hardware=True, use_impedance=True, control_mode=ControlMode.CARTESIAN_VELOCITY)
    # quat, pos = robotloop.get_arm_pose('iiwa_link_7', initial_q)
    
    def robot_loop_process():
        robotloop = RobotLoopDiagram(config='configs/kuka_default.yaml', use_simulated_hardware=False, use_impedance=True, control_mode=ControlMode.CARTESIAN_VELOCITY)
        robotloopdiagram = robotloop.setup_diagram()
        # robotloop.run_system(robotloopdiagram, initial_q = initial_q)
        robotloop.run_system(robotloopdiagram)
    
    robot_process = mp.Process(target=robot_loop_process, daemon=True)
    robot_process.start()
    
    workstation = SpacemouseTeleopWorkstation(
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