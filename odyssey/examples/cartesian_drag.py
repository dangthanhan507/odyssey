import multiprocessing as mp
import numpy as np
from odyssey.robot import RobotLoopDiagram, ControlMode
from odyssey.workstation import OdysseyBaseWorkstation

class CartesianDragWorkstation(OdysseyBaseWorkstation):
    def __init__(self, urdf_path, robot_description_path, initial_wxyz, initial_position):
        OdysseyBaseWorkstation.__init__(self,
                                        robot_urdf_path=urdf_path,
                                        robot_description_path=robot_description_path,
                                        load_meshes=True,
                                        load_collision_meshes=False,
                                        host='127.0.0.1',
                                        port=8080
        )
        
        # add in cartesian drag controls
        transform_control = self.server.scene.add_transform_controls(
            "/ee_transform",
            wxyz=initial_wxyz,
            position=initial_position,
            scale=0.5
        )
        
        transform_control.on_update(
            lambda _: self.send_pose_command(
                quat=transform_control.wxyz[[3,0,1,2]],
                pos=transform_control.position
            )
        )
        
        
if __name__ == '__main__':
    # run robot.py
    initial_q = np.array([0.0, np.pi/6, 0.0, -80*np.pi/180, 0.0, np.pi/6, 0.0])
    robotloop = RobotLoopDiagram(config='configs/kuka_default.yaml', use_simulated_hardware=True, use_impedance=True, control_mode=ControlMode.DIFFIK_POSE)
    quat, pos = robotloop.get_arm_pose('iiwa_link_7', initial_q)
    
    def robot_loop_process():
        robotloop = RobotLoopDiagram(config='configs/kuka_default.yaml', use_simulated_hardware=True, use_impedance=True, control_mode=ControlMode.DIFFIK_POSE)
        robotloopdiagram = robotloop.setup_diagram()
        robotloop.run_system(robotloopdiagram, initial_q = initial_q)
    
    robot_process = mp.Process(target=robot_loop_process, daemon=True)
    robot_process.start()
    
    workstation = CartesianDragWorkstation(
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

if __name__ == '__main__':
    pass