import multiprocessing as mp
import viser
import numpy as np
from odyssey.robot import RobotLoopDiagram, ControlMode
from odyssey.workstation import OdysseyBaseWorkstation

class JointSliderWorkstation(OdysseyBaseWorkstation):
    def __init__(self, urdf_path, robot_description_path):
        OdysseyBaseWorkstation.__init__(self,
                                         robot_urdf_path=urdf_path,
                                         robot_description_path=robot_description_path,
                                         load_meshes=True,
                                         load_collision_meshes=False,
                                         host='127.0.0.1',
                                         port=8080
        )
        
        # add in sliders for each joint
        with self.server.gui.add_folder('Joint Sliders'):
            slider_handles = []
            for joint_name, (lower, upper) in self.viser_urdf.get_actuated_joint_limits().items():
                lower = lower if lower is not None else -np.pi
                upper = upper if upper is not None else np.pi
                
                initial_pos = 0.0 if lower < -0.1 and upper > 0.1 else (lower + upper) / 2.0
                slider = self.server.gui.add_slider(
                    label=joint_name,
                    min=lower,
                    max=upper,
                    step=1e-3,
                    initial_value=initial_pos
                )
                
                slider.on_update(
                    lambda _: self.send_joint_command([slider.value for slider in slider_handles]) 
                )
                slider_handles.append(slider)
        
        
if __name__ == '__main__':
    
    # run robot.py
    def robot_loop_process():
        robotloop = RobotLoopDiagram(config='configs/kuka_default.yaml', use_simulated_hardware=True, use_impedance=True, control_mode=ControlMode.JOINT)
        robotloopdiagram = robotloop.setup_diagram()
        robotloop.run_system(robotloopdiagram, initial_q = np.zeros(7))
    robot_process = mp.Process(target=robot_loop_process, daemon=True)
    robot_process.start()
    
    workstation = JointSliderWorkstation(
        urdf_path='urdf/med.urdf',
        robot_description_path='urdf/lbr_description/'
    )
    while True:
        try:
            workstation.handle()
        except KeyboardInterrupt:
            break
    # kill robot process
    robot_process.terminate()
    robot_process.join()