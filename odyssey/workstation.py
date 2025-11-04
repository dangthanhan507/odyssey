import numpy as np
import yourdfpy
import viser
from viser.extras import ViserUrdf
from odyssey.msgs.lcm_msgs import lcmt_iiwa_status, iiwa_commands_t
import lcm


# this allows access to kuka iiwa status messages over lcm
# we can use this outside of drake simulation to prevent overhead in control loop
class KukaLCM:
    def __init__(self):
        self.lcm = lcm.LCM()
        self.sub = self.lcm.subscribe('IIWA_STATUS', lambda channel, data: self.msg_handler(channel, data))

        self.joint_commanded  = None
        self.joint_measured   = None
        self.torque_commanded = None
        self.torque_measured  = None
        self.torque_external  = None
        self.joint_velocity   = None
        
    def msg_handler(self, channel, data):
        fri_msg = lcmt_iiwa_status.decode(data)
        
        self.joint_commanded  = fri_msg.joint_position_commanded
        self.joint_measured   = fri_msg.joint_position_measured
        self.torque_commanded = fri_msg.joint_torque_commanded
        self.torque_measured  = fri_msg.joint_torque_measured
        self.torque_external  = fri_msg.joint_torque_external
        self.joint_velocity   = fri_msg.joint_velocity_estimated
    
    def get_joint_position_commanded(self):
        return self.joint_commanded

    def get_joint_position_measured(self):
        return self.joint_measured
    
    def get_joint_velocity_estimated(self):
        return self.joint_velocity
    
    def get_joint_torque_commanded(self):
        return self.torque_commanded
    
    def get_joint_torque_measured(self):
        return self.torque_measured
    
    def get_joint_torque_external(self):
        return self.torque_external
    
    def handle(self):
        self.lcm.handle_timeout(10)

    def send_joint_command(self, joint_positions, torque):
        msg = iiwa_commands_t()
        msg.desired_joints = joint_positions
        msg.desired_torque = torque
        
        msg.desired_cartesian_vel = np.zeros(6).tolist()
        msg.desired_quat = [0, 0, 0, 1]
        msg.desired_pos = [0, 0, 0]
        
        self.lcm.publish('ODYSSEY_IIWA_TARGETS', msg.encode())
    
    def send_pose_command(self, quat, pos, torque):
        msg = iiwa_commands_t()
        msg.desired_quat = quat.tolist()
        msg.desired_pos = pos.tolist()
        msg.desired_torque = torque
        
        msg.desired_joints = np.zeros(7).tolist()
        msg.desired_cartesian_vel = np.zeros(6).tolist()
        self.lcm.publish('ODYSSEY_IIWA_TARGETS', msg.encode())
    
    def send_cartesian_velocity_command(self, V_WE, torque):
        msg = iiwa_commands_t()
        msg.desired_cartesian_vel = V_WE.tolist()
        msg.desired_torque = torque
        
        msg.desired_joints = np.zeros(7).tolist()
        msg.desired_quat = [0, 0, 0, 1]
        msg.desired_pos = [0, 0, 0]
        self.lcm.publish('ODYSSEY_IIWA_TARGETS', msg.encode())
    
'''
    Base workstation gives standard functionality to visualize a robot in viser server.
'''
class OdysseyBaseWorkstation:
    def __init__(self, 
                 robot_urdf_path,
                 robot_description_path,
                 load_meshes=True,
                 load_collision_meshes=False,
                 host='127.0.0.1',
                 port=8080):

        # setup viser with URDF
        self.server = viser.ViserServer(host=host, port=port)
        
        self.kuka_lcm = KukaLCM()
        
        urdf = yourdfpy.URDF.load(
            robot_urdf_path,
            mesh_dir=robot_description_path,
            load_meshes=load_meshes,
            build_scene_graph=load_meshes,
            load_collision_meshes=load_collision_meshes,
            build_collision_scene_graph=load_collision_meshes,
        )
        self.viser_urdf = ViserUrdf(
            self.server,
            urdf_or_path=urdf,
            load_meshes=load_meshes,
            load_collision_meshes=load_collision_meshes,
            collision_mesh_color_override=(1.0, 0.0, 0.0, 0.5),
        )
        
        trimesh_scene = self.viser_urdf._urdf.scene or self.viser_urdf._urdf.collision_scene
        self.server.scene.add_grid(
            "/grid",
            width=2,
            height=2,
            position=(
                0.0,
                0.0,
                # Get the minimum z value of the trimesh scene.
                trimesh_scene.bounds[0, 2] if trimesh_scene is not None else 0.0,
            ),
        )
        self.viser_urdf.show_visual = True
        self.viser_urdf.show_collision = True
    
    def update_robot_joints(self, joint_positions):
        self.viser_urdf.update_cfg(np.array(joint_positions))

    def handle(self):
        self.kuka_lcm.handle()
        joint_position = self.kuka_lcm.get_joint_position_measured()
        if joint_position is not None:
            self.update_robot_joints(joint_position)

    def send_joint_command(self, joint_positions, torque = np.zeros(7)):
        self.kuka_lcm.send_joint_command(joint_positions, torque)
    
    def send_pose_command(self, quat, pos, torque = np.zeros(7)):
        self.kuka_lcm.send_pose_command(quat, pos, torque)
    
    def send_cartesian_velocity_command(self, V_WE, torque = np.zeros(7)):
        self.kuka_lcm.send_cartesian_velocity_command(V_WE, torque)

if __name__ == "__main__":
    workstation = OdysseyBaseWorkstation(
        robot_urdf_path='urdf/med.urdf',
        robot_description_path='urdf/lbr_description/',
        load_meshes=True,
        load_collision_meshes=False,
        host='127.0.0.1',
        port=8080,
    )
    
    while 1:
        workstation.handle()