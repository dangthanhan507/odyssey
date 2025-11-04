import time
import numpy as np
import yourdfpy

import viser
from viser.extras import ViserUrdf
from kuka_lcm import KukaLCM

class OdysseyWorkstation:
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


if __name__ == "__main__":
    workstation = OdysseyWorkstation(
        robot_urdf_path='urdf/med.urdf',
        robot_description_path='urdf/lbr_description/',
        load_meshes=True,
        load_collision_meshes=False,
        host='127.0.0.1',
        port=8080,
    )
    
    while 1:
        workstation.handle()