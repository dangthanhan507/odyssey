from pydrake.all import (
    LeafSystem,
    RigidTransform,
    Value,
    DiagramBuilder,
    MultibodyPlant,
    Multiplexer,
    ValueProducer,
    AbstractValue,
    Quaternion,
    ConstantValueSource,
    PassThrough
)
import numpy as np
from odyssey.msgs.lcm_msgs import lcmt_iiwa_status
from drake import lcmt_iiwa_status
from odyssey.robot_lcm import KukaLoopLCM
from enum import Enum
from odyssey.utils import AddIiwaDifferentialIK, VelocityDiffIK

'''
    The robot is running joint position / joint impedance.
    These control modes are defined to switch between what kind of targets we want to supply to the robot.
'''
class ControlMode(Enum):
    JOINT = 1
    DIFFIK_POSE = 2
    CARTESIAN_VELOCITY = 3

class ExternalSystem(LeafSystem):
    def __init__(self, 
                 plant: MultibodyPlant,
                ee_frame = "iiwa_link_7",
                 simulated: bool = False,
                 control_mode: ControlMode = ControlMode.JOINT,
                 max_joint_speed = 30.0 * np.pi / 180.0,
                 ):
        LeafSystem.__init__(self)
        
        self.simulated = simulated
        self._plant = plant
        self._plant_context = plant.CreateDefaultContext()
        self.ee_frame = ee_frame
        self.control_mode = control_mode
        self.max_joint_speed = max_joint_speed
        self.prev_time = None
        
        self.desired_quat = None
        self.desired_pos = None
        self.feedforward_torque = None
        self.first_ee_pose = None
        
        self.prev_V_WE = np.zeros(6)
        
        self.lcm = KukaLoopLCM()
        
        # Inputs
        self.DeclareVectorInputPort("iiwa_position", 7)
        self.DeclareVectorInputPort("iiwa_velocity", 7)
        self.DeclareVectorInputPort("iiwa_torque_external", 7)
        self.DeclareVectorInputPort("iiwa_position_commanded", 7)
        
        self._calc_external = self.DeclareCacheEntry(
            description="calculate out for robot",
            value_producer=ValueProducer(
                allocate=lambda: AbstractValue.Make(tuple()),
                calc=self.CalcExternalFn),
            )
        
        # Outputs
        if self.control_mode == ControlMode.JOINT:
            self.DeclareVectorOutputPort(
                "desired_joints",
                7,
                self.calc_desired,
                prerequisites_of_calc={self._calc_external.ticket()}
            )
        
        elif self.control_mode == ControlMode.DIFFIK_POSE:
            self.DeclareAbstractOutputPort(
                "desired_pose",
                lambda: Value(RigidTransform()),
                self.calc_desired,
                prerequisites_of_calc={self._calc_external.ticket()}
            )
        
        elif self.control_mode == ControlMode.CARTESIAN_VELOCITY:
            self.DeclareVectorOutputPort(
                "desired_velocity",
                6,
                self.calc_desired,
                prerequisites_of_calc={self._calc_external.ticket()}
            )
        
        self.DeclareVectorOutputPort(
            "feedforward_torque",
            7,
            self.calc_feedforward_torque,
            prerequisites_of_calc={self._calc_external.ticket()}
        )
    
    def CalcExternalFn(self, context, output):
        self.lcm.handle()
        
        position = self.GetInputPort("iiwa_position").Eval(context)
        velocity = self.GetInputPort("iiwa_velocity").Eval(context)
        torque_external = self.GetInputPort("iiwa_torque_external").Eval(context)
        position_commanded = self.GetInputPort("iiwa_position_commanded").Eval(context)
        
        # if simulated, the error may be large on first step but thats ok, otherwise check for large errors
        if ((self.simulated and self.desired_quat is not None) or not self.simulated) and np.max(np.abs(position_commanded - position)) > 30.0 * np.pi/180: # cannot travel more than 10 deg in one step
            raise RuntimeError("Large position error between commanded and measured! max joint error: {}".format(np.max(np.abs(position_commanded - position)) * 180/np.pi))
        
        if self.simulated:
            # publish iiwa_status message for programs to use
            msg = lcmt_iiwa_status()
            msg.num_joints = 7
            msg.joint_position_commanded = position_commanded.tolist()
            msg.joint_position_measured = position.tolist()
            msg.joint_position_ipo = [0.0]*7
            msg.joint_torque_commanded = [0.0]*7
            msg.joint_torque_measured = [0.0]*7
            msg.joint_torque_external = torque_external.tolist()
            msg.joint_velocity_estimated = velocity.tolist()
            self.lcm.lcm.publish('IIWA_STATUS', lcmt_iiwa_status.encode(msg))

        if self.control_mode == ControlMode.JOINT:
            q_desired = self.lcm.get_desired_joints()
            if not q_desired is None:
                q_desired = np.array(q_desired)
                
                if self.prev_time is None:
                    self.prev_time = context.get_time()
                prev_time = self.prev_time
                curr_time = context.get_time()
                delta_t = max(curr_time - prev_time, 1e-3)
                qdot_approx = (q_desired - position) / delta_t
                qdot_approx = np.clip(qdot_approx, -self.max_joint_speed, self.max_joint_speed)
                # q_desired = position + qdot_approx * delta_t
                
                self.prev_time = context.get_time()
                
            else:
                q_desired = position if np.max(np.abs(position_commanded - position)) > 30.0 * np.pi/180 or self.simulated else position_commanded
            
            desired_out = q_desired
            
        elif self.control_mode == ControlMode.DIFFIK_POSE:
            
            self._plant.SetPositions(self._plant_context, position)
            ee_pose = self._plant.GetFrameByName(self.ee_frame).CalcPoseInWorld(self._plant_context)
            
            if self.first_ee_pose is None:
                self.first_ee_pose = ee_pose
            
            self.desired_quat = self.lcm.get_desired_quat() # [w,x,y,z]
            self.desired_pos = self.lcm.get_desired_pos()
            self.feedforward_torque = self.lcm.get_feedforward_torque()
            
            # normalize self.desired_quat
            # if not self.desired_quat is None:
            #     norm = np.linalg.norm(self.desired_quat)
            #     if norm > 1e-6:
            #         self.desired_quat = (np.array(self.desired_quat) / norm).tolist()
            
            self.desired_pose = RigidTransform(
                quaternion=Quaternion(self.desired_quat[0], self.desired_quat[1], self.desired_quat[2], self.desired_quat[3]),
                p=self.desired_pos
            ) if not (self.desired_quat is None or self.desired_pos is None) else None
            
            desired_pose = self.desired_pose if not self.desired_pose is None else self.first_ee_pose
            
            desired_out = desired_pose
            # print("Debug DiffIK - End Effector Position Commanded: pos {}, quat {}".format(desired_pose.translation(), desired_pose.rotation().ToQuaternion().wxyz()))
        
        elif self.control_mode == ControlMode.CARTESIAN_VELOCITY:
            self.desired_cartesian_vel = self.lcm.get_desired_cartesian_vel()
            desired_out = np.array(self.desired_cartesian_vel) if not self.desired_cartesian_vel is None else np.zeros(6)
        
        feedforward_torque = self.feedforward_torque if not self.feedforward_torque is None else np.zeros(7)
        output.set_value((desired_out, feedforward_torque))
    
    def calc_desired(self, context, output):
        out, _ = self._calc_external.Eval(context)
        output.set_value(out)
        
    def calc_feedforward_torque(self, context, output):
        _, torque = self._calc_external.Eval(context)
        output.SetFromVector(torque)


def joint_control_diagram(plant: MultibodyPlant, simulated: bool = False, max_joint_speed = 30.0 * np.pi / 180.0):
    builder = DiagramBuilder()
    
    iiwa_pos_passblock = builder.AddSystem(PassThrough(7))
    iiwa_vel_passblock = builder.AddSystem(PassThrough(7))
    iiwa_pos_cmd_passblock = builder.AddSystem(PassThrough(7))
    iiwa_torque_ext_passblock = builder.AddSystem(PassThrough(7))
    
    builder.ExportInput(iiwa_pos_passblock.get_input_port(), "iiwa.position_measured")
    builder.ExportInput(iiwa_vel_passblock.get_input_port(), "iiwa.velocity_estimated")
    builder.ExportInput(iiwa_pos_cmd_passblock.get_input_port(), "iiwa.position_commanded")
    builder.ExportInput(iiwa_torque_ext_passblock.get_input_port(), "iiwa.torque_external")
    
    external_sys_block = builder.AddSystem(
        ExternalSystem(
            plant,
            simulated=simulated,
            control_mode=ControlMode.JOINT,
            max_joint_speed=max_joint_speed
        )
    )
    builder.Connect(
        iiwa_pos_passblock.get_output_port(),
        external_sys_block.GetInputPort("iiwa_position")
    )
    builder.Connect(
        iiwa_vel_passblock.get_output_port(),
        external_sys_block.GetInputPort("iiwa_velocity")
    )
    builder.Connect(
        iiwa_torque_ext_passblock.get_output_port(),
        external_sys_block.GetInputPort("iiwa_torque_external")
    )
    builder.Connect(
        iiwa_pos_cmd_passblock.get_output_port(),
        external_sys_block.GetInputPort("iiwa_position_commanded")
    )
    
    builder.ExportOutput(
        external_sys_block.GetOutputPort('desired_joints'), 
        "iiwa.position"
    )
    builder.ExportOutput(
        external_sys_block.GetOutputPort("feedforward_torque"),
        "feedforward_torque"
    )
    
    diagram = builder.Build()
    diagram.set_name("JointControlDiagram")
    return diagram



def diffik_pose_diagram(plant: MultibodyPlant, simulated: bool = False, ee_frame = 'iiwa_link_7'):
    builder = DiagramBuilder()
    
    iiwa_pos_passblock = builder.AddSystem(PassThrough(7))
    iiwa_vel_passblock = builder.AddSystem(PassThrough(7))
    
    iiwa_pos_cmd_passblock = builder.AddSystem(PassThrough(7))
    iiwa_torque_ext_passblock = builder.AddSystem(PassThrough(7))
    
    builder.ExportInput(iiwa_pos_passblock.get_input_port(), "iiwa.position_measured")
    builder.ExportInput(iiwa_vel_passblock.get_input_port(), "iiwa.velocity_estimated")
    builder.ExportInput(iiwa_pos_cmd_passblock.get_input_port(), "iiwa.position_commanded")
    builder.ExportInput(iiwa_torque_ext_passblock.get_input_port(), "iiwa.torque_external")
    
    external_sys_block = builder.AddSystem(
        ExternalSystem(
            plant,
            ee_frame=ee_frame,
            simulated=simulated,
            control_mode=ControlMode.DIFFIK_POSE
        )
    )
    
    builder.Connect(
        iiwa_pos_passblock.get_output_port(),
        external_sys_block.GetInputPort("iiwa_position")
    )
    builder.Connect(
        iiwa_vel_passblock.get_output_port(),
        external_sys_block.GetInputPort("iiwa_velocity")
    )
    builder.Connect(
        iiwa_torque_ext_passblock.get_output_port(),
        external_sys_block.GetInputPort("iiwa_torque_external")
    )
    builder.Connect(
        iiwa_pos_cmd_passblock.get_output_port(),
        external_sys_block.GetInputPort("iiwa_position_commanded")
    )
    
    iiwa_state = builder.AddSystem(Multiplexer([7,7]))
    builder.Connect(
        iiwa_pos_passblock.get_output_port(),
        iiwa_state.get_input_port(0)
    )
    builder.Connect(
        iiwa_vel_passblock.get_output_port(),
        iiwa_state.get_input_port(1)
    )

    diffik_block = AddIiwaDifferentialIK(builder, plant, plant.GetFrameByName(ee_frame), xyz_speed_limit=1.0, angular_speed_limit=180.0 * np.pi / 180, time_step=plant.time_step())       
    builder.Connect(
        iiwa_state.get_output_port(),
        diffik_block.GetInputPort("robot_state"),
    )
    use_state = builder.AddSystem(ConstantValueSource(Value(True)))
    builder.Connect(
        use_state.get_output_port(),
        diffik_block.GetInputPort("use_robot_state"),
    )
    
    builder.ExportOutput(
        diffik_block.get_output_port(), 
        "iiwa.position"
    )
    
    builder.Connect(
        external_sys_block.GetOutputPort("desired_pose"),
        diffik_block.GetInputPort("X_AE_desired")
    )
    
    builder.ExportOutput(
        external_sys_block.GetOutputPort("feedforward_torque"),
        "feedforward_torque"
    )

    diagram = builder.Build()
    diagram.set_name("DiffIKPoseControlDiagram")
    return diagram


def cartesian_velocity_diagram(plant: MultibodyPlant, ee_frame='iiwa_link_7', simulated: bool = False, vel_limit=0.03):
    builder = DiagramBuilder()
    
    iiwa_pos_passblock = builder.AddSystem(PassThrough(7))
    iiwa_vel_passblock = builder.AddSystem(PassThrough(7))
    
    iiwa_pos_cmd_passblock = builder.AddSystem(PassThrough(7))
    iiwa_torque_ext_passblock = builder.AddSystem(PassThrough(7))
    
    builder.ExportInput(iiwa_pos_passblock.get_input_port(), "iiwa.position_measured")
    builder.ExportInput(iiwa_vel_passblock.get_input_port(), "iiwa.velocity_estimated")
    builder.ExportInput(iiwa_pos_cmd_passblock.get_input_port(), "iiwa.position_commanded")
    builder.ExportInput(iiwa_torque_ext_passblock.get_input_port(), "iiwa.torque_external")
    
    external_sys_block = builder.AddSystem(
        ExternalSystem(
            plant,
            ee_frame=ee_frame,
            simulated=simulated,
            control_mode=ControlMode.CARTESIAN_VELOCITY
        )
    )
    
    builder.Connect(
        iiwa_pos_passblock.get_output_port(),
        external_sys_block.GetInputPort("iiwa_position")
    )
    builder.Connect(
        iiwa_vel_passblock.get_output_port(),
        external_sys_block.GetInputPort("iiwa_velocity")
    )
    builder.Connect(
        iiwa_torque_ext_passblock.get_output_port(),
        external_sys_block.GetInputPort("iiwa_torque_external")
    )
    builder.Connect(
        iiwa_pos_cmd_passblock.get_output_port(),
        external_sys_block.GetInputPort("iiwa_position_commanded")
    )
    
    vel_diffik_block = builder.AddSystem(
        VelocityDiffIK(
            plant,
            frame_name=ee_frame,
            vel_limit=vel_limit
        )
    )
    builder.Connect(
        iiwa_pos_passblock.get_output_port(),
        vel_diffik_block.GetInputPort("iiwa_position")
    )

    builder.ExportOutput(
        vel_diffik_block.get_output_port(), 
        "iiwa.position"
    )
    
    builder.Connect(
        external_sys_block.GetOutputPort("desired_velocity"),
        vel_diffik_block.GetInputPort("V_WE")
    )
    

    builder.ExportOutput(
        external_sys_block.GetOutputPort("feedforward_torque"),
        "feedforward_torque"
    )
    
    diagram = builder.Build()
    diagram.set_name("CartesianVelocityControlDiagram")
    return diagram