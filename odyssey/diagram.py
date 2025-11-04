from pydrake.all import (
    LeafSystem,
    RigidTransform,
    Value,
    DiagramBuilder,
    MultibodyPlant,
    Multiplexer,
    Simulator,
    DifferentialInverseKinematicsIntegrator,
    DifferentialInverseKinematicsParameters,
    Context,
    AddMultibodyPlant,
    Parser,
    ValueProducer,
    AbstractValue,
    Quaternion,
    ConstantValueSource
)
from manipulation.station import (
    Scenario,
    ConfigureParser,
    ProcessModelDirectives,
    ModelDirectives,
    load_scenario, 
    MakeHardwareStation
)
import typing
import numpy as np
import lcm
from odyssey.msgs.lcm_msgs import lcmt_iiwa_status, iiwa_commands_t
'''
    GOAL: Create a Hardware implementation for a user that does not know drake at all to minimally use.
'''

def MakeFakeStation(
    scenario: Scenario,
    *,
    package_xmls: typing.List[str] = [],    
):
    builder = DiagramBuilder()

    # Create the multibody plant and scene graph.
    sim_plant, scene_graph = AddMultibodyPlant(
        config=scenario.plant_config, builder=builder
    )
    

    parser = Parser(sim_plant)
    for p in package_xmls:
        parser.package_map().AddPackageXml(p)
    ConfigureParser(parser)

    # Add model directives.
    added_models = ProcessModelDirectives(
        directives=ModelDirectives(directives=scenario.directives),
        parser=parser,
    )

    # Now the plant is complete.
    sim_plant.Finalize()
    
    diagram = builder.Build()
    return diagram



def DiffIKParams(plant, xyz_speed_limit = 0.03, time_step=1e-4):
    params = DifferentialInverseKinematicsParameters(
        plant.num_positions(), plant.num_velocities()
    )
    q0 = plant.GetPositions(plant.CreateDefaultContext())
    params.set_nominal_joint_position(q0)
    # params.set_joint_acceleration_limits()
    params.set_end_effector_angular_speed_limit(20.0 * np.pi / 180) # 20 deg/s
    params.set_end_effector_translational_velocity_limits(
        [-xyz_speed_limit, -xyz_speed_limit, -xyz_speed_limit], [xyz_speed_limit, xyz_speed_limit, xyz_speed_limit]
    )
    
    iiwa14_velocity_limits = np.array([1.4, 1.4, 1.7, 1.3, 2.2, 2.3, 2.3])
    params.set_joint_velocity_limits(
        (-iiwa14_velocity_limits, iiwa14_velocity_limits)
    )
    params.set_joint_centering_gain(0 * np.eye(7)) # no nullspace crap
    params.set_time_step(time_step)
    return params

def AddIiwaDifferentialIK(builder, plant, frame=None, xyz_speed_limit = 0.03, time_step=1e-4):
    params = DifferentialInverseKinematicsParameters(
        plant.num_positions(), plant.num_velocities()
    )
    q0 = plant.GetPositions(plant.CreateDefaultContext())
    params.set_nominal_joint_position(q0)
    # params.set_joint_acceleration_limits()
    params.set_end_effector_angular_speed_limit(20.0 * np.pi / 180) # 20 deg/s
    params.set_end_effector_translational_velocity_limits(
        [-xyz_speed_limit, -xyz_speed_limit, -xyz_speed_limit], [xyz_speed_limit, xyz_speed_limit, xyz_speed_limit]
    )
    
    iiwa14_velocity_limits = np.array([1.4, 1.4, 1.7, 1.3, 2.2, 2.3, 2.3])
    params.set_joint_velocity_limits(
        (-iiwa14_velocity_limits, iiwa14_velocity_limits)
    )
    params.set_joint_centering_gain(0 * np.eye(7)) # no nullspace crap
    
    if frame is None:
        frame = plant.GetFrameByName("body")
    differential_ik = builder.AddSystem(
        DifferentialInverseKinematicsIntegrator(
            plant,
            frame,
            time_step,
            params,
            log_only_when_result_state_changes=True,
        )
    )
    return differential_ik

class RobotLoopDiagram:
    '''
        A diagram that tries to abstract the "drake"-specific elements away for the user to easily use.
    '''
    
    # keep these class inner so that it doesn't get used outside
    class KukaLoopLCM:
        def __init__(self):
            self.lcm = lcm.LCM()
            self.sub = self.lcm.subscribe('ODYSSEY_IIWA_TARGETS', lambda channel, data: self.msg_handler(channel, data))
            self.desired_quat = None
            self.desired_pos  = None
            self.feedforward_torque = None
        def msg_handler(self, channel, data):
            fri_msg = iiwa_commands_t.decode(data)
            
            self.desired_quat = fri_msg.desired_quat
            self.desired_pos  = fri_msg.desired_pos
            self.feedforward_torque = fri_msg.feedforward_torque
            
        def get_desired_quat(self):
            return self.desired_quat
        
        def get_desired_pos(self):
            return self.desired_pos
        
        def handle(self):
            self.lcm.handle_timeout(10)
    
    class ExternalSystem(LeafSystem):
        def __init__(self, plant: MultibodyPlant, ee_frame = "iiwa_link_7", use_impedance: bool = False, simulated: bool = False):
            LeafSystem.__init__(self)
            
            self.simulated = simulated
            self._plant = plant
            self._plant_context = plant.CreateDefaultContext()
            self.ee_frame = ee_frame
            
            self.desired_quat = None
            self.desired_pos = None
            self.feedforward_torque = None
            
            self.lcm = RobotLoopDiagram.KukaLoopLCM()
            
            # Inputs
            self.DeclareVectorInputPort("iiwa_position", 7)
            self.DeclareVectorInputPort("iiwa_velocity", 7)
            self.DeclareVectorInputPort("iiwa_torque_external", 7)
            self.DeclareVectorInputPort("iiwa_position_commanded", 7)
            
            self._calc_external = self.DeclareCacheEntry(
                description="add_sub",
                value_producer=ValueProducer(
                    allocate=lambda: AbstractValue.Make(tuple()),
                    calc=self.CalcExternalFn)
                )
            
            # Outputs
            self.DeclareAbstractOutputPort(
                "desired_pose",
                lambda: Value(RigidTransform()),
                self.calc_desired_pose,
                prerequisites_of_calc={self._calc_external.ticket()}
            )
            if use_impedance:
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

            self._plant.SetPositions(self._plant_context, position)
            ee_pose = self._plant.GetFrameByName(self.ee_frame).CalcPoseInWorld(self._plant_context)
            
            self.desired_quat = self.lcm.get_desired_quat() # [x,y,z,w]
            self.desired_pos = self.lcm.get_desired_pos()
            
            
            self.desired_pose = RigidTransform(
                quaternion=Quaternion(self.desired_quat[3], self.desired_quat[0], self.desired_quat[1], self.desired_quat[2]),
                p=self.desired_pos
            ) if not (self.desired_quat is None or self.desired_pos is None) else None
            self.feedforward_torque = None
            
            desired_pose = self.desired_pose if not self.desired_pose is None else ee_pose
            feedforward_torque = self.feedforward_torque if not self.feedforward_torque is None else np.zeros(7)
            
            output.set_value((desired_pose, feedforward_torque))
        
        def calc_desired_pose(self, context, output):
            pose, _ = self._calc_external.Eval(context)
            
            output.set_value(pose)
            
        def calc_feedforward_torque(self, context, output):
            _, torque = self._calc_external.Eval(context)
            output.SetFromVector(torque)
    
    def __init__(self, config, use_simulated_hardware: bool = False, use_impedance: bool = False):
        
        self.simulated = use_simulated_hardware
        self.use_impedance = use_impedance
        scenario = load_scenario(filename=config) # load robot setup yaml
        self._station = MakeHardwareStation(scenario, hardware=not use_simulated_hardware)
        
        self._fake_station = MakeFakeStation(scenario)
        self._plant: MultibodyPlant = self._fake_station.GetSubsystemByName("plant")
        self._plant_context = self._plant.CreateDefaultContext()
        
    def setup_diagram(self, diffik_frame="iiwa_link_7"):
        builder = DiagramBuilder()
        station = builder.AddNamedSystem("station", self._station)
        
        external_sys = builder.AddSystem(RobotLoopDiagram.ExternalSystem(
            plant=self._plant,
            ee_frame=diffik_frame,
            use_impedance=self.use_impedance,
            simulated=self.simulated
            )
        )
        
        iiwa_state = builder.AddSystem(Multiplexer([7,7]))
        builder.Connect(
            station.GetOutputPort("iiwa.position_measured"),
            iiwa_state.get_input_port(0)
        )
        builder.Connect(
            station.GetOutputPort("iiwa.velocity_estimated"),
            iiwa_state.get_input_port(1)
        )
        
        builder.Connect(
            station.GetOutputPort("iiwa.position_measured"),
            external_sys.GetInputPort("iiwa_position")
        )
        builder.Connect(
            station.GetOutputPort("iiwa.velocity_estimated"),
            external_sys.GetInputPort("iiwa_velocity")
        )
        builder.Connect(
            station.GetOutputPort("iiwa.torque_external"),
            external_sys.GetInputPort("iiwa_torque_external")
        )
        builder.Connect(
            station.GetOutputPort("iiwa.position_commanded"),
            external_sys.GetInputPort("iiwa_position_commanded")
        )
        
        
        diffik_block = AddIiwaDifferentialIK(builder, self._plant, self._plant.GetFrameByName(diffik_frame))        
        builder.Connect(
            diffik_block.get_output_port(), 
            station.GetInputPort("iiwa.position")
        )
        builder.Connect(
            iiwa_state.get_output_port(),
            diffik_block.GetInputPort("robot_state"),
        )
        use_state = builder.AddSystem(ConstantValueSource(Value(True)))
        builder.Connect(
            use_state.get_output_port(),
            diffik_block.GetInputPort("use_robot_state"),
        )
        
        builder.Connect(
            external_sys.GetOutputPort("desired_pose"), diffik_block.GetInputPort("X_AE_desired")
        )
        
        if self.use_impedance:
            builder.Connect(
                external_sys.GetOutputPort("feedforward_torque"),
                station.GetInputPort("iiwa.torque")
            )
        
        diagram = builder.Build()
        return diagram
    
    def run_system(self, diagram, duration=np.inf):
        simulator = Simulator(diagram)
        simulator.set_target_realtime_rate(1.0)
        simulator.Initialize()
        if self.simulated:
            # set joint positions to initial positions
            initial_q = np.array([0.0, np.pi/6, 0.0, -80*np.pi/180, 0.0, np.pi/6, 0.0])
            simulator_context = simulator.get_mutable_context()
            plant = self._station.GetSubsystemByName("plant")
            plant_context = plant.GetMyMutableContextFromRoot(simulator_context)
            plant.SetPositions(plant_context, initial_q)
            
        simulator.AdvanceTo(duration)
        
        
if __name__ == '__main__':
    config = "configs/kuka_default.yaml"
    loop_diagram = RobotLoopDiagram(config, use_simulated_hardware=True, use_impedance=True)
    diagram = loop_diagram.setup_diagram()
    loop_diagram.run_system(diagram)