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
    BasicVector,
    AbstractValue
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

class Option2(LeafSystem):
    """Provides two output ports from a single CalcSumDiff function.
    Option 2 uses caching so that CalcAdd and CalcSub reuse the tuple of values
    computed by the CalcAddSub helper function.
    """

    def __init__(self):
        LeafSystem.__init__(self)
        self._pair = self.DeclareVectorInputPort("pair", BasicVector(2))
        self._add_sub = self.DeclareCacheEntry(
            description="add_sub",
            value_producer=ValueProducer(
                allocate=lambda: AbstractValue.Make(tuple()),
                calc=self.CalcAddSub),
            prerequisites_of_calc={self._pair.ticket()})
        self.DeclareVectorOutputPort(
            "add", BasicVector(1), self.CalcAdd,
            prerequisites_of_calc={self._add_sub.ticket()})
        self.DeclareVectorOutputPort(
            "sub", BasicVector(1), self.CalcSub,
            prerequisites_of_calc={self._add_sub.ticket()})

    def CalcAddSub(self, context, output):
        u = self.GetInputPort("pair").Eval(context)
        add = np.array([u[0] + u[1]])
        sub = np.array([u[0] - u[1]])
        output.set_value((add, sub))

    def CalcAdd(self, context, output):
        y0, _ = self._add_sub.Eval(context)
        output.set_value(y0)

    def CalcSub(self, context, output):
        _, y1 = self._add_sub.Eval(context)
        output.set_value(y1)

class RobotDiagram:
    
    # keep this class inner so that it doesn't get used outside
    class ExternalSystem(LeafSystem):
        def __init__(self, plant: MultibodyPlant, ee_frame = "iiwa_link_7", use_impedance: bool = False, callback_fn = lambda **kwargs: (None, None)):
            LeafSystem.__init__()
            
            self._plant = plant
            self._plant_context = plant.CreateDefaultContext()
            self.ee_frame = ee_frame
            
            # Inputs
            position_input_port = self.DeclareInputPort("iiwa_position", 7)
            velocity_input_port = self.DeclareInputPort("iiwa_velocity", 7)
            torque_ext_input_port = self.DeclareInputPort("iiwa_torque_external", 7)
            torque_com_input_port = self.DeclareInputPort("iiwa_torque_commanded", 7)
            
            
            self._callback_fn = callback_fn
            self._calc_external = self.DeclareCacheEntry(
                description="add_sub",
                value_producer=ValueProducer(
                    allocate=lambda: AbstractValue.Make(tuple()),
                    calc=self.CalcExternalFn),
                prerequisites_of_calc={position_input_port.ticket(),
                                       velocity_input_port.ticket(),
                                       torque_ext_input_port.ticket(),
                                       torque_com_input_port.ticket()})
            
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
            position = self.GetInputPort("iiwa_position").Eval(context)
            velocity = self.GetInputPort("iiwa_velocity").Eval(context)
            torque_external = self.GetInputPort("iiwa_torque_external").Eval(context)
            torque_commanded = self.GetInputPort("iiwa_torque_commanded").Eval(context)

            self._plant.SetPositions(self._plant_context, position)
            ee_pose = self._plant.GetFrameByName(self.ee_frame).CalcPoseInWorld(self._plant_context)
            
            desired_pose, feedforward_torque = self._callback_fn(
                iiwa_position=position,
                iiwa_velocity=velocity,
                iiwa_torque_external=torque_external,
                iiwa_torque_commanded=torque_commanded
            )
            

            if desired_pose is None:
                desired_pose = ee_pose
            if feedforward_torque is None:
                feedforward_torque = np.zeros(7)
            
            output.set_value((desired_pose, feedforward_torque))
        
        def calc_desired_pose(self, context, output):
            pose, _ = self._calc_external.Eval(context)
            
            output.set_value(pose)
            
        def calc_feedforward_torque(self, context, output):
            _, torque = self._calc_external.Eval(context)
            output.SetFromVector(torque)
    
    def __init__(self, config, use_simulated_hardware: bool = False, use_impedance: bool = False):
        
        self.use_impedance = use_impedance
        scenario = load_scenario(config) # load robot setup yaml
        self._station = MakeHardwareStation(scenario, hardware=not use_simulated_hardware)
        
        self._fake_station = MakeFakeStation(scenario)
        self._plant: MultibodyPlant = self._station.GetSubsystemByName("plant")
        self._plant_context = self._plant.CreateDefaultContext()
        
    def get_position(self, context: Context):
        self._plant.SetPositions(self._plant_context, )
        
    def setup_diagram(self, diffik_frame="iiwa_link_7", callback_fn = lambda **kwargs: (None, None)):
        builder = DiagramBuilder()
        station = builder.AddNamedSystem("station", self._station)
        
        external_sys = builder.AddSystem(RobotDiagram.ExternalSystem(
            plant=self._plant,
            ee_frame=diffik_frame,
            use_impedance=self.use_impedance,
            callback_fn=callback_fn)
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
            station.GetOutputPort("iiwa.torque_commanded"),
            external_sys.GetInputPort("iiwa_torque_commanded")
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
        
        builder.Connect(
            external_sys.GetOutputPort("desired_pose"), diffik_block.GetInputPort("X_WE_desired")
        )
        
        if self.use_impedance:
            builder.Connect(
                external_sys.GetOutputPort("feedforward_torque"),
                station.GetInputPort("iiwa.feedforward_torque")
            )
        
        diagram = builder.Build()
        return diagram
    
    def run_system(self, diagram, duration=np.inf):
        simulator = Simulator(diagram)
        simulator.set_target_realtime_rate(1.0)
        simulator.Initialize()
        simulator.AdvanceTo(duration)