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
from diagrams import ControlMode, joint_control_diagram, diffik_pose_diagram, cartesian_velocity_diagram

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

class RobotLoopDiagram:
    def __init__(self, config, use_simulated_hardware: bool = False, use_impedance: bool = False, control_mode: ControlMode = ControlMode.JOINT):
        
        self.control_mode = control_mode
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
        
        if self.control_mode == ControlMode.JOINT:
            control_diagram = joint_control_diagram(
                builder,
                station,
                use_impedance=self.use_impedance
            )
        elif self.control_mode == ControlMode.DIFFIK_POSE:
            control_diagram = diffik_pose_diagram(
                builder,
                station,
                self._plant,
                self._plant_context,
                diffik_frame,
                use_impedance=self.use_impedance
            )
        elif self.control_mode == ControlMode.CARTESIAN_VELOCITY:
            control_diagram = cartesian_velocity_diagram(
                builder,
                station,
                self._plant,
                self._plant_context,
                diffik_frame,
                use_impedance=self.use_impedance
            )
        else:
            raise ValueError("Unsupported control mode: {}".format(self.control_mode))

        # inputs
        builder.Connect(
            station.GetOutputPort("iiwa.position_measured"),
            control_diagram.GetInputPort("iiwa.position_measured")
        )
        builder.Connect(
            station.GetOutputPort("iiwa.velocity_estimated"),
            control_diagram.GetInputPort("iiwa.velocity_estimated")
        )
        builder.Connect(
            station.GetOutputPort("iiwa.torque_measured"),
            control_diagram.GetInputPort("iiwa.torque_measured")
        )
        builder.Connect(
            station.GetOutputPort("iiwa.torque_external"),
            control_diagram.GetInputPort("iiwa.torque_external")
        )
    
        # outputs
        builder.Connect(
            control_diagram.GetOutputPort("iiwa.position"),
            station.GetInputPort("iiwa.positions")
        )
        builder.Connect(
            control_diagram.GetOutputPort("feedforward_torque"),
            station.GetInputPort("iiwa.torque")
        )
        
        diagram = builder.Build()
        return diagram
    
    def run_system(self, diagram, duration=np.inf, initial_q = np.array([0.0, np.pi/6, 0.0, -80*np.pi/180, 0.0, np.pi/6, 0.0])):
        simulator = Simulator(diagram)
        simulator.set_target_realtime_rate(1.0)
        simulator.Initialize()
        
        if self.simulated:
            # set joint positions to initial positions
            simulator_context = simulator.get_mutable_context()
            plant = self._station.GetSubsystemByName("plant")
            plant_context = plant.GetMyMutableContextFromRoot(simulator_context)
            plant.SetPositions(plant_context, initial_q)
        
        else:
            simulator_context = simulator.get_mutable_context()
            station_context = self._station.GetMyMtuableContextFromRoot(simulator_context)
            self._station.ExecuteInitializationEvents(station_context)
            curr_q = self._station.GetOutputPort("iiwa.position_measured").Eval(station_context)
            if np.max(np.abs(initial_q - curr_q)) > 1e-3:
                raise RuntimeError("Initial joint positions for real robot differ from measured positions! measured: {}, initial_q: {}".format(curr_q, initial_q))
        
        simulator.AdvanceTo(duration)
        
if __name__ == '__main__':
    config = "configs/kuka_default.yaml"
    loop_diagram = RobotLoopDiagram(config, use_simulated_hardware=True, use_impedance=True)
    diagram = loop_diagram.setup_diagram()
    loop_diagram.run_system(diagram)