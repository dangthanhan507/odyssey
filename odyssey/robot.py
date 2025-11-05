from pydrake.all import (
    DiagramBuilder,
    MultibodyPlant,
    Simulator,
    AddMultibodyPlant,
    Parser,
    LeafSystem,
    PassThrough
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
from odyssey.diagrams import ControlMode, joint_control_diagram, diffik_pose_diagram, cartesian_velocity_diagram

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
            control_diagram = builder.AddSystem(
                joint_control_diagram(
                    self._plant,
                    simulated=self.simulated,
                )
            )
        
        elif self.control_mode == ControlMode.DIFFIK_POSE:
            control_diagram = builder.AddSystem(
                diffik_pose_diagram(
                    self._plant,
                    simulated=self.simulated,
                    ee_frame=diffik_frame,
                )
            )
        
        elif self.control_mode == ControlMode.CARTESIAN_VELOCITY:
            control_diagram = builder.AddSystem(
                cartesian_velocity_diagram(
                    self._plant,
                    ee_frame=diffik_frame,
                    simulated=self.simulated,
                    vel_limit=0.03
                )
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
            station.GetOutputPort("iiwa.torque_external"),
            control_diagram.GetInputPort("iiwa.torque_external")
        )
        
        if not self.simulated:
            # NOTE: this causes algebraic loop in simulation since iiwa.position is passed back as iiwa.position_commanded
            builder.Connect(
                station.GetOutputPort("iiwa.position_commanded"),
                control_diagram.GetInputPort("iiwa.position_commanded")
            )
        else:
            class HackCommand(LeafSystem):
                def __init__(self, time_step=1e-4):
                    LeafSystem.__init__(self)
                    
                    self.DeclareVectorInputPort("iiwa.position_measured", 7)
                    self.DeclareVectorInputPort("iiwa.position_commanded", 7)
                    
                    discrete_state_index = self.DeclareDiscreteState(7) # dummy state for position commanded
                    self.DeclareInitializationDiscreteUpdateEvent(self.Initialize)
                    self.DeclarePeriodicDiscreteUpdateEvent(time_step, 0, self.UpdateCommanded)
                    
                    self.DeclareVectorOutputPort("iiwa.position_commanded_hack", 7, self.OutputCommanded, prerequisites_of_calc={self.discrete_state_ticket(discrete_state_index)})
                
                def Initialize(self, context, discrete_state):
                    discrete_state.set_value(0,self.GetInputPort("iiwa.position_measured").Eval(context))
                
                def UpdateCommanded(self, context, discrete_state):
                    discrete_state.set_value(0, self.GetInputPort("iiwa.position_commanded").Eval(context))
                
                def OutputCommanded(self, context, output):
                    commanded = context.get_discrete_state(0).get_value()
                    print("Hack commanded positions: {}".format(commanded))
                    output.SetFromVector(commanded)
                    
            hack_command = builder.AddSystem(HackCommand(self._plant.time_step()))
            builder.Connect(
                station.GetOutputPort("iiwa.position_measured"),
                hack_command.GetInputPort("iiwa.position_measured")
            )
            builder.Connect(
                station.GetOutputPort("iiwa.position_commanded"),
                hack_command.GetInputPort("iiwa.position_commanded")
            )
            builder.Connect(
                hack_command.GetOutputPort("iiwa.position_commanded_hack"),
                control_diagram.GetInputPort("iiwa.position_commanded")
            )

        class PrintDebug(LeafSystem):
            def __init__(self, plant, frame_E = 'iiwa_link_7'):
                LeafSystem.__init__(self)
                self._plant = plant
                self._plant_context = plant.CreateDefaultContext()
                self.frame_E = frame_E
                
                self.DeclareVectorInputPort("iiwa.position_commanded", 7)
                self.DeclarePeriodicPublishEvent(0.01, 0, self.DoPublish)
            def DoPublish(self, context):
                pos_cmd = self.GetInputPort("iiwa.position_commanded").Eval(context)
                self._plant.SetPositions(self._plant_context, pos_cmd)
                frame = self._plant.GetFrameByName(self.frame_E)
                X_WE = frame.CalcPoseInWorld(self._plant_context)
                print("Commanded positions: {}".format(pos_cmd))
                print("End-effector position: {}".format(X_WE.translation()))
        debug = builder.AddSystem(PrintDebug(self._plant, frame_E=diffik_frame))
        passthrough_block = builder.AddSystem(PassThrough(7))
        builder.Connect(
            station.GetOutputPort("iiwa.position_commanded"),
            passthrough_block.get_input_port()
        )
        builder.Connect(
            # control_diagram.GetOutputPort("iiwa.position"),
            passthrough_block.get_output_port(),
            debug.GetInputPort("iiwa.position_commanded")
        )
        
        # outputs
        builder.Connect(
            control_diagram.GetOutputPort("iiwa.position"),
            station.GetInputPort("iiwa.position")
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
        
    def get_arm_pose(self, frame_name, joint_positions):
        plant_context = self._plant.CreateDefaultContext()
        self._plant.SetPositions(plant_context, joint_positions)
        frame = self._plant.GetFrameByName(frame_name)
        X_WF = frame.CalcPoseInWorld(plant_context)
        pos = X_WF.translation()
        quat = X_WF.rotation().ToQuaternion().wxyz()
        return quat, pos
        
if __name__ == '__main__':
    config = "configs/kuka_default.yaml"
    loop_diagram = RobotLoopDiagram(config, use_simulated_hardware=True, use_impedance=True)
    diagram = loop_diagram.setup_diagram()
    loop_diagram.run_system(diagram)