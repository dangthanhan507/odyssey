from pydrake.all import (
    DoDifferentialInverseKinematics,
    DifferentialInverseKinematicsStatus,
    DifferentialInverseKinematicsIntegrator,
    DifferentialInverseKinematicsParameters,
    LeafSystem,
    MultibodyPlant,
)
import numpy as np

def AddIiwaDifferentialIK(builder, plant, frame=None, xyz_speed_limit = 0.03, angular_speed_limit = 20 * np.pi / 180, time_step=1e-4):
    params = DifferentialInverseKinematicsParameters(
        plant.num_positions(), plant.num_velocities()
    )
    q0 = plant.GetPositions(plant.CreateDefaultContext())
    params.set_nominal_joint_position(q0)
    # params.set_joint_acceleration_limits()
    params.set_end_effector_angular_speed_limit(angular_speed_limit) # 20 deg/s
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

class VelocityDiffIK(LeafSystem):
    def __init__(self, plant: MultibodyPlant, frame_name = "iiwa_link_7", vel_limit = 0.03):
        LeafSystem.__init__(self)
        
        self._plant = plant
        self._plant_context = plant.CreateDefaultContext()
        self._frame_E = plant.GetFrameByName(frame_name)
        
        
        self.velocity_limit = vel_limit
        self.DeclareVectorInputPort("iiwa_position", plant.num_positions())
        self.DeclareVectorInputPort("V_WE", 6)
        self.DeclareInitializationDiscreteUpdateEvent(self.Initialize)
        
        self._time_step = plant.time_step()
        self.DeclareDiscreteState(plant.num_positions()) # one discrete state for tracking target joint positions
        self.DeclarePeriodicDiscreteUpdateEvent(self._time_step, 0, self.CalcJointPos)
        
        self._diff_ik_params = DiffIKParams(plant, xyz_speed_limit = vel_limit, time_step=self._time_step)
        
        self.DeclareVectorOutputPort(
            "iiwa.position", plant.num_positions(), self.OutputIiwaPosition
        )
    
    def Initialize(self, context, discrete_state):
        print(self.GetInputPort("iiwa_position").Eval(context))
        discrete_state.set_value(0,self.GetInputPort("iiwa_position").Eval(context))
    
    def CalcJointPos(self, context, discrete_state):
        V_WE = self.GetInputPort("V_WE").Eval(context)
        
        q = np.copy(context.get_discrete_state(0).get_value())
        self._plant.SetPositions(self._plant_context, q)
        result = DoDifferentialInverseKinematics(
            self._plant,
            self._plant_context,
            V_WE,
            self._frame_E,
            self._diff_ik_params
        )    
        q_next = q + result.joint_velocities * self._time_step if result.status == DifferentialInverseKinematicsStatus.kSolutionFound else q
        discrete_state.set_value(0, q_next)
        
    def OutputIiwaPosition(self, context, output):
        q_next = context.get_discrete_state(0).get_value()
        output.SetFromVector(q_next)