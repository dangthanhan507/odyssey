import lcm
from odyssey.msgs.lcm_msgs import iiwa_commands_t


class KukaLoopLCM:
    def __init__(self):
        self.lcm = lcm.LCM()
        self.sub = self.lcm.subscribe('ODYSSEY_IIWA_TARGETS', lambda channel, data: self.msg_handler(channel, data))
        self.sub.set_queue_capacity(1)
        self.desired_quat = None
        self.desired_pos  = None
        self.feedforward_torque = None
        self.desired_cartesian_vel = None
        self.desired_joints = None
    
    def msg_handler(self, channel, data):
        fri_msg = iiwa_commands_t.decode(data)
        
        self.desired_quat = fri_msg.desired_quat
        self.desired_pos  = fri_msg.desired_pos
        self.desired_cartesian_vel = fri_msg.desired_cartesian_vel
        self.desired_joints = fri_msg.desired_joints
        self.desired_torque = fri_msg.desired_torque
        
    def get_desired_quat(self):
        return self.desired_quat
    
    def get_desired_pos(self):
        return self.desired_pos

    def get_feedforward_torque(self):
        return self.feedforward_torque
    
    def get_desired_cartesian_vel(self):
        return self.desired_cartesian_vel

    def get_desired_joints(self):
        return self.desired_joints
    
    def handle(self):
        self.lcm.handle_timeout(10)