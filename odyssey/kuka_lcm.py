from odyssey.msgs.lcm_msgs import iiwa_commands_t, lcmt_iiwa_status
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
        
        
if __name__ == "__main__":
    kuka_lcm = KukaLCM()
    while True:
        kuka_lcm.handle()