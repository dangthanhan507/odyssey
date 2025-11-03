from odyssey.msgs.lcm_msgs import iiwa_commands_t, lcmt_iiwa_status
import lcm


# this allows access to kuka iiwa status messages over lcm
# we can use this outside of drake simulation to prevent overhead in control loop
class KukaLCM:
    def __init__(self,):
        self.lcm = lcm.LCM()
        self.sub = self.lcm.subscribe('IIWA_STATUS', lambda channel, data: self.msg_handler(channel, data))
    
    def msg_handler(self, channel, data):
        fri_msg = lcmt_iiwa_status.decode(data)
        
        self.joint_commanded  = fri_msg.joint_position_commanded
        self.joint_measured   = fri_msg.joint_position_measured
        self.torque_commanded = fri_msg.joint_torque_commanded
        self.torque_measured  = fri_msg.joint_torque_measured
        self.torque_external  = fri_msg.joint_torque_external
        self.joint_velocity   = fri_msg.joint_velocity_estimated
        
        print("nice")
    
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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--publish_channel_name', type=str)
    args = parser.parse_args()
    
    kuka_lcm = KukaLCM()
    while True:
        kuka_lcm.handle()
        
        fake_pos = [0.0]*7
        fake_torque = [0.0]*7
        msg = lcmt_iiwa_status()
        msg.joint_position_commanded = fake_pos
        msg.joint_position_measured = fake_pos
        msg.joint_torque_commanded = fake_torque
        msg.joint_torque_measured = fake_torque
        msg.joint_torque_external = fake_torque
        msg.joint_velocity_estimated = fake_pos
        # kuka_lcm.lcm.publish(args.publish_channel_name, lcmt_iiwa_status.encode(msg))
        
        msg = iiwa_commands_t()
        msg.desired_pos = [0.0, 0.0, 0.0]
        msg.desired_quat = [0.0, 0.0, 0.0, 1.0]
        kuka_lcm.lcm.publish('ODYSSEY_IIWA_TARGETS', iiwa_commands_t.encode(msg))