# odyssey

Since people like adding weird library names, here's one of them.

Odyssey (noun) - a long wandering or voyage usually marked by many changes of fortune.

This library should be my one-stop shop for running code on the kuka hardware. 

## How to use

Checkout examples for ways to use the odyssey library in simulation and on hardware.

First we want to run our `RobotLoopDiagram` which will create the robot system and controller. Note this will be ran separately from the workstation code. You can look at examples for details on how to run both in one script.

```python
import numpy as np
from odyssey.robot import RobotLoopDiagram, ControlMode
robotloop = RobotLoopDiagram(config='configs/kuka_default.yaml', use_simulated_hardware=True, use_impedance=True, control_mode=ControlMode.JOINT)
robotloopdiagram = robotloop.setup_diagram()
robotloop.run_system(robotloopdiagram, initial_q = np.zeros(7))
```

For the workstation code, we can create a custom workstation that adds sliders for each joint in the robot. This will allow us to control the robot's joints through a GUI. This is where the user does most of their interaction with the robot.


Note that `self.send_joint_command` is the method that does the interaction with the robot system created in the `RobotLoopDiagram`.

```python
import numpy as np
from odyssey.workstation import OdysseyBaseWorkstation

class JointSliderWorkstation(OdysseyBaseWorkstation):
    def __init__(self, urdf_path, robot_description_path):
        OdysseyBaseWorkstation.__init__(self,
                                         robot_urdf_path=urdf_path,
                                         robot_description_path=robot_description_path,
                                         load_meshes=True,
                                         load_collision_meshes=False,
                                         host='127.0.0.1',
                                         port=8080
        )
        
        # add in sliders for each joint
        with self.server.gui.add_folder('Joint Sliders'):
            slider_handles = []
            for joint_name, (lower, upper) in self.viser_urdf.get_actuated_joint_limits().items():
                lower = lower if lower is not None else -np.pi
                upper = upper if upper is not None else np.pi
                
                initial_pos = 0.0 if lower < -0.1 and upper > 0.1 else (lower + upper) / 2.0
                slider = self.server.gui.add_slider(
                    label=joint_name,
                    min=lower,
                    max=upper,
                    step=1e-3,
                    initial_value=initial_pos
                )
                
                slider.on_update(
                    lambda _: self.send_joint_command([slider.value for slider in slider_handles]) 
                )
                slider_handles.append(slider)

workstation = JointSliderWorkstation(
    urdf_path='urdf/med.urdf',
    robot_description_path='urdf/lbr_description/'
)

while True:
    workstation.handle()
```

We have 3 control modes for the kuka robot:
- JOINT: send joint position commands
- DIFFIK_POSE: send end-effector pose (can be any pose depends on what frame you supply)
- CARTESIAN_VELOCITY: send end-effector twist commands (can be any frame depends on what frame you supply)

We can change the control mode by changing the `control_mode` argument in the `RobotLoopDiagram`.

For `OdysseyBaseWorkstation`, we use different commands for different control modes:
- JOINT: `send_joint_command`0
- DIFFIK_POSE: `send_pose_command`
- CARTESIAN_VELOCITY: `send_cartesian_velocity_command`


## Running code

## Docker

remember to pull docker image for ubuntu 24.04:

```bash
docker pull ubuntu:24.04
```

## LCM Messages

```bash
lcm-gen -p example_t.lcm
```

Create custom lcm messages

```bash
lcm-gen -j ./path_to_lcm_files/*.lcm
javac -cp /usr/share/java/lcm.jar lcm_msgs/*.java
jar cf my_types.jar lcm_msgs/*.class
```

**NOTE**: this is assuming we are using Docker which installed lcm-dev into `/usr/share/java/lcm.jar`.


## Pyspacemouse 

Pyspacemouse instructions

```bash
sudo apt-get install libhidapi-dev
sudo echo 'KERNEL=="hidraw*", SUBSYSTEM=="hidraw", MODE="0664", GROUP="plugdev"' > /etc/udev/rules.d/99-hidraw-permissions.rules
sudo usermod -aG plugdev $USER
newgrp plugdev
```