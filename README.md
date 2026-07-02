# odyssey

Since people like adding weird library names, here's one of them.

Odyssey (noun) - a long wandering or voyage usually marked by many changes of fortune.

This library should be my one-stop shop for running code on the kuka hardware. 

## How to use

Checkout examples for ways to use the odyssey library in simulation and on hardware.

### Running the examples

All examples live in `odyssey/examples/` and are run **from the repo package
directory** so their relative paths (`configs/...`, `urdf/...`) resolve:

```bash
cd odyssey            # i.e. /path/to/odyssey/odyssey (the package dir)
uv run python examples/spacemouse_teleop.py
```

Each teleop example follows the same two-process pattern: the `__main__` block
spawns the `RobotLoopDiagram` (the robot loop / LCM server) in a background
`multiprocessing.Process`, then constructs a workstation (the LCM client) and
loops `workstation.handle()`. Ctrl-C exits and cleans up the robot process.

| Example | Device | Control mode | Notes |
|---|---|---|---|
| `joint_sliders.py` | viser GUI sliders | JOINT | simulated by default |
| `cartesian_drag.py` | viser transform gizmo | DIFFIK_POSE | simulated by default |
| `spacemouse_teleop.py` | 3Dconnexion SpaceMouse | CARTESIAN_VELOCITY | needs a SpaceMouse |
| `oculus_teleop.py` | Oculus/Quest controller | DIFFIK_POSE | needs `oculus_reader` + Quest (see below) |
| `gamepad_teleop.py` | gamepad + Schunk WSG gripper | CARTESIAN_VELOCITY | needs a gamepad + schunk driver (see below) |

Device-specific teleop examples (spacemouse / oculus / gamepad) import their
device library lazily, so the modules import and their mapping math can be
validated even when the hardware is absent. Each has a matching hardware-free
`check_*.py` script that validates the math with `uv run python
examples/check_<name>_teleop.py`.

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

## Oculus (Quest) teleop

`examples/oculus_teleop.py` teleops the end-effector with an Oculus/Quest
controller (DIFFIK_POSE control mode). Hold the right controller's grip button
to enable motion; the relative controller motion is applied to the arm's pose
and re-anchors on release. This robot has no gripper, so the right trigger is
read but unused.

This example needs the `oculus_reader` package, which is **not** a declared
dependency because it requires a physical Quest headset connected over ADB. The
import is lazy, so the rest of the module (and the pose math) works without it.
Install it separately when running on real hardware:

```bash
pip install git+https://github.com/rail-berkeley/oculus_reader.git
# also requires adb (android-tools-adb) and a connected Quest
```

You can validate the teleop pose math without any hardware or robot process:

```bash
uv run python odyssey/examples/check_oculus_teleop.py
```

## Gamepad teleop (with Schunk WSG gripper)

`examples/gamepad_teleop.py` teleops the end-effector with a gamepad
(CARTESIAN_VELOCITY control mode) and drives a Schunk WSG gripper with the right
trigger:

- **left stick** → end-effector translation in x / y
- **right stick** → z translation (vertical) and a rotation term (horizontal)
- **right trigger** → proportional gripper: released = open, fully squeezed =
  closed, partial = partway. The trigger value maps linearly between
  `gripper_open_mm` and `gripper_closed_mm`.

`pygame` (used to read the gamepad) is a declared dependency, but a physical
gamepad must be connected at runtime. The `pygame` import is lazy, so the module
and its mapping helpers work without a joystick present.

**Gripper wiring.** The gripper is commanded over LCM using Drake's
`lcmt_schunk_wsg_command` on channel `SCHUNK_WSG_COMMAND`, published on odyssey's
LCM bus (`udpm://239.241.129.92:20185`, the same bus as the iiwa). The separate
[`drake-schunk-driver`](../drake-schunk-driver) daemon must be running and
listening on that same bus/channel to actually move the gripper — run it with
odyssey's LCM URL (e.g. export `LCM_DEFAULT_URL=udpm://239.241.129.92:20185`)
so both ends share a bus. Axis indices in `GamepadTeleopWorkstation.AXIS_*`
target an Xbox-style layout via pygame; remap them for your controller.

Run it (arm on real hardware; set `use_simulated_hardware=True` in the
`__main__` block to run the arm in sim):

```bash
cd odyssey
uv run python examples/gamepad_teleop.py
```

Validate the stick→velocity and trigger→gripper mapping math without a gamepad,
robot, or gripper:

```bash
uv run python odyssey/examples/check_gamepad_teleop.py
```