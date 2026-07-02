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


## Geometric fabrics collision avoidance

`odyssey.fabrics` is a CPU/numpy port of the geometric-fabric collision
avoidance in NVIDIA's [FABRICS](https://github.com/NVlabs/FABRICS)
(`fabrics_sim`). The upstream library is GPU/Warp-only and wired for the
Kuka-Allegro; here the fabric *math* is reimplemented for the iiwa using Drake
for forward kinematics / Jacobians and analytic box SDFs for obstacles. No
torch, warp, or Isaac Sim required.

Each control step it integrates `M(q, q̇) q̈ + f(q, q̇) = 0` forward, combining:
- an **end-effector attractor** (forcing) pulling the arm toward a target,
- **body-sphere repulsion** (forcing + geometric) pushing the arm's collision
  spheres away from boxes via a rank-1, `1/d²`-scaled barrier metric,
- **joint-limit repulsion** and cspace damping.

Behavior is tuned via `configs/fabric_params.yaml` (mirrors the upstream
`kuka_allegro_pose_params.yaml`).

```python
import numpy as np
from odyssey.fabrics import IiwaBoxFabric, ObstacleSet

obstacles = ObstacleSet()
obstacles.add_box(center=[0.55, 0.30, 0.60], size=[0.30, 0.20, 0.50])

fabric = IiwaBoxFabric(obstacles, use_posture_attractor=True)
fabric.set_posture_target(np.zeros(7))
fabric.set_ee_target([0.55, 0.0, 0.35])

q, qd = np.array([0.0, 0.4, 0.0, -1.0, 0.0, 0.6, 0.0]), np.zeros(7)
for _ in range(600):
    q, qd, qdd = fabric.step(q, qd, 1.0 / 60.0)
```

Run the "kuka surrounded by boxes" example (viser + LCM, cycles through targets
while avoiding the boxes):

```bash
python -m odyssey.examples.fabric_box_avoidance            # viser + LCM
python -m odyssey.examples.fabric_box_avoidance --headless # no viser, no LCM
```

### Editing workspace constraints in viser

Use the interactive editor to place constraint boxes around the robot with
transform gizmos (drag to translate/rotate, number fields to resize), then
export them to JSON:

```bash
python -m odyssey.examples.constraint_editor --output my_constraints.json
python -m odyssey.examples.constraint_editor --load my_constraints.json   # re-edit
```

Open the viser URL, add/move/resize/delete boxes, and click **save JSON**. The
boxes are defined in the robot's world frame, so the same file is valid in
simulation and on the real hardware. Feed it back into the fabric:

```bash
python -m odyssey.examples.fabric_box_avoidance --constraints my_constraints.json
```

Or load it directly in code for a hardware control loop:

```python
from odyssey.fabrics import ObstacleSet, IiwaBoxFabric
obstacles = ObstacleSet.load_json("my_constraints.json")
fabric = IiwaBoxFabric(obstacles)
# ... fabric.step(q, qd, dt) each control step, send q over LCM ...
```

The JSON format is a list of boxes with `center`, `size` (full extents), and a
`wxyz` orientation quaternion:

```json
{
  "version": 1, "frame": "world",
  "boxes": [{"name": "front_wall", "center": [0.55, 0.3, 0.6],
             "size": [0.3, 0.2, 0.5], "wxyz": [1.0, 0.0, 0.0, 0.0]}]
}
```

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