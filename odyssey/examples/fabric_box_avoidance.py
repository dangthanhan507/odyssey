"""Example: iiwa surrounded by boxes, reaching a target while avoiding them.

This is the odyssey analogue of ``fabrics_sim``'s
``kuka_allegro_pose_fabric_example.py`` (Kuka surrounded by a boxes world). It
uses the CPU/numpy geometric-fabric port in ``odyssey.fabrics``:

* a set of box obstacles surrounds the arm,
* an end-effector attractor pulls the iiwa toward a target,
* body-sphere repulsion deflects the arm away from the boxes.

The fabric integrates forward each step and (optionally) visualizes the robot,
the obstacles, the collision spheres, and the current target in viser. By
default it also streams the fabric's joint solution to the robot over LCM via
``OdysseyBaseWorkstation`` (send_joint_command), so it drives the same
simulated/real hardware path as the other odyssey examples.

Usage:
    python -m odyssey.examples.fabric_box_avoidance            # viser + LCM
    python -m odyssey.examples.fabric_box_avoidance --headless # no viser, no LCM
"""

import argparse
import os
import time

import numpy as np

from odyssey.fabrics import IiwaBoxFabric, ObstacleSet

# Package directory (.../odyssey/odyssey), so URDF paths resolve no matter where
# the script is launched from.
_PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# A ring of boxes surrounding the arm's workspace. Each entry is
# (name, center, size) with size as full side lengths (meters). Loosely inspired
# by the walls/table in fabrics_sim's kuka_allegro_boxes.yaml.
DEFAULT_BOXES = [
    ("front_wall", (0.55, 0.30, 0.60), (0.30, 0.20, 0.50)),
    ("left_wall", (0.20, 0.55, 0.60), (0.60, 0.15, 0.60)),
    ("right_wall", (0.20, -0.55, 0.60), (0.60, 0.15, 0.60)),
    ("table", (0.55, 0.00, 0.02), (0.60, 1.00, 0.06)),
]

# A few end-effector targets the arm cycles through, all in reachable free space
# but requiring the arm to route around the boxes.
DEFAULT_TARGETS = [
    (0.50, -0.35, 0.55),
    (0.55, 0.00, 0.35),
    (0.15, 0.35, 0.85),
    (0.30, 0.00, 0.80),
]

HOME_Q = np.array([0.0, 0.4, 0.0, -1.0, 0.0, 0.6, 0.0])


def build_obstacles(boxes=DEFAULT_BOXES):
    obs = ObstacleSet()
    for name, center, size in boxes:
        obs.add_box(center=center, size=size, name=name)
    return obs


def _setup_viser(workstation, obstacles, sphere_radii):
    """Draw obstacles and return handles for the collision spheres + target."""
    server = workstation.server
    for i, obs in enumerate(obstacles.obstacles):
        server.scene.add_box(
            f"/obstacles/{obs.name or i}",
            color=(200, 60, 60),
            dimensions=tuple(obs.size),
            position=tuple(obs.center),
            opacity=0.5,
        )
    sphere_handles = []
    for i, r in enumerate(sphere_radii):
        sphere_handles.append(server.scene.add_icosphere(
            f"/collision_spheres/{i}", radius=float(r),
            color=(60, 120, 220), opacity=0.4))
    target_handle = server.scene.add_icosphere(
        "/ee_target", radius=0.04, color=(60, 220, 60))
    return sphere_handles, target_handle


def main():
    parser = argparse.ArgumentParser(description="iiwa box-avoidance fabric example.")
    parser.add_argument("--headless", action="store_true",
                        help="Run without viser and without sending LCM commands.")
    parser.add_argument("--control_rate", type=float, default=60.0)
    parser.add_argument("--seconds_per_target", type=float, default=4.0)
    parser.add_argument("--total_time", type=float, default=60.0)
    parser.add_argument("--no_spheres", action="store_true",
                        help="Do not render collision spheres in viser.")
    args = parser.parse_args()

    dt = 1.0 / args.control_rate

    obstacles = build_obstacles()
    fabric = IiwaBoxFabric(obstacles, use_posture_attractor=True)
    fabric.set_posture_target(HOME_Q)

    # Optional viser + LCM workstation (imported lazily so --headless has no deps
    # on viser/hardware being available).
    workstation = None
    sphere_handles = None
    target_handle = None
    if not args.headless:
        from odyssey.workstation import OdysseyBaseWorkstation
        workstation = OdysseyBaseWorkstation(
            robot_urdf_path=os.path.join(_PKG_DIR, "urdf", "med.urdf"),
            robot_description_path=os.path.join(_PKG_DIR, "urdf", "lbr_description/"),
            load_meshes=True,
            load_collision_meshes=False,
        )
        sphere_handles, target_handle = _setup_viser(
            workstation, obstacles, fabric.kin.sphere_radii)

    q = HOME_Q.copy()
    qd = np.zeros(fabric.num_joints)

    targets = DEFAULT_TARGETS
    steps_per_target = int(args.control_rate * args.seconds_per_target)
    fabric.set_ee_target(targets[0])

    start = time.time()
    num_steps = int(args.control_rate * args.total_time)
    for i in range(num_steps):
        if i % steps_per_target == 0:
            target = targets[(i // steps_per_target) % len(targets)]
            fabric.set_ee_target(target)

        q, qd, qdd = fabric.step(q, qd, dt)

        if workstation is not None:
            workstation.update_robot_joints(q)
            workstation.send_joint_command(q.tolist())
            if not args.no_spheres and sphere_handles is not None:
                sphere_pos = fabric.kin.sphere_positions(q)
                for h, p in zip(sphere_handles, sphere_pos):
                    h.position = tuple(p)
            if target_handle is not None:
                target_handle.position = tuple(fabric.ee_attractor.target)
            time.sleep(dt)  # roughly real-time for visualization

        if i % int(args.control_rate) == 0:
            xe, _ = fabric.kin.ee_taskmap(q)
            print(f"t={i * dt:5.1f}s ee={np.round(xe, 3)} "
                  f"min_dist={fabric.min_signed_distance:+.3f} "
                  f"collision={fabric.collision_status}")

    print(f"Done. {num_steps} steps in {time.time() - start:.1f}s wallclock.")


if __name__ == "__main__":
    main()
