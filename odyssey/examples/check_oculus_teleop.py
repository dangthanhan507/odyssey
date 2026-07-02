"""
Validate the Oculus teleop pose math WITHOUT any Oculus hardware or robot.

Run from the repo root:

    uv run python odyssey/examples/check_oculus_teleop.py

This does NOT require oculus_reader to be installed and does NOT start a robot
process. It exercises OculusTeleopWorkstation.compute_commanded_pose (a
staticmethod) with synthetic controller poses. The OculusReader import in
oculus_teleop.py is lazy (inside __init__), so importing the module and calling
the staticmethod is safe without the hardware library.
"""
import numpy as np
from pydrake.all import RigidTransform, RotationMatrix, Quaternion
from odyssey.examples.oculus_teleop import OculusTeleopWorkstation

compute = OculusTeleopWorkstation.compute_commanded_pose

results = []


def check(name, cond):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}")
    results.append(cond)


def poses_close(a: RigidTransform, b: RigidTransform, atol=1e-9):
    return np.allclose(a.GetAsMatrix4(), b.GetAsMatrix4(), atol=atol)


# A representative base ee pose (rotated + translated so bugs show up).
base_ee = RigidTransform(RotationMatrix.MakeZRotation(0.3), np.array([0.4, -0.1, 0.5]))

# ---------------------------------------------------------------------------
# (a) grip NOT held -> commanded pose == base ee pose
# ---------------------------------------------------------------------------
controller = RigidTransform(RotationMatrix.MakeYRotation(0.2), np.array([1.0, 2.0, 3.0]))
commanded, new_base_ee, new_base_ctrl = compute(
    base_ee, None, controller, grip_held=False, prev_grip=False
)
check("(a) grip not held: commanded == base ee pose", poses_close(commanded, base_ee))
check("(a) grip not held: base controller pose tracks current controller",
      poses_close(new_base_ctrl, controller))

# ---------------------------------------------------------------------------
# (b) grip held and controller translates by delta -> commanded translates by
#     the same delta (pure translation, no rotation change).
# ---------------------------------------------------------------------------
# First, with no grip, anchor the base controller pose.
base_ctrl = RigidTransform(RotationMatrix.MakeYRotation(0.2), np.array([1.0, 2.0, 3.0]))
_, _, base_ctrl = compute(base_ee, None, base_ctrl, grip_held=False, prev_grip=False)

delta = np.array([0.05, -0.02, 0.10])
moved_ctrl = RigidTransform(base_ctrl.rotation(), base_ctrl.translation() + delta)
commanded, _, _ = compute(base_ee, base_ctrl, moved_ctrl, grip_held=True, prev_grip=False)

# The relative controller motion is measured in the base controller frame and
# then applied in the base ee frame, so a world-frame controller delta maps to
# the ee via: ee_delta = base_ee.R @ base_ctrl.R^-1 @ delta. When the base ee
# and base controller share the same orientation this reduces to `delta`.
expected_pos = base_ee.translation() + (
    base_ee.rotation().matrix() @ base_ctrl.rotation().inverse().matrix() @ delta
)
check("(b) grip held: translation delta matches relative-motion mapping",
      np.allclose(commanded.translation(), expected_pos, atol=1e-9))
check("(b) grip held: rotation unchanged when controller only translated",
      np.allclose(commanded.rotation().matrix(), base_ee.rotation().matrix(), atol=1e-9))

# Sanity: when base ee and base controller orientations match, a world delta on
# the controller produces the SAME world delta on the ee (1:1 translation).
aligned_ctrl = RigidTransform(base_ee.rotation(), np.array([0.0, 0.0, 0.0]))
aligned_moved = RigidTransform(base_ee.rotation(), delta)
commanded_aligned, _, _ = compute(
    base_ee, aligned_ctrl, aligned_moved, grip_held=True, prev_grip=False
)
check("(b) grip held: aligned frames -> ee translates by exactly delta",
      np.allclose(commanded_aligned.translation(), base_ee.translation() + delta, atol=1e-9))

# Also verify a rotation-only controller move rotates the ee correctly.
rot_delta = RotationMatrix.MakeXRotation(0.25)
rotated_ctrl = RigidTransform(base_ctrl.rotation() @ rot_delta, base_ctrl.translation())
commanded_rot, _, _ = compute(base_ee, base_ctrl, rotated_ctrl, grip_held=True, prev_grip=False)
# expected: base_ee @ (base_ctrl^-1 @ rotated_ctrl)
expected_rot = base_ee @ (base_ctrl.inverse() @ rotated_ctrl)
check("(b) grip held: rotation delta matches", poses_close(commanded_rot, expected_rot))

# When controller sits exactly at the anchor, commanded == base ee pose.
commanded_id, _, _ = compute(base_ee, base_ctrl, base_ctrl, grip_held=True, prev_grip=False)
check("(b) grip held: no motion -> commanded == base ee pose",
      poses_close(commanded_id, base_ee))

# ---------------------------------------------------------------------------
# (c) re-anchoring on release: after moving with grip held then releasing, the
#     base ee pose becomes the commanded pose and base controller re-anchors to
#     the controller's current pose. A subsequent grip press should then move
#     relative to the NEW anchor (no jump).
# ---------------------------------------------------------------------------
# Move with grip held.
commanded_moved, base_ee2, base_ctrl2 = compute(
    base_ee, base_ctrl, moved_ctrl, grip_held=True, prev_grip=False
)
# Release the grip (prev_grip=True, now False).
commanded_rel, base_ee_after, base_ctrl_after = compute(
    base_ee2, base_ctrl2, moved_ctrl, grip_held=False, prev_grip=True
)
check("(c) release: base ee pose re-anchors to last commanded pose",
      poses_close(base_ee_after, commanded_moved))
check("(c) release: base controller re-anchors to current controller",
      poses_close(base_ctrl_after, moved_ctrl))

# After release, re-grip and move by a new delta -> commanded should be the
# re-anchored ee pose plus only the NEW delta (no accumulated jump).
delta2 = np.array([-0.03, 0.04, 0.0])
moved_ctrl2 = RigidTransform(moved_ctrl.rotation(), moved_ctrl.translation() + delta2)
commanded_regrip, _, _ = compute(
    base_ee_after, base_ctrl_after, moved_ctrl2, grip_held=True, prev_grip=False
)
expected_regrip_pos = base_ee_after.translation() + (
    base_ee_after.rotation().matrix()
    @ base_ctrl_after.rotation().inverse().matrix()
    @ delta2
)
check("(c) re-grip after release: moves only by new delta from new anchor",
      np.allclose(commanded_regrip.translation(), expected_regrip_pos, atol=1e-9))

# ---------------------------------------------------------------------------
print()
if all(results):
    print(f"ALL {len(results)} CHECKS PASSED")
else:
    n_fail = sum(1 for r in results if not r)
    print(f"{n_fail}/{len(results)} CHECKS FAILED")
    raise SystemExit(1)
