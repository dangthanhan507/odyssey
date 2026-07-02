"""
Validate the gamepad teleop mapping math WITHOUT a joystick, robot, or gripper.

Run from the repo root:

    uv run python odyssey/examples/check_gamepad_teleop.py

This does NOT require a connected gamepad and does NOT start a robot process or
publish to LCM. It exercises the pure mapping helpers on
GamepadTeleopWorkstation (staticmethods + stick->velocity), which are factored
out so they can be validated on their own. pygame is imported lazily (inside
__init__), so importing the module and calling these helpers is safe without a
joystick present.
"""
import numpy as np
from odyssey.examples.gamepad_teleop import (
    GamepadTeleopWorkstation,
    stick_deadzone,
)

trigger_to_gripper_mm = GamepadTeleopWorkstation.trigger_to_gripper_mm
normalize_trigger = GamepadTeleopWorkstation.normalize_trigger

results = []


def check(name, cond):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}")
    results.append(bool(cond))


OPEN_MM = 100.0
CLOSED_MM = 0.0

# ---------------------------------------------------------------------------
# Trigger normalization: pygame triggers commonly rest at -1 and read +1 when
# fully pressed.
# ---------------------------------------------------------------------------
check("normalize_trigger: rest (-1.0) -> 0.0", np.isclose(normalize_trigger(-1.0), 0.0))
check("normalize_trigger: full (+1.0) -> 1.0", np.isclose(normalize_trigger(1.0), 1.0))
check("normalize_trigger: mid (0.0) -> 0.5", np.isclose(normalize_trigger(0.0), 0.5))

# ---------------------------------------------------------------------------
# Proportional gripper mapping: released -> open, squeezed -> closed, linear.
# ---------------------------------------------------------------------------
check("gripper: released (0.0) -> open_mm",
      np.isclose(trigger_to_gripper_mm(0.0, OPEN_MM, CLOSED_MM), OPEN_MM))
check("gripper: squeezed (1.0) -> closed_mm",
      np.isclose(trigger_to_gripper_mm(1.0, OPEN_MM, CLOSED_MM), CLOSED_MM))
check("gripper: half (0.5) -> midpoint",
      np.isclose(trigger_to_gripper_mm(0.5, OPEN_MM, CLOSED_MM), (OPEN_MM + CLOSED_MM) / 2.0))
check("gripper: monotonic decreasing in trigger",
      trigger_to_gripper_mm(0.25, OPEN_MM, CLOSED_MM) > trigger_to_gripper_mm(0.75, OPEN_MM, CLOSED_MM))
check("gripper: out-of-range trigger clipped to [closed, open]",
      (trigger_to_gripper_mm(-5.0, OPEN_MM, CLOSED_MM) == OPEN_MM)
      and (trigger_to_gripper_mm(5.0, OPEN_MM, CLOSED_MM) == CLOSED_MM))
# Also works with a non-zero closed value (e.g. a gripper that never fully closes).
check("gripper: respects non-zero closed_mm",
      np.isclose(trigger_to_gripper_mm(1.0, 100.0, 20.0), 20.0))

# ---------------------------------------------------------------------------
# Radial deadzone: small inputs are zeroed, large inputs pass through scaled.
# ---------------------------------------------------------------------------
check("deadzone: tiny input zeroed",
      np.allclose(stick_deadzone(0.05, 0.0, 0.0, deadzone=0.1), 0.0))
check("deadzone: full input preserved (unit stays unit)",
      np.allclose(stick_deadzone(1.0, 0.0, 0.0, deadzone=0.1), np.array([1.0, 0.0, 0.0])))

# ---------------------------------------------------------------------------
# Stick -> velocity mapping. Build a minimal instance WITHOUT running __init__
# (which would need viser + a joystick) so we can test sticks_to_velocity.
# ---------------------------------------------------------------------------
ws = GamepadTeleopWorkstation.__new__(GamepadTeleopWorkstation)
ws.velocity_limit = 0.1
ws.deadzone = 0.1

# Neutral sticks -> zero twist.
V0 = ws.sticks_to_velocity(0.0, 0.0, 0.0, 0.0)
check("velocity: neutral sticks -> zero twist", np.allclose(V0, 0.0))

# Full left-stick x -> +x linear velocity at the velocity limit; no angular.
Vx = ws.sticks_to_velocity(1.0, 0.0, 0.0, 0.0)
check("velocity: left-x -> +x linear at velocity_limit",
      np.isclose(Vx[3], ws.velocity_limit) and np.allclose(Vx[:3], 0.0))

# Full left-stick y -> -y linear velocity (sign matches spacemouse example).
Vy = ws.sticks_to_velocity(0.0, 1.0, 0.0, 0.0)
check("velocity: left-y -> -y linear at velocity_limit",
      np.isclose(Vy[4], -ws.velocity_limit))

# Right-stick vertical (ry) -> z linear velocity.
Vz = ws.sticks_to_velocity(0.0, 0.0, 0.0, 1.0)
check("velocity: right-y -> +z linear at velocity_limit",
      np.isclose(Vz[5], ws.velocity_limit))

# Right-stick horizontal (rx) -> angular x term (nonzero, no linear leakage).
Vr = ws.sticks_to_velocity(0.0, 0.0, 1.0, 0.0)
check("velocity: right-x -> angular term, no linear",
      (Vr[0] != 0.0) and np.allclose(Vr[3:], 0.0))

# Output is always a length-6 twist.
check("velocity: output is length-6 twist", V0.shape == (6,))

# ---------------------------------------------------------------------------
print()
if all(results):
    print(f"ALL {len(results)} CHECKS PASSED")
else:
    n_fail = sum(1 for r in results if not r)
    print(f"{n_fail}/{len(results)} CHECKS FAILED")
    raise SystemExit(1)
