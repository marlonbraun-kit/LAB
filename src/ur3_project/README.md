# UR3 Pick-and-Place (MoveIt2)

**Gripper: Zimmer HRC 03** (2-finger parallel gripper, mounted on UR3 flange via adapter plate)

ROS 2 Humble package that drives a UR3 robot arm through a full
pick-and-place cycle using MoveIt2 (OMPL + KDL IK + Cartesian path service).

---

## Package layout

```
ur3_project/
├── config/                         # MoveIt + ros2_control configuration
├── launch/
│   ├── pick_place_moveit.launch.py # MAIN launch file (MoveIt2 stack)
│   └── view_ur3_camera.launch.py   # Standalone URDF/camera viewer
├── meshes/                         # Gripper + can + camera STLs
├── rviz/                           # RViz configs
├── src/
│   ├── pick_place_manager_node.py        # State machine (MoveIt2 client)
│   ├── planning_scene_manager_node.py    # MoveIt planning scene (table + cans)
│   ├── depth_camera_node.py              # Fake camera source (sim only)
│   └── rviz_visualizer_node.py           # Markers + place-slot visualisation
└── urdf/                           # URDF/SRDF/xacro

Custom messages live in **ur3_interfaces** (sibling package):
  ur3_interfaces/msg/CanDetection.msg
  ur3_interfaces/msg/CanDetectionArray.msg
```

---

## Build

```bash
cd ~/ros2_ws/LAB
colcon build --packages-select ur3_interfaces ur3_project
source install/setup.bash
```

Rebuild just the project after code changes:

```bash
colcon build --packages-select ur3_project && source install/setup.bash
```

---

## Run

The launch file has three modes selected by two arguments (`rviz`, `debug`).
The hardware target is **derived** from those flags — the user does not set
`use_fake_hardware` directly anymore.

| Mode | Command | Hardware | RViz | Trajectory ghost | Step gate | Fake camera |
|---|---|---|---|---|---|---|
| **Real-robot run** (default) | `ros2 launch ur3_project pick_place_moveit.launch.py` | real UR3 | off | — | — | off |
| **Home simulation** | `ros2 launch ur3_project pick_place_moveit.launch.py rviz:=true` | `mock_components/GenericSystem` | on | off (sim config) | — | on |
| **Real-robot debug** | `ros2 launch ur3_project pick_place_moveit.launch.py rviz:=true debug:=true` | real UR3 | on | on (debug config) | on | off |

Argument summary:

| Argument | Default | Effect |
|---|---|---|
| `rviz` | `false` | Start RViz alongside the pipeline. |
| `debug` | `false` | Only meaningful with `rviz:=true`. Forces real hardware, swaps to the debug RViz config, and enables the step gate inside `pick_place_manager_node`. |
| `fake_camera` | `auto` | `'auto'` starts `depth_camera_node` only in home-sim mode. `'true'`/`'false'` overrides regardless of the rviz/debug settings — set to `false` to silence the fake source while you publish `/target_can_pose` manually. |

Hardware-selection rule:

```
use_fake_hardware = (rviz == true) and (debug == false)
```

So `rviz:=true debug:=true` deliberately disables fake hardware — the point of
that mode is to inspect the planned trajectory in RViz before it is sent to
the real robot.

### Step gate (debug mode)

When `debug:=true` is passed, `pick_place_manager_node` receives the parameter
`debug_step:=true` and gates every motion behind two Enter presses on the
launching terminal:

1. Plan only — `MoveGroup` is invoked with `plan_only=True` (joint goals) or
   the existing `GetCartesianPath` path is republished on
   `/display_planned_path`. The Trajectory display in RViz shows the ghost
   robot.
2. **Press Enter** → cached trajectory is sent to `/execute_trajectory`. The
   real robot moves.
3. **Press Enter** → the FSM transitions to the next state, which plans the
   next segment.

In any non-debug mode the gate is a no-op and `MoveGroup` plans+executes in a
single shot.

---

## External topic contract

The teammates' camera/recognition nodes publish here on the real robot. For at-home
testing publish them yourself (or rely on `depth_camera_node` in sim mode):

| Topic | Type | Direction | Notes |
|---|---|---|---|
| `/target_can_pose` | `ur3_interfaces/CanDetectionArray` | external → manager | each `CanDetection.source` is `"top"` (above-camera, positions only) or `"front"` (front-camera, positions + `class_name`) |
| `/human_proximity` | `std_msgs/Float32` | external → manager | `0.0` = hand close (danger), `1.0` = safe. Below `HUMAN_PROXIMITY_THRESHOLD = 0.5` the next motion segment uses 0.5 × normal vel/acc scaling. |

## Operator topics

| Topic | Type | Notes |
|---|---|---|
| `/pick_command` | `std_msgs/String` | comma-separated classes, e.g. `"coke,mahou"`. Only honoured while the manager is in `WAIT_FOR_COMMAND`. |
| `/clear_place_zone` | `std_msgs/Empty` | resets the 2x2 place-slot map so future picks fill from slot 0 again. |

Manual-publish examples (testing at home):

```bash
# fake "top" localisation scan (positions only)
ros2 topic pub --once /target_can_pose ur3_interfaces/msg/CanDetectionArray \
  '{header: {frame_id: base_link}, detections: [
    {position: {x: 0.30, y: 0.20, z: 0.06}, source: "top"},
    {position: {x: 0.36, y: 0.16, z: 0.06}, source: "top"}
  ]}'

# fake "front" identification scan (positions + classes)
ros2 topic pub --once /target_can_pose ur3_interfaces/msg/CanDetectionArray \
  '{header: {frame_id: base_link}, detections: [
    {class_name: "coke",  position: {x: 0.30, y: 0.20, z: 0.06}, source: "front"},
    {class_name: "mahou", position: {x: 0.36, y: 0.16, z: 0.06}, source: "front"}
  ]}'

# trigger the cycle
ros2 topic pub --once /pick_command std_msgs/String '{data: "coke,mahou"}'

# simulate human approach
ros2 topic pub --once /human_proximity std_msgs/Float32 '{data: 0.0}'

# reset slot map
ros2 topic pub --once /clear_place_zone std_msgs/Empty '{}'
```

## Node overview

| Node | Role | Key topics / services |
|---|---|---|
| `pick_place_manager_node` | State machine; calls MoveIt2 | pub `/pick_place_state`, pub `/gripper_controller/commands`, pub `/current_pick_target`, pub `/display_planned_path` (debug only); sub `/target_can_pose`, `/pick_command`, `/clear_place_zone`, `/human_proximity`; action `/move_action`, `/execute_trajectory`; srv `/compute_cartesian_path` |
| `planning_scene_manager_node` | Keeps MoveIt planning scene in sync | sub `/target_can_pose` (CanDetectionArray), `/current_pick_target`, `/pick_place_state`; srv `/apply_planning_scene` |
| `depth_camera_node` | Sim-only fake camera source | pub `/target_can_pose` (alternates `source="top"` and `source="front"`), pub `/camera/depth/image_raw` |
| `rviz_visualizer_node` | Markers + TF for RViz | pub `/visualization_markers`, broadcasts TF `pick_zone`, `place_zone`, `can_active`, `approach_point` |

### Startup sequence (managed by launch file)

```
ros2_control_node + robot_state_publisher
  └─ spawner: joint_state_broadcaster
       └─ spawner: joint_trajectory_controller + gripper_controller
            └─ move_group
                 └─ (5 s delay) planning_scene_manager_node
                                pick_place_manager_node    (debug_step from `debug` arg)
                                depth_camera_node
                                rviz_visualizer_node
                                rviz2                       (only when rviz:=true)
```

---

## State machine

```
BOOT
 └─ MOVE_TO_WAIT          joint-space: arm → WAIT_DOWN_JOINTS  (camera looks down)
 └─ WAIT_FOR_COMMAND      hold until /pick_command (e.g. "coke,mahou")
 └─ LOCALIZE              wait for /target_can_pose with source="top"
                          (positions of every can in the pickup zone)
 └─ ORIENT_FORWARD        joint-space: arm → WAIT_FORWARD_JOINTS
 └─ MOVE_TO_IDENTIFY      joint-space: arm → IDENTIFY_JOINTS
 └─ IDENTIFY_FROM_FRONT   wait for /target_can_pose with source="front"
                          and pair each class with its localised XY twin
 ┌─→ NEXT_TARGET          pop next class from queue, pick the next free
 │                        place-zone slot, solve IK chain
 │ └─ PLAN_TO_PREGRASP
 │ └─ CARTESIAN_APPROACH
 │ └─ GRASP
 │ └─ CARTESIAN_LIFT
 │ └─ PLAN_TO_PLACE        targets the next free 2x2 grid slot
 │ └─ RELEASE              also marks that slot as filled
 │ └─ CARTESIAN_RETREAT
 └── AFTER_RETREAT (loop back to NEXT_TARGET)
 └─ ALL_DONE               command queue empty (or place zone full)
 └─ ORIENT_DOWN_AT_WAIT
 └─ WAIT_FOR_COMMAND       (cycle ready for the next /pick_command)
```

Any motion failure aborts the remaining queue and falls back to
`WAIT_FOR_COMMAND` (still preserving the place-slot map).

The slot map persists across cycles. Send `/clear_place_zone` to free all
four slots before testing a fresh run.

---

## Planning strategy

### Joint-space goals for named poses
`MOVE_TO_WAIT`, `ORIENT_FORWARD`, and `ORIENT_DOWN_AT_WAIT` use explicit
`JointConstraint` goals (`WAIT_DOWN_JOINTS` / `WAIT_FORWARD_JOINTS`) instead of
pose goals. This eliminates IK branch ambiguity: OMPL plans directly in joint
space to a single known configuration and can never over-rotate the base joint.

### Pose goals without path constraints
`PLAN_TO_PREGRASP` and `PLAN_TO_PLACE` use pose goals with OMPL RRTConnect but
**no path constraints**. Orientation path constraints on a 6-DOF arm with KDL IK
almost always cause planning to time out because the constraint manifold is too
thin for the sampler. The correct orientation is achieved by:
- Starting from a configuration already in the right IK branch
- Using a tight goal orientation tolerance (0.05 rad ≈ 3°)

### Cartesian paths for short straight-line segments
`CARTESIAN_APPROACH`, `CARTESIAN_LIFT`, and `CARTESIAN_RETREAT` use
`GetCartesianPath` with:
- `max_step = 0.005 m` (5 mm waypoints)
- `jump_threshold = 5.0` (prevents IK branch flips between waypoints — setting
  this to 0 disables the check and produces curved, non-straight paths)
- `min_fraction = 1.0` (reject any incomplete path)
- `CARTESIAN_RETREAT` uses `avoid_collisions=False` because the TCP is right
  next to the just-placed can and collision-aware IK fails on the first waypoint

---

## Critical geometry notes

### gripper_base_joint offset (⚠ important for FK/IK)

`gripper.urdf.xacro` defines `gripper_base_joint` with `rpy="0 0 -1.5707963"`.
This is a **Rz(−90°)** rotation between `tool0` and the gripper frame. The full
TCP chain is:

```
fk_tcp(q) = fk_tool0(q) · Rz(−90°) · translate(0, 0, 0.173)
```

All named joint configurations and quaternion constants in
`pick_place_manager_node.py` were computed using this full chain targeting
`gripper_tcp_link`, **not** `tool0`. The KDL kinematics plugin is also
configured with `tip_link: gripper_tcp_link`.

### Orientation quaternions (in `base_link` frame)

The robot is mounted with shoulder_pan rotated 90° from the canonical UR3
spawn so the working direction is **+Y** in `base_link`. All TCP orientation
quaternions reflect this.

| Name | Value `(x,y,z,w)` | TCP Z-axis | Use |
|---|---|---|---|
| `FORWARD_QUAT` | `(0.0, 0.7071, 0.7071, 0.0)` | `[0,1,0]` +Y | Grasp, approach, carry, place |
| `DOWN_QUAT`    | `(0.0, 1.0, 0.0, 0.0)` | `[0,0,−1]` | Wait poses (camera pointed down) |

For `FORWARD_QUAT`: TCP-Z points along +Y (approach direction), TCP-Y points
along +Z (fingers vertical). For `DOWN_QUAT`: TCP-Z points along −Z (down),
TCP-Y along +Y so the wrist-mounted camera ends up on the +Y front side.

### Named joint configurations

The two wait-pose joint vectors (`wait_forward_joints`, `wait_down_joints`)
and the front-camera identification pose (`identify_joints`) are **computed at
node startup** by running `ik_for_tcp` against `WAIT_TCP` / `IDENTIFY_TCP` with
the matching orientation quaternion. The hard-coded `_WAIT_FORWARD_SEED` and
`_WAIT_DOWN_SEED` dictionaries in `pick_place_manager_node.py` are only IK
seeds — they pick the desired branch (elbow direction, wrist_3 sign) but the
final joint values come from LM optimisation against the actual chain.

This means `WAIT_TCP` (or `IDENTIFY_TCP`) can be edited without manually
re-tuning every joint angle. If a value becomes unreachable from its seed the
node logs an error and falls back to the seed itself.

**Seeds (radians):**

`_WAIT_FORWARD_SEED` — TCP at `WAIT_TCP` facing +Y, wrist_3 = +π/2:
```
shoulder_pan: −0.2647   shoulder_lift: −1.2651   elbow:  1.0231
wrist_1:       0.2420   wrist_2:       −0.2647   wrist_3: 1.5708
```

`_WAIT_DOWN_SEED` — TCP at `WAIT_TCP` pointing down, camera on +Y side:
```
shoulder_pan:  0.2711   shoulder_lift: −1.1688   elbow:  0.4602
wrist_1:      −0.8622   wrist_2:       −1.5708   wrist_3: −2.8705
```

**All pick-cycle IK solutions share the wrist_3 ≈ +π/2 branch.** This is
essential — a 2π jump in wrist_3 between consecutive trajectory points triggers
`PATH_TOLERANCE_VIOLATED` in the controller. The two-stage IK in
`_compute_cycle_ik` (high pre-place pose first, then low place pose seeded
from it) keeps the entire cycle in this single branch.

**Spawn pose** (`config/initial_positions.yaml`) matches the SRDF
`wait_forward` named state so the robot boots already at `WAIT_FORWARD_JOINTS`:
```
shoulder_pan:  0.2210   shoulder_lift: −2.2994   elbow:  1.8025
wrist_1:       0.4969   wrist_2:        0.2210   wrist_3: 1.5708
gripper_left_finger_joint: 0.055   (gripper open)
```

---

## Task-space parameters

| Parameter | Value | Meaning |
|---|---|---|
| `WAIT_TCP` | (0.30, 0.20, 0.26) | TCP position of both wait configurations |
| `IDENTIFY_TCP` | (0.30, 0.09, 0.10) | TCP pose from which the wrist camera frames the pickup zone from the front |
| `APPROACH_OFFSET_Y` | 0.08 m | Stand-off (in `−y`) from the can centre during pre-grasp |
| `LIFT_Z` | 0.12 m | Height of Cartesian lift / retreat above the can |
| `PLACE_ZONE_CENTER` | (−0.30, 0.20) | Centre of the 2x2 place grid in XY |
| `PLACE_GRID_SPACING` | 0.12 m | Centre-to-centre spacing between adjacent slots |
| `PLACE_TCP_Z` | 0.15 m | TCP height when placing (can bottom ≈ 2 cm above the table) |
| Slot fill order | back-left → back-right → front-left → front-right | "back" = +Y |
| `HUMAN_PROXIMITY_THRESHOLD` | 0.5 | Below this, next motion uses 0.5 × normal speed |
| Pick zone | (0.30, 0.20) XY | Marker only — actual positions come from the top-camera scan |

---

## Planning scene

`planning_scene_manager_node` maintains the MoveIt collision world:

- **Static objects** (added once on first tick, in `base_link` frame so they
  match the rviz visualizer markers exactly):
  - `table_top`: 1.5 × 0.8 × 0.05 m box at (0.0, 0.0, −0.025) — workspace
    surface flush with the arm mount
  - `backboard`: 1.5 × 0.05 × 0.5 m box at (0.0, −0.325, 0.25) — vertical wall
    behind the robot
  - SRDF disables `base_link`/`base_link_inertia` ↔ `table_top` collision pairs
    so the flush mount does not register as a permanent collision.
- **Dynamic objects** (refreshed at 2 Hz):
  - `can_<i>`: one cylinder (r=0.040 m, h=0.130 m) per detection in the
    last `/target_can_pose` array. The active target — designated via
    `/current_pick_target` — is suppressed during the approach so the planner
    can drive into the can's location.

**Static-environment collision avoidance:**  
Robot-vs-environment collisions are checked by MoveIt/FCL against the actual
UR3 collision meshes already referenced by the URDF (`ur_description`'s
per-link convex hulls). No manual link-thickness tuning is required — adding
the `table_top` and `backboard` boxes as `CollisionObject`s is enough for OMPL
to route the entire arm geometry around them.

**Can suppression during approach:**  
When the state machine enters `PLAN_TO_PREGRASP` the target can is removed from
the scene so OMPL/IK can freely plan the arm into the can's location. The can is
re-added as a free object when the state returns to `ORIENT_FORWARD` (start of
next cycle). While grasped, the can is attached to `gripper_base_link` as an
`AttachedCollisionObject` (touch links: gripper base + both fingers).

---

## Simulated camera source (sim mode only)

`depth_camera_node` is a stand-in for the teammates' camera/recognition nodes
during home simulation. It only runs when `fake_camera:=true` (or `auto`
which resolves to true in home-sim mode).

It publishes a `CanDetectionArray` to `/target_can_pose` at 1 Hz, alternating
between a `source="top"` message (positions, no class) and a `source="front"`
message (positions + class) so the manager can step through both phases.

Default cans (override via parameters `fake_can_<i>_class`/`x`/`y`/`z`):

| i | class | xy |
|---|---|---|
| 0 | coke | (0.30, 0.20) |
| 1 | mahou | (0.36, 0.16) |
| 2 | fanta | (0.24, 0.24) |

For real-robot runs the node is *not* started — the teammates' nodes publish
on `/target_can_pose` directly.

---

## Monitoring

```bash
# Watch state machine transitions
ros2 topic echo /pick_place_state

# What the camera nodes are reporting
ros2 topic echo /target_can_pose

# Active pick target (one PoseStamped per pick)
ros2 topic echo /current_pick_target

# Hand-distance signal driving the speed scaling
ros2 topic echo /human_proximity

# Watch joint states
ros2 topic echo /joint_states --field name,position
```

---

## MoveIt configuration files

| File | Purpose |
|---|---|
| `config/moveit_kinematics.yaml` | KDL plugin, `tip_link: gripper_tcp_link`, timeout 0.5 s, 30 attempts |
| `config/moveit_ompl_planning.yaml` | OMPL pipeline, default planner RRTConnect |
| `config/moveit_controllers.yaml` | Maps `joint_trajectory_controller` and `gripper_controller` to MoveIt |
| `config/moveit_joint_limits.yaml` | `vel_scaling=0.3`, `acc_scaling=0.3`, per-joint velocity/acceleration limits |
| `urdf/ur3.srdf.xacro` | Planning groups (`ur_manipulator` chain base→`gripper_tcp_link`, `gripper`), named states (`wait_forward`, `home`, `up`, gripper open/grip/closed), ACM disabled-collision pairs |

---

## SRDF named states

| Name | Group | Description |
|---|---|---|
| `wait_forward` | `ur_manipulator` | Forward-facing wait pose: TCP at `WAIT_TCP`=(0.30, 0.20, 0.35) under `FORWARD_QUAT`. Identical to `config/initial_positions.yaml` so the spawn config sits exactly at this state. |
| `home` | `ur_manipulator` | Legacy forward-facing pose; kept for reference only |
| `up` | `ur_manipulator` | All-zeros upright configuration |
| `open` | `gripper` | Fingers at 0.055 m |
| `grip` | `gripper` | Fingers at 0.040 m |
| `closed` | `gripper` | Fingers at 0.000 m |

---

## Known issues / design decisions

- **No orientation path constraint on PLAN_TO_PLACE**: start and goal are both in
  the wrist_3=+π/2 branch with FORWARD_QUAT, so the joint-interpolated path has
  near-zero orientation error without a constraint. Adding one reliably causes
  OMPL to time out.

- **CARTESIAN_RETREAT skips collision checking**: the placed can is right next to
  the TCP at release. Collision-aware IK cannot solve the first Cartesian
  waypoint (fraction=0). The upward retreat trajectory is geometrically clear, so
  `avoid_collisions=False` is safe here.

- **KDL IK branch consistency**: the planner must find IK solutions in the
  wrist_3≈+π/2 branch for all pick-cycle poses. The seed state provided to OMPL
  is the current robot state, which is already in that branch after
  `ORIENT_FORWARD`. Departing from `WAIT_IDLE` before `ORIENT_FORWARD` completes
  risks branch inconsistency.

- **No Cartesian descent for placement**: an earlier revision used a
  `CARTESIAN_PLACE` segment to lower the can onto the table. With the table
  modelled as a real collision object the Cartesian descent failed at 92% (the
  attached can / forearm clipped the table near the bottom of the path), so
  `PLAN_TO_PLACE` was retargeted to the final `PLACE_TCP` directly and
  `CARTESIAN_PLACE` was removed. The placement is now a single joint-space
  goal from the lift pose to the place pose.

- **Two-stage IK seed for `q_place`**: `PLACE_TCP` lives across the workspace
  from the standard wait pose, and seeding LM directly from `WAIT_FORWARD_JOINTS`
  at the low place height converged to an unreachable / colliding contortion.
  `_compute_cycle_ik` now solves a high pre-place pose first
  (`PLACE_TCP[2] + LIFT_Z`, seeded from `WAIT_FORWARD_JOINTS`), then re-solves
  the low place pose seeded from that — the same IK branch all the way down.

- **`PLACE_TCP[2]` ≥ 0.15 m elbow clearance**: with the table modelled,
  reaching across to (−0.30, 0.30) at very low TCP heights pushes the forearm
  into the table even when the TCP itself clears it. Lowering further than
  ~0.15 m makes the goal joint configuration land in collision and MoveIt
  rejects the plan with `error_code=99999`. Going below 0.15 m would require
  biasing the IK toward an elbow-up branch (custom seed).
