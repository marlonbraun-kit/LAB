#!/usr/bin/env python3
"""Pick-and-place state machine driven by MoveIt2.

Cycle (one /order can trigger many picks)
-----------------------------------------------
BOOT
  -> MOVE_TO_WAIT          joint goal -> WAIT_FORWARD_JOINTS
  -> WAIT_FOR_COMMAND      hold until /order (e.g. "coke,mahou")
  -> ORIENT_FORWARD        joint goal -> WAIT_FORWARD_JOINTS
  -> MOVE_TO_IDENTIFY      joint goal -> IDENTIFY_JOINTS  (FORWARD_QUAT branch)
  -> ROTATE_FOR_IDENTIFY   joint goal -> IDENTIFY_JOINTS_ROTATED
                           (wrist_3 += +π/2 so the camera frames the cans)
  -> IDENTIFY_FROM_FRONT   wait for a /front_detections message from
                           native_vision_node — each detection carries both
                           class_id and 3D position in camera_optical_link.
                           Positions are transformed to base_link and the
                           z component is overridden with FIXED_CAN_Z.
  -> ROTATE_BACK_FROM_IDENTIFY
                           joint goal -> IDENTIFY_JOINTS (wrist_3 back to
                           FORWARD_QUAT branch) so pregrasp IK seeds from
                           the canonical wrist branch.
  -> NEXT_TARGET           pop next class from queue, look up matched position,
                           solve IK chain.  When queue empty -> ALL_DONE.
  -> PLAN_TO_PREGRASP
  -> CARTESIAN_APPROACH
  -> GRASP
  -> CARTESIAN_LIFT
  -> PLAN_TO_PLACE         next free slot in 2x2 place grid (filled back-first)
  -> RELEASE
  -> CARTESIAN_RETREAT
  -> NEXT_TARGET           (loop)
  -> ALL_DONE
  -> ORIENT_DOWN_AT_WAIT
  -> WAIT_FOR_COMMAND

Place zone slot map persists across cycles until /clear_place_zone is received.

Speed scaling
-------------
/human_proximity is a Float32 in [0, 1].  When it falls below
HUMAN_PROXIMITY_THRESHOLD, every subsequent motion-plan request uses the
slow scaling (0.5 x normal).  This gates the robot at segment boundaries —
fully dynamic in-flight slowing would require the ur_driver speed-slider
service and is not implemented here.

Gripper: Float64MultiArray on /gripper_controller/commands.
"""
import math
import sys
import threading

import numpy as np
from scipy.optimize import least_squares

import rclpy
from rclpy.action import ActionClient
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from std_msgs.msg import String, Float64MultiArray, Float32, Empty, Bool
from ur_msgs.srv import SetIO
from geometry_msgs.msg import Pose, PoseStamped, Vector3, PointStamped
from sensor_msgs.msg import JointState

from moveit_msgs.action import ExecuteTrajectory, MoveGroup
from moveit_msgs.srv import GetCartesianPath
from moveit_msgs.msg import (
    Constraints,
    JointConstraint,
    MotionPlanRequest,
    PlanningOptions,
    WorkspaceParameters,
)

import tf2_ros
import tf2_geometry_msgs  # noqa: F401  (registers PointStamped do_transform)

from ur3_interfaces.msg import CanDetection, CanDetectionArray


# ---------------------------------------------------------------------------
# Orientation quaternions (in planning frame = base_link)
# ---------------------------------------------------------------------------
FORWARD_QUAT = (0.5,       0.5,       0.5,       0.5)

# ---------------------------------------------------------------------------
# IK seeds for wait poses
# ---------------------------------------------------------------------------
_WAIT_FORWARD_SEED = {
    'shoulder_pan_joint':   4.1540926,
    'shoulder_lift_joint': -1.2629,
    'elbow_joint':          1.4456,
    'wrist_1_joint':       -0.1827,
    'wrist_2_joint':       -0.5583,
    'wrist_3_joint':        2.7052640,
}

# Seed for the place pose. Same TCP region as the wait-down pose (camera
# looking down over the place zone) but elbow folded *opposite* to the
# pickup branch so the upper arm stays clear of the table when reaching
# down to PLACE_TCP_Z < 0.15 m. Used as the LM seed for q_high during
# place IK so the whole place chain stays in the elbow-folded branch.
_PLACE_SEED = {
    'shoulder_pan_joint':   5.3407,    # 3.7699 + π/2  (≈ 306°)
    'shoulder_lift_joint': -1.8850,    # -3π/5 (-108°)
    'elbow_joint':         -1.6965,    # ≈ -97.2°
    'wrist_1_joint':        3.5186,    # ≈ 201.6°
    'wrist_2_joint':        0.6283,    # π/5   (36°)
    'wrist_3_joint':       -3.5151,    # original -3.0788 - 25° (gripper +25°)
}

# ---------------------------------------------------------------------------
# MoveIt references
# ---------------------------------------------------------------------------
PLANNING_FRAME = 'base_link'
PLANNING_GROUP = 'ur_manipulator'
EE_LINK = 'gripper_tcp_link'

# ---------------------------------------------------------------------------
# Task-space geometry  (gripper_tcp_link in base_link frame)
# ---------------------------------------------------------------------------
WAIT_TCP = (0.2, -0.3, 0.26)
APPROACH_OFFSET_X = 0.08
LIFT_Z = 0.12
# Vertical offset added to the reported can z when the TCP closes on the can.
# Raises the grasp point up the can body so the gripper closes on the side
# rather than on the rim.
GRASP_Z_OFFSET = 0.05

# Place zone is a 2x2 grid centred on PLACE_ZONE_CENTER in XY.
# Slots are filled in this order (back-row first).  "back" = +Y.
PLACE_ZONE_CENTER = (0.20, 0.30)
PLACE_GRID_SPACING = 0.12
PLACE_TCP_Z = 0.1
PLACE_SLOT_OFFSETS = [
    (+PLACE_GRID_SPACING / 2.0, +PLACE_GRID_SPACING / 2.0),  # 3: front-right
    (+PLACE_GRID_SPACING / 2.0, -PLACE_GRID_SPACING / 2.0),  # 2: front-left
    (-PLACE_GRID_SPACING / 2.0, +PLACE_GRID_SPACING / 2.0),  # 1: back-right
    (-PLACE_GRID_SPACING / 2.0, -PLACE_GRID_SPACING / 2.0),  # 0: back-left
]
NUM_PLACE_SLOTS = len(PLACE_SLOT_OFFSETS)

IDENTIFY_TCP = (0.02, -0.3, 0.15)
# Extra wrist_3 rotation (about the wrist_3_link Z-axis) applied to the IK
# solution for the identify pose. The IK is solved with FORWARD_QUAT and then
# wrist_3_joint += IDENTIFY_WRIST3_OFFSET so the camera frames the picking
# zone from the side without needing a separate goal quaternion.
IDENTIFY_WRIST3_OFFSET = math.pi / 2.0

# Fixed can height (z) in base_link frame, applied to every detection from the
# front camera. The camera's depth estimate is noisy at close range and the
# grasp height is determined geometrically by the table + can geometry, so we
# override the reported z with this constant.
FIXED_CAN_Z = 0.06

# Source frame published by native_vision_node — detections arrive in the
# camera optical frame and are transformed to base_link before use.
CAMERA_OPTICAL_FRAME = 'camera_optical_link'

# YOLO class index → can class name. Detections on /front_detections carry
# class_id (int); the manager maps it to a name for the command queue.
CAN_CLASS_NAMES = {0: 'beer', 1: 'coke', 2: 'lemon', 3: 'orange'}
CAN_CLASS_IDS = {v: k for k, v in CAN_CLASS_NAMES.items()}

# ---------------------------------------------------------------------------
# Gripper commands
# ---------------------------------------------------------------------------
GRIPPER_OPEN = 0.020
GRIPPER_GRIP = 0.002

# ---------------------------------------------------------------------------
# Planner parameters
# ---------------------------------------------------------------------------
JOINT_TOL          = 0.01
PLANNING_TIME_S    = 10.0
PLANNING_ATTEMPTS  = 20

VEL_SCALING_NORMAL = 0.3
ACC_SCALING_NORMAL = 0.3
SLOWDOWN_FACTOR    = 0.5  # speed multiplier when a hand is close
HUMAN_PROXIMITY_THRESHOLD = 0.5  # /human_proximity below this -> slow

CARTESIAN_MAX_STEP       = 0.005
CARTESIAN_JUMP_THRESHOLD = 0.0  # 0 = disabled; allow KDL branch flips along the path
CARTESIAN_MIN_FRACTION   = 1.0

# If no motion progress is observed for this many seconds, abort the motion
# and recover to WAIT_FOR_COMMAND. Catches stuck action futures (e.g. when
# move_group's first goal-response handshake is dropped by the DDS layer).
MOTION_WATCHDOG_S = 30.0


# ---------------------------------------------------------------------------
# UR3e inverse kinematics (numerical, seeded)
# ---------------------------------------------------------------------------
# Nominal DH from ur_description/config/ur3e/default_kinematics.yaml.
# MUST match the ur_type the URDF is loaded with (see launch file +
# ur3_camera_gripper.urdf.xacro). UR3 vs UR3e differ in d4 (~19 mm) and
# d6 (~10 mm); a mismatch shows up as a few-mm position error that
# flips sign with wrist_2.
UR3_DH_A     = [0.0,        -0.24355, -0.2132,  0.0,         0.0,         0.0]
UR3_DH_D     = [0.15185,     0.0,      0.0,     0.13105,     0.08535,     0.0921]
UR3_DH_ALPHA = [math.pi/2.0, 0.0,      0.0,     math.pi/2.0, -math.pi/2.0, 0.0]

_RZ_180 = np.array([
    [-1.0,  0.0, 0.0, 0.0],
    [ 0.0, -1.0, 0.0, 0.0],
    [ 0.0,  0.0, 1.0, 0.0],
    [ 0.0,  0.0, 0.0, 1.0],
])


# tool0 → gripper_tcp_link offset along the gripper extension axis.
# MUST MATCH the gripper_tcp_joint origin Z in urdf/gripper.urdf.xacro.
# This is the only place the value lives on the Python side — if you change
# the URDF, change this too (or vice versa).
TCP_Z_OFFSET = 0.173


def _build_tool0_to_tcp():
    angle = -math.pi + math.radians(25)
    c, s = math.cos(angle), math.sin(angle)
    return np.array([
        [c, -s, 0.0, 0.0],
        [s,  c, 0.0, 0.0],
        [0.0, 0.0, 1.0, TCP_Z_OFFSET],
        [0.0, 0.0, 0.0, 1.0],
    ])


_TOOL0_TO_TCP = _build_tool0_to_tcp()

ARM_JOINT_NAMES = (
    'shoulder_pan_joint',
    'shoulder_lift_joint',
    'elbow_joint',
    'wrist_1_joint',
    'wrist_2_joint',
    'wrist_3_joint',
)


def _dh(a, d, alpha, theta):
    ct, st = math.cos(theta), math.sin(theta)
    ca, sa = math.cos(alpha), math.sin(alpha)
    return np.array([
        [ct, -st * ca,  st * sa, a * ct],
        [st,  ct * ca, -ct * sa, a * st],
        [0.0, sa,       ca,      d],
        [0.0, 0.0,      0.0,     1.0],
    ])


def _fk_tcp(q):
    T = np.eye(4)
    for i in range(6):
        T = T @ _dh(UR3_DH_A[i], UR3_DH_D[i], UR3_DH_ALPHA[i], q[i])
    return _RZ_180 @ T @ _TOOL0_TO_TCP


def _quat_to_R(quat):
    x, y, z, w = quat
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if n < 1e-9:
        return np.eye(3)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w),       2.0 * (x * z + y * w)      ],
        [2.0 * (x * y + z * w),       1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)      ],
        [2.0 * (x * z - y * w),       2.0 * (y * z + x * w),       1.0 - 2.0 * (x * x + y * y)],
    ])


def _ik_residual(q, T_target):
    T = _fk_tcp(q)
    pos_err = T[:3, 3] - T_target[:3, 3]
    R_err = T_target[:3, :3].T @ T[:3, :3]
    rot_err = 0.5 * np.array([
        R_err[2, 1] - R_err[1, 2],
        R_err[0, 2] - R_err[2, 0],
        R_err[1, 0] - R_err[0, 1],
    ])
    return np.concatenate([pos_err, rot_err])


def _shift_near(q, seed):
    while q - seed > math.pi:
        q -= 2.0 * math.pi
    while q - seed < -math.pi:
        q += 2.0 * math.pi
    return q


_IK_RESIDUAL_TOL = 1e-3
_IK_BOUND_HALFWIDTH = math.pi


def ik_for_tcp(x, y, z, quat, seed):
    T_target = np.eye(4)
    T_target[:3, :3] = _quat_to_R(quat)
    T_target[:3, 3]  = (float(x), float(y), float(z))

    seed_arr = np.array([seed[name] for name in ARM_JOINT_NAMES])
    lb = seed_arr - _IK_BOUND_HALFWIDTH
    ub = seed_arr + _IK_BOUND_HALFWIDTH

    res = least_squares(
        _ik_residual, seed_arr, args=(T_target,),
        method='trf', bounds=(lb, ub),
        xtol=1e-10, ftol=1e-10, max_nfev=200,
    )
    if not res.success or float(np.max(np.abs(res.fun))) > _IK_RESIDUAL_TOL:
        return None

    shifted = tuple(_shift_near(q, s) for q, s in zip(res.x.tolist(), seed_arr.tolist()))
    return dict(zip(ARM_JOINT_NAMES, shifted))


# ---------------------------------------------------------------------------
# MoveGroup helpers
# ---------------------------------------------------------------------------

def make_joint_goal_constraints(joint_values, tol=JOINT_TOL):
    c = Constraints()
    c.name = 'joint_goal'
    for name, value in joint_values.items():
        jc = JointConstraint()
        jc.joint_name = name
        jc.position = float(value)
        jc.tolerance_above = tol
        jc.tolerance_below = tol
        jc.weight = 1.0
        c.joint_constraints.append(jc)
    return c


def make_motion_plan_request(goal_constraints, frame, group, vel, acc):
    req = MotionPlanRequest()
    req.group_name = group
    req.num_planning_attempts = PLANNING_ATTEMPTS
    req.allowed_planning_time = PLANNING_TIME_S
    req.max_velocity_scaling_factor = vel
    req.max_acceleration_scaling_factor = acc

    ws = WorkspaceParameters()
    ws.header.frame_id = frame
    ws.min_corner = Vector3(x=-1.0, y=-1.0, z=-1.0)
    ws.max_corner = Vector3(x=1.0, y=1.0, z=1.5)
    req.workspace_parameters = ws

    req.goal_constraints.append(goal_constraints)
    return req


def _make_pose(x, y, z, quat):
    p = Pose()
    p.position.x = float(x)
    p.position.y = float(y)
    p.position.z = float(z)
    p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w = quat
    return p


def make_forward_tcp_pose(x, y, z):
    return _make_pose(x, y, z, FORWARD_QUAT)


# ---------------------------------------------------------------------------
# State machine node
# ---------------------------------------------------------------------------

class PickPlaceManagerNode(Node):
    TICK_PERIOD = 0.2

    def __init__(self):
        super().__init__('pick_place_manager_node')

        self.declare_parameter('debug_step', False)
        debug_param = self.get_parameter('debug_step').value
        if isinstance(debug_param, str):
            self._debug_step = debug_param.strip().lower() == 'true'
        else:
            self._debug_step = bool(debug_param)

        # Real-robot launches must wait for the URCap "External Control" program
        # to be running and connected to the reverse interface before the FSM
        # can send any motion. The launch file sets this False in fake-hardware
        # mode so home-sim does not hang on a topic that never publishes.
        self.declare_parameter('wait_for_robot_program', True)
        wait_param = self.get_parameter('wait_for_robot_program').value
        if isinstance(wait_param, str):
            self._wait_for_robot_program = wait_param.strip().lower() == 'true'
        else:
            self._wait_for_robot_program = bool(wait_param)
        self._robot_program_running = False

        self._step_event = threading.Event()
        self._planned_traj = None
        self._debug_phase = 'idle'
        if self._debug_step:
            self.get_logger().info(
                'DEBUG STEP MODE: press Enter to execute the displayed '
                'trajectory; press Enter again after each motion to plan the '
                'next one.'
            )
            threading.Thread(target=self._stdin_loop, daemon=True).start()

        self._action_group = MutuallyExclusiveCallbackGroup()
        self._sub_group = ReentrantCallbackGroup()

        self.state_pub = self.create_publisher(String, '/pick_place_state', 10)
        # Zimmer HRC-03: 2-wire tool DO control. fun=1 (FUN_SET_DIGITAL_OUT)
        # for all digital outputs; tool outputs use pin 16 (TOOL_DOUT0=open)
        # and pin 17 (TOOL_DOUT1=close). Never assert both simultaneously.
        self.gripper_io_client = self.create_client(
            SetIO, '/io_and_status_controller/set_io'
        )
        self._gripper_io_fun = 1
        # RViz-only visualisation: publish the same open/close intent to the
        # mock-backed gripper position controller so the URDF fingers move.
        self.gripper_pub = self.create_publisher(
            Float64MultiArray, '/gripper_controller/commands', 10
        )
        # Tells the planning_scene_manager which detected can is the active
        # pick (so it can be suppressed during approach and attached on grasp).
        self.current_target_pub = self.create_publisher(
            PoseStamped, '/current_pick_target', 1,
        )

        # Republish transformed detections so planning_scene_manager and
        # rviz_visualizer (both subscribers of /target_can_pose) see the cans
        # in base_link with the fixed z applied.
        self.target_pose_pub = self.create_publisher(
            CanDetectionArray, '/target_can_pose', 10,
        )

        # TF used to transform native_vision_node detections from
        # camera_optical_link into base_link.
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.create_subscription(
            CanDetectionArray, '/target_can_pose', self._on_detections, 10,
            callback_group=self._sub_group,
        )
        self.create_subscription(
            CanDetectionArray, '/front_detections', self._on_front_detections, 10,
            callback_group=self._sub_group,
        )
        self.create_subscription(
            String, '/order', self._on_command, 10,
            callback_group=self._sub_group,
        )
        self.create_subscription(
            Empty, '/clear_place_zone', self._on_clear_place_zone, 10,
            callback_group=self._sub_group,
        )
        self.create_subscription(
            Float32, '/human_proximity', self._on_proximity, 10,
            callback_group=self._sub_group,
        )
        self._latest_joint_state: JointState | None = None
        self.create_subscription(
            JointState, '/joint_states', self._on_joint_states, 10,
            callback_group=self._sub_group,
        )
        self.create_subscription(
            Bool, '/io_and_status_controller/robot_program_running',
            self._on_robot_program_state, 10,
            callback_group=self._sub_group,
        )

        self.move_group_client = ActionClient(
            self, MoveGroup, '/move_action', callback_group=self._action_group,
        )
        self.execute_client = ActionClient(
            self, ExecuteTrajectory, '/execute_trajectory',
            callback_group=self._action_group,
        )
        self.cartesian_client = self.create_client(
            GetCartesianPath, '/compute_cartesian_path',
            callback_group=self._action_group,
        )

        self.state = 'BOOT'
        self.last_state_published = None

        # Cycle inputs / matched data
        self.command_queue = []          # list[str] of remaining can classes to pick
        self.identified = []             # list[{'class': str, 'pos': (x,y,z)}]

        # /order debounce: a single logical order can arrive as multiple
        # back-to-back messages, one per type, e.g.
        #   "type: coke, amount: 1"
        #   "type: beer, amount: 1"
        # Collect them all and only start the cycle after no new order has
        # arrived for ORDER_DEBOUNCE_S seconds.
        self.ORDER_DEBOUNCE_S = 0.7
        self._order_debounce_timer = None

        # Per-pick state
        self.active_target_pos = None    # (x, y, z)
        self.active_target_class = None
        self.active_target_frame = 'base_link'

        # Place zone slot map (persists across cycles)
        self.place_slots_filled = [False] * NUM_PLACE_SLOTS
        self.active_place_slot = None    # slot index reserved for current pick

        # Human-proximity gating
        self._human_proximity = 1.0      # default: assume safe until told otherwise

        self._motion_mode = None
        self._goal_future = None
        self._result_future = None
        self._cartesian_call = None
        self._exec_goal_future = None
        self._exec_result_future = None
        self._motion_failed = False
        self._motion_started_at = None
        self._wait_until = None
        self._next_state_after_wait = None

        # Pre-computed canonical-branch IK for static poses.
        self.wait_forward_joints = ik_for_tcp(*WAIT_TCP, FORWARD_QUAT, _WAIT_FORWARD_SEED)
        if self.wait_forward_joints is None:
            self.get_logger().error(
                f'WAIT_TCP={WAIT_TCP} unreachable with FORWARD_QUAT — using seed as fallback.'
            )
            self.wait_forward_joints = _WAIT_FORWARD_SEED

        self.identify_joints = ik_for_tcp(*IDENTIFY_TCP, FORWARD_QUAT, self.wait_forward_joints)
        if self.identify_joints is None:
            self.get_logger().error(
                f'IDENTIFY_TCP={IDENTIFY_TCP} unreachable — IDENTIFY step will be skipped.'
            )
            self.identify_joints_rotated = None
        else:
            # Same TCP / branch as identify_joints, but with wrist_3 rotated
            # +90° about wrist_3_link Z so the camera frames the pick zone
            # from the side. Applied as a discrete FSM step *after* arriving
            # at the identify pose, then unwound *before* planning the
            # pregrasp so that pregrasp IK seeds from the canonical
            # FORWARD_QUAT branch.
            self.identify_joints_rotated = dict(self.identify_joints)
            self.identify_joints_rotated['wrist_3_joint'] = (
                self.identify_joints_rotated['wrist_3_joint']
                + IDENTIFY_WRIST3_OFFSET
            )

        # Per-pick chained IK results.
        self.q_pregrasp = None
        self.q_grasp = None
        self.q_lift = None
        self.q_retreat = None
        self.q_place = None

        self.create_timer(self.TICK_PERIOD, self._tick)
        self.get_logger().info('Pick-and-place manager (MoveIt2) started.')

    # ------------------------------------------------------------------
    # Debug step gate
    # ------------------------------------------------------------------

    def _stdin_loop(self):
        # When launched via ros2 launch, sys.stdin is a pipe not the terminal.
        # /dev/tty always refers to the controlling terminal directly.
        try:
            tty = open('/dev/tty', 'r')
        except OSError:
            tty = sys.stdin
        for _ in iter(tty.readline, ''):
            self._step_event.set()

    def _step_gate_open(self, prompt):
        if not self._debug_step:
            return True
        if self._step_event.is_set():
            self._step_event.clear()
            return True
        if not getattr(self, '_last_prompt', None) == prompt:
            self.get_logger().info(prompt)
            self._last_prompt = prompt
        return False

    # ------------------------------------------------------------------
    # Subscriptions
    # ------------------------------------------------------------------

    def _on_front_detections(self, msg):
        """Bridge native_vision_node -> /target_can_pose.

        native_vision_node publishes /front_detections in camera_optical_link
        with the depth-camera estimate of (x, y, z). For planning we need
        base_link coordinates and a known-good z (the camera's depth at close
        range is noisy and the grasp height is set by table geometry). This
        callback transforms each detection into base_link, replaces z with
        FIXED_CAN_Z, and republishes the result on /target_can_pose so the
        rest of the pipeline (this node's IDENTIFY handler, the planning
        scene, the RViz markers) sees a single canonical detection stream.

        Only republish while the robot is stationary in IDENTIFY_FROM_FRONT.
        Otherwise the wrist_3 rotation (and any other motion) would feed
        moving camera readings into the planning scene / RViz markers, making
        the can positions drift before they are locked in.
        """
        if not msg.detections:
            return
        if self.state != 'IDENTIFY_FROM_FRONT' or self._motion_mode is not None:
            return
        src_frame = msg.header.frame_id or CAMERA_OPTICAL_FRAME
        out = CanDetectionArray()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = 'base_link'
        for det in msg.detections:
            pt_in = PointStamped()
            pt_in.header.frame_id = src_frame
            pt_in.header.stamp = msg.header.stamp
            pt_in.point.x = float(det.position.x)
            pt_in.point.y = float(det.position.y)
            pt_in.point.z = float(det.position.z)
            try:
                pt_out = self.tf_buffer.transform(
                    pt_in, 'base_link', timeout=Duration(seconds=0.2)
                )
            except tf2_ros.TransformException as e:
                self.get_logger().warn(
                    f'TF {src_frame}->base_link failed: {e}; skipping detection.'
                )
                continue
            new_det = CanDetection()
            new_det.header = out.header
            new_det.id = det.id
            new_det.class_id = det.class_id
            new_det.confidence = det.confidence
            new_det.position.x = pt_out.point.x
            new_det.position.y = pt_out.point.y
            new_det.position.z = FIXED_CAN_Z
            new_det.source = 'front'
            out.detections.append(new_det)
        if out.detections:
            self.target_pose_pub.publish(out)

    def _on_detections(self, msg):
        """Receive a /target_can_pose message.

        Only acts in IDENTIFY_FROM_FRONT: every detection carries a
        class_id and a (x, y, z) position already in base_link with the
        fixed z applied (see _on_front_detections). We build the
        {class -> position} list directly and transition to
        ROTATE_BACK_FROM_IDENTIFY so the wrist returns to the canonical
        FORWARD_QUAT branch before pregrasp IK runs.
        """
        if not msg.detections:
            return
        if self.state != 'IDENTIFY_FROM_FRONT' or self._motion_mode is not None:
            return

        matched = []
        for det in msg.detections:
            cls = CAN_CLASS_NAMES.get(int(det.class_id))
            if not cls:
                continue
            matched.append({
                'class': cls,
                'pos': (float(det.position.x),
                        float(det.position.y),
                        float(det.position.z)),
            })
        if not matched:
            self.get_logger().info('  IDENTIFY: no detections with known class_id, ignoring')
            return
        self.identified = matched
        classes = sorted({m['class'] for m in matched})
        self.get_logger().info(
            f'Identified {len(matched)} can(s): classes={classes}'
        )
        self._enter('ROTATE_BACK_FROM_IDENTIFY')

    def _on_command(self, msg):
        """Handle a /order message.

        Expected payload: "type: <class>, amount: <N>"

        A single logical order can arrive as multiple back-to-back messages
        (one per type) — they are all queued together. The cycle starts only
        after no new /order has arrived for ORDER_DEBOUNCE_S seconds.
        """
        raw = (msg.data or '').strip()
        if not raw:
            self.get_logger().warn('Empty /order ignored.')
            return
        fields = {}
        for part in raw.split(','):
            if ':' not in part:
                continue
            k, v = part.split(':', 1)
            fields[k.strip().lower()] = v.strip()
        cls = fields.get('type', '').lower()
        if not cls or cls not in CAN_CLASS_IDS:
            self.get_logger().warn(
                f'/order: unknown or missing type {fields!r} (expected one of '
                f'{sorted(CAN_CLASS_IDS)}).'
            )
            return
        try:
            amount = int(fields.get('amount', '1'))
        except ValueError:
            self.get_logger().warn(f'/order: bad amount in {raw!r}.')
            return
        if amount <= 0:
            self.get_logger().warn(f'/order: non-positive amount in {raw!r}.')
            return
        if self.state != 'WAIT_FOR_COMMAND':
            self.get_logger().warn(
                f'/order received in state {self.state}; ignored '
                '(robot must be at WAIT_FOR_COMMAND).'
            )
            return
        self.command_queue.extend([cls] * amount)
        self.get_logger().info(
            f'Order line accepted: type={cls}, amount={amount}; '
            f'queue now {self.command_queue}'
        )
        # (re)start the debounce — every new /order pushes the start back so
        # all parts of the same logical order get collected first.
        if self._order_debounce_timer is not None:
            self._order_debounce_timer.cancel()
        self._order_debounce_timer = self.create_timer(
            self.ORDER_DEBOUNCE_S, self._start_cycle_after_debounce
        )

    def _start_cycle_after_debounce(self):
        if self._order_debounce_timer is not None:
            self._order_debounce_timer.cancel()
            self._order_debounce_timer = None
        if self.state != 'WAIT_FOR_COMMAND':
            return
        if not self.command_queue:
            return
        self.identified = []
        self.get_logger().info(
            f'Order complete — starting cycle. Full queue: {self.command_queue}'
        )
        self._enter('ORIENT_FORWARD')

    def _on_clear_place_zone(self, _msg):
        self.place_slots_filled = [False] * NUM_PLACE_SLOTS
        self.get_logger().info('Place zone cleared (all slots free).')

    def _on_proximity(self, msg):
        try:
            self._human_proximity = float(msg.data)
        except (TypeError, ValueError):
            return

    def _on_joint_states(self, msg: JointState):
        self._latest_joint_state = msg

    def _on_robot_program_state(self, msg: Bool):
        was_running = self._robot_program_running
        self._robot_program_running = bool(msg.data)
        if not was_running and self._robot_program_running:
            self.get_logger().info(
                'UR robot program is running and reverse interface is ready — '
                'starting motion FSM.'
            )

    def _current_arm_joints_dict(self):
        js = self._latest_joint_state
        if js is None:
            return None
        out = {}
        for name in ARM_JOINT_NAMES:
            if name not in js.name:
                return None
            out[name] = float(js.position[js.name.index(name)])
        return out

    # ------------------------------------------------------------------
    # Speed scaling
    # ------------------------------------------------------------------

    def _scaling_pair(self):
        if self._human_proximity < HUMAN_PROXIMITY_THRESHOLD:
            return (VEL_SCALING_NORMAL * SLOWDOWN_FACTOR,
                    ACC_SCALING_NORMAL * SLOWDOWN_FACTOR)
        return (VEL_SCALING_NORMAL, ACC_SCALING_NORMAL)

    # ------------------------------------------------------------------
    # Per-pick IK chain
    # ------------------------------------------------------------------

    def _compute_pick_ik(self, can_x, can_y, can_z, place_xyz):
        seed = self.identify_joints if self.identify_joints is not None else self.wait_forward_joints

        target_x = can_x - APPROACH_OFFSET_X
        target_y = can_y
        target_z = can_z + GRASP_Z_OFFSET
        self.get_logger().info(
            f'IK pregrasp target: x={target_x:.3f} y={target_y:.3f} z={target_z:.3f} '
            f'quat={FORWARD_QUAT}')
        q_pregrasp = ik_for_tcp(target_x, target_y, target_z, FORWARD_QUAT, seed)
        if q_pregrasp is None:
            self.get_logger().error('IK failed: pregrasp pose unreachable.')
            return False
        # FK back the IK solution and report the achieved TCP — confirms the
        # solver actually hit FORWARD_QUAT and the requested xyz.
        q_arr = np.array([q_pregrasp[n] for n in ARM_JOINT_NAMES])
        T = _fk_tcp(q_arr)
        self.get_logger().info(
            f'IK pregrasp achieved: pos=({T[0,3]:.3f},{T[1,3]:.3f},{T[2,3]:.3f}) '
            f'R col_z=({T[0,2]:.3f},{T[1,2]:.3f},{T[2,2]:.3f})')

        q_grasp = ik_for_tcp(can_x, can_y, can_z + GRASP_Z_OFFSET, FORWARD_QUAT, q_pregrasp)
        if q_grasp is None:
            self.get_logger().error('IK failed: grasp pose unreachable.')
            return False

        q_lift = ik_for_tcp(can_x, can_y, can_z + LIFT_Z + GRASP_Z_OFFSET, FORWARD_QUAT, q_grasp)
        if q_lift is None:
            self.get_logger().error('IK failed: lift pose unreachable.')
            return False

        px, py, pz = place_xyz
        # Seed the place IK chain with _PLACE_SEED — an elbow-folded branch
        # that keeps the upper arm clear of the table when reaching down to
        # low PLACE_TCP_Z. wait_forward_joints would land in the elbow-up
        # branch, which scrapes the table for PLACE_TCP_Z < 0.15 m.
        q_high = ik_for_tcp(px, py, pz + LIFT_Z + GRASP_Z_OFFSET, FORWARD_QUAT,
                            _PLACE_SEED)
        if q_high is None:
            self.get_logger().error('IK failed: pre-place pose unreachable.')
            return False
        q_place = ik_for_tcp(px, py, pz, FORWARD_QUAT, q_high)
        if q_place is None:
            self.get_logger().error('IK failed: place pose unreachable.')
            return False

        self.q_pregrasp = q_pregrasp
        self.q_grasp    = q_grasp
        self.q_lift     = q_lift
        self.q_place    = q_place
        self.q_retreat  = q_high  # high pre-place pose, reused for retreat
        return True

    # ------------------------------------------------------------------
    # Place slot helpers
    # ------------------------------------------------------------------

    def _next_free_slot(self):
        for i, filled in enumerate(self.place_slots_filled):
            if not filled:
                return i
        return None

    def _slot_xyz(self, slot_index):
        dx, dy = PLACE_SLOT_OFFSETS[slot_index]
        return (PLACE_ZONE_CENTER[0] + dx,
                PLACE_ZONE_CENTER[1] + dy,
                PLACE_TCP_Z)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _enter(self, new_state):
        self.state = new_state
        self._publish_state(new_state)

    def _publish_state(self, state):
        if state == self.last_state_published:
            return
        self.last_state_published = state
        self.state_pub.publish(String(data=state))
        self.get_logger().info(f'State -> {state}')

    def _publish_active_target(self):
        if self.active_target_pos is None:
            return
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.active_target_frame or 'base_link'
        msg.pose.position.x = float(self.active_target_pos[0])
        msg.pose.position.y = float(self.active_target_pos[1])
        msg.pose.position.z = float(self.active_target_pos[2])
        msg.pose.orientation.w = 1.0
        self.current_target_pub.publish(msg)

    def _now_s(self):
        return self.get_clock().now().nanoseconds / 1e9

    def _sleep(self, seconds, next_state):
        self._wait_until = self._now_s() + seconds
        self._next_state_after_wait = next_state
        self._enter('WAIT')

    def _send_gripper(self, value):
        # 2-wire control: TDO0=1 opens, TDO1=1 closes. Never assert both.
        close = value < GRIPPER_OPEN
        if not self.gripper_io_client.service_is_ready():
            self.get_logger().warn(
                'set_io service not ready — gripper command dropped.')
            return
        # Deactivate the opposing solenoid before activating the desired one
        # so there is never a brief window where both are high (= no-op).
        if close:
            pins_ordered = ((16, 0.0), (17, 1.0))  # kill open, then fire close
        else:
            pins_ordered = ((17, 0.0), (16, 1.0))  # kill close, then fire open
        for pin, state in pins_ordered:
            req = SetIO.Request()
            req.fun = self._gripper_io_fun
            req.pin = pin
            req.state = state
            self.gripper_io_client.call_async(req)
        # Mirror to RViz visualisation joints.
        msg = Float64MultiArray()
        msg.data = [float(value), float(value)]
        self.gripper_pub.publish(msg)

    # ------------------------------------------------------------------
    # Motion plumbing
    # ------------------------------------------------------------------

    def _send_joint_goal(self, joint_values):
        if not self.move_group_client.wait_for_server(timeout_sec=2.0):
            self.get_logger().error('move_action server unavailable.')
            return False

        vel, acc = self._scaling_pair()
        goal = MoveGroup.Goal()
        goal.request = make_motion_plan_request(
            make_joint_goal_constraints(joint_values),
            PLANNING_FRAME, PLANNING_GROUP,
            vel, acc,
        )
        goal.planning_options = PlanningOptions()
        # Always plan+execute in a single call. The previous debug-mode
        # plan_only path published a ghost trajectory that did not match the
        # actual executed motion (KDL branch flips during execution), so the
        # preview was misleading. The debug step gate still pauses between
        # motions — it just no longer shows the ghost.
        goal.planning_options.plan_only = False
        goal.planning_options.look_around = False
        goal.planning_options.replan = False
        goal.planning_options.planning_scene_diff.is_diff = True
        goal.planning_options.planning_scene_diff.robot_state.is_diff = True

        self._motion_mode = 'joint'
        self._motion_failed = False
        self._planned_traj = None
        self._debug_phase = 'idle'
        self._motion_started_at = self._now_s()
        self._goal_future = self.move_group_client.send_goal_async(goal)
        self._goal_future.add_done_callback(self._on_joint_goal_response)
        self._result_future = None
        return True

    def _on_joint_goal_response(self, future):
        handle = future.result()
        if handle is None or not handle.accepted:
            self.get_logger().error('MoveGroup goal rejected.')
            self._motion_failed = True
            return
        self._result_future = handle.get_result_async()

    def _send_cartesian_goal(self, pose, avoid_collisions=True):
        if not self.cartesian_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().error('compute_cartesian_path service unavailable.')
            return False

        vel, acc = self._scaling_pair()
        req = GetCartesianPath.Request()
        req.header.frame_id = PLANNING_FRAME
        req.group_name = PLANNING_GROUP
        req.link_name = EE_LINK
        req.waypoints = [pose]
        req.max_step = CARTESIAN_MAX_STEP
        req.jump_threshold = CARTESIAN_JUMP_THRESHOLD
        req.avoid_collisions = avoid_collisions
        if self._latest_joint_state is not None:
            req.start_state.joint_state = self._latest_joint_state
        # Newer MoveIt versions expose velocity scaling on the cartesian
        # service; older ones ignore unknown fields silently.
        try:
            req.max_velocity_scaling_factor = vel
            req.max_acceleration_scaling_factor = acc
        except AttributeError:
            pass

        self._motion_mode = 'cartesian'
        self._motion_failed = False
        self._exec_goal_future = None
        self._exec_result_future = None
        self._planned_traj = None
        self._debug_phase = 'idle'
        self._motion_started_at = self._now_s()
        self._cartesian_call = self.cartesian_client.call_async(req)
        self._cartesian_call.add_done_callback(self._on_cartesian_response)
        return True

    def _on_cartesian_response(self, future):
        res = future.result()
        if res is None:
            self.get_logger().error('Cartesian path service returned None.')
            self._motion_failed = True
            return
        if res.fraction < CARTESIAN_MIN_FRACTION:
            self.get_logger().error(
                f'Cartesian path incomplete: fraction={res.fraction:.3f} '
                f'(need {CARTESIAN_MIN_FRACTION:.1f}). Aborting.'
            )
            self._motion_failed = True
            return
        self.get_logger().info(
            f'Cartesian path OK (fraction={res.fraction:.3f}, '
            f'{len(res.solution.joint_trajectory.points)} pts).'
        )
        self._planned_traj = res.solution
        if not self._dispatch_cached_execute():
            return

    def _on_exec_goal_response(self, future):
        handle = future.result()
        if handle is None or not handle.accepted:
            self.get_logger().error('ExecuteTrajectory goal rejected.')
            self._motion_failed = True
            return
        self._exec_result_future = handle.get_result_async()

    def _move_done(self):
        if self._motion_failed:
            return True, False

        # Watchdog: bail out if a future never resolves (e.g. move_group's
        # action goal-response was dropped by DDS — known transient on first
        # call after launch).  Skip while the user is paused at the step gate
        # in debug mode.
        if (self._motion_started_at is not None
                and not self._debug_step
                and self._now_s() - self._motion_started_at > MOTION_WATCHDOG_S):
            self.get_logger().error(
                f'Motion watchdog timeout ({MOTION_WATCHDOG_S:.0f}s) in state '
                f'{self.state} — aborting motion.'
            )
            self._motion_failed = True
            return True, False

        if self._motion_mode == 'cartesian':
            if self._exec_result_future is None or not self._exec_result_future.done():
                return False, False
            result = self._exec_result_future.result()
            if result is None:
                return True, False
            err = result.result.error_code.val
            if err != 1:
                self.get_logger().error(
                    f'ExecuteTrajectory error_code={err} in state {self.state}.'
                )
            return True, err == 1

        if self._result_future is None or not self._result_future.done():
            return False, False
        result = self._result_future.result()
        if result is None:
            return True, False
        err = result.result.error_code.val
        if err != 1:
            self.get_logger().error(
                f'MoveGroup error_code={err} in state {self.state}.'
            )
        return True, err == 1

    def _normalize_trajectory_to_robot(self, traj):
        """Shift each joint's waypoints by multiples of 2π so the first
        waypoint matches the robot's current joint position within ±π.
        This prevents PATH_TOLERANCE_VIOLATED when the UR controller reports
        a joint in a different revolution than the planner used."""
        js = self._latest_joint_state
        if js is None or not traj.joint_trajectory.joint_names:
            return traj
        import copy, math
        traj = copy.deepcopy(traj)
        jt = traj.joint_trajectory
        for ji, name in enumerate(jt.joint_names):
            if name not in js.name:
                continue
            robot_pos = js.position[js.name.index(name)]
            for pi, pt in enumerate(jt.points):
                if ji >= len(pt.positions):
                    continue
                prev = robot_pos if pi == 0 else jt.points[pi - 1].positions[ji]
                diff = pt.positions[ji] - prev
                # wrap diff into (-π, π]
                diff = (diff + math.pi) % (2 * math.pi) - math.pi
                positions = list(pt.positions)
                positions[ji] = prev + diff
                jt.points[pi].positions = tuple(positions)
        return traj

    def _dispatch_cached_execute(self):
        if self._planned_traj is None:
            self.get_logger().error('No cached trajectory to execute.')
            self._motion_failed = True
            return False
        if not self.execute_client.wait_for_server(timeout_sec=2.0):
            self.get_logger().error('execute_trajectory action unavailable.')
            self._motion_failed = True
            return False
        goal = ExecuteTrajectory.Goal()
        goal.trajectory = self._normalize_trajectory_to_robot(self._planned_traj)
        self._exec_goal_future = self.execute_client.send_goal_async(goal)
        self._exec_goal_future.add_done_callback(self._on_exec_goal_response)
        return True

    def _reset_move(self):
        self._motion_mode = None
        self._motion_failed = False
        self._motion_started_at = None
        self._goal_future = None
        self._result_future = None
        self._cartesian_call = None
        self._exec_goal_future = None
        self._exec_result_future = None
        self._planned_traj = None
        self._debug_phase = 'idle'
        self._last_prompt = None

    # ------------------------------------------------------------------
    # Unified motion dispatcher
    # ------------------------------------------------------------------

    def _handle_joint_move(self, joint_values, next_on_success, fail_state='WAIT_FOR_COMMAND'):
        if self._motion_mode is None:
            if not self._send_joint_goal(joint_values):
                self._enter(fail_state)
            return
        self._await_done(next_on_success, fail_state)

    def _handle_cartesian_move(self, pose, next_on_success, avoid_collisions=True,
                               fail_state='WAIT_FOR_COMMAND'):
        if self._motion_mode is None:
            if not self._send_cartesian_goal(pose, avoid_collisions=avoid_collisions):
                self._enter(fail_state)
            return
        self._await_done(next_on_success, fail_state)

    def _await_done(self, next_on_success, fail_state):
        done, success = self._move_done()
        if not done:
            return
        if success and self._debug_step and self._debug_phase != 'exec_done':
            self._debug_phase = 'exec_done'
        if success and self._debug_step and self._debug_phase == 'exec_done':
            if not self._step_gate_open(
                'Motion complete. Press Enter to plan the next segment.'
            ):
                return
        self._reset_move()
        if success:
            self._enter(next_on_success)
        else:
            self.get_logger().error(
                f'Motion failed in state {self.state}; aborting cycle.'
            )
            self._abort_cycle()
            self._enter(fail_state)

    def _abort_cycle(self):
        self.command_queue = []
        self.active_target_pos = None
        self.active_target_class = None
        if self.active_place_slot is not None:
            # The reservation was never used; release it so a future pick
            # doesn't skip a slot.
            self.active_place_slot = None

    # ------------------------------------------------------------------
    # State step implementations
    # ------------------------------------------------------------------

    def move_to_waiting_pose(self):
        self._handle_joint_move(self.wait_forward_joints, next_on_success='WAIT_FOR_COMMAND')

    def set_forward_orientation(self):
        self._handle_joint_move(self.wait_forward_joints, next_on_success='MOVE_TO_IDENTIFY')

    def move_to_identify_pose(self):
        # Step 1 of identify: arrive at IDENTIFY_TCP in the FORWARD_QUAT branch
        # (canonical wrist). Wrist rotation happens in the next state so the
        # camera-side framing only affects the identify capture itself.
        target = self.identify_joints if self.identify_joints is not None else self.wait_forward_joints
        self._handle_joint_move(target, next_on_success='ROTATE_FOR_IDENTIFY')

    def rotate_for_identify(self):
        # Step 2 of identify: rotate wrist_3 by +90° in place. After this the
        # FSM holds in IDENTIFY_FROM_FRONT until a /front_detections message
        # is consumed by _on_detections.
        if self.identify_joints_rotated is None:
            self._enter('IDENTIFY_FROM_FRONT')
            return
        self._handle_joint_move(
            self.identify_joints_rotated, next_on_success='IDENTIFY_FROM_FRONT'
        )

    def rotate_back_from_identify(self):
        # Step 4 of identify: unwind the +90° wrist_3 rotation so subsequent
        # pregrasp IK seeds from the canonical FORWARD_QUAT branch.
        target = self.identify_joints if self.identify_joints is not None else self.wait_forward_joints
        self._handle_joint_move(target, next_on_success='NEXT_TARGET')

    def plan_to_pregrasp(self):
        self._handle_joint_move(self.q_pregrasp, next_on_success='CARTESIAN_APPROACH')

    def joint_approach(self):
        # Joint-space move to the pre-computed q_grasp. q_grasp shares the
        # same wrist branch as q_pregrasp (seeded chain), so the path stays
        # branch-consistent and avoids the KDL flip that broke the Cartesian
        # planner. Over 8 cm the TCP path is visually near-straight.
        if self._motion_mode is None:
            self._send_gripper(GRIPPER_OPEN)
        self._handle_joint_move(self.q_grasp, next_on_success='GRASP')

    def grasp(self):
        self._send_gripper(GRIPPER_GRIP)
        self._sleep(1.0, 'CARTESIAN_LIFT')

    def joint_lift(self):
        self._handle_joint_move(self.q_lift, next_on_success='PLAN_TO_PLACE')

    def plan_to_place(self):
        self._handle_joint_move(self.q_place, next_on_success='RELEASE')

    def joint_retreat(self):
        # Joint-space move to q_retreat (the high pre-place IK solution).
        # Same reasoning as joint_approach: a Cartesian planner here flips
        # KDL branches mid-path and lands the robot in a twisted
        # configuration that can't plan back to the wait pose afterwards.
        if self.q_retreat is None:
            self.get_logger().warn('No q_retreat available — aborting.')
            self._abort_cycle()
            self._enter('ORIENT_DOWN_AT_WAIT')
            return
        self._handle_joint_move(self.q_retreat, next_on_success='AFTER_RETREAT')

    def orient_down_at_wait(self):
        self._handle_joint_move(self.wait_forward_joints, next_on_success='WAIT_FOR_COMMAND')

    # ------------------------------------------------------------------
    # State machine tick
    # ------------------------------------------------------------------

    def _tick(self):
        s = self.state

        if s == 'BOOT':
            self._enter('WAIT_FOR_ROBOT')
            return

        if s == 'WAIT_FOR_ROBOT':
            if self._wait_for_robot_program and not self._robot_program_running:
                return
            self._send_gripper(GRIPPER_OPEN)
            self._enter('MOVE_TO_WAIT')
            return

        if s == 'MOVE_TO_WAIT':
            self.move_to_waiting_pose()
            return

        if s == 'WAIT_FOR_COMMAND':
            return

        if s == 'WAIT':
            if self._now_s() >= self._wait_until:
                self._enter(self._next_state_after_wait)
            return

        if s == 'ORIENT_FORWARD':
            self.set_forward_orientation()
            return

        if s == 'MOVE_TO_IDENTIFY':
            self.move_to_identify_pose()
            return

        if s == 'ROTATE_FOR_IDENTIFY':
            self.rotate_for_identify()
            return

        if s == 'IDENTIFY_FROM_FRONT':
            # Held here until _on_detections consumes a source="front" message.
            return

        if s == 'ROTATE_BACK_FROM_IDENTIFY':
            self.rotate_back_from_identify()
            return

        if s == 'NEXT_TARGET':
            self._select_next_target()
            return

        if s == 'PLAN_TO_PREGRASP':
            self.plan_to_pregrasp()
            return

        if s == 'CARTESIAN_APPROACH':
            self.joint_approach()
            return

        if s == 'GRASP':
            self.grasp()
            return

        if s == 'CARTESIAN_LIFT':
            self.joint_lift()
            return

        if s == 'PLAN_TO_PLACE':
            self.plan_to_place()
            return

        if s == 'RELEASE':
            self._send_gripper(GRIPPER_OPEN)
            self._mark_active_slot_filled()
            self._sleep(1.0, 'CARTESIAN_RETREAT')
            return

        if s == 'CARTESIAN_RETREAT':
            self.joint_retreat()
            return

        if s == 'AFTER_RETREAT':
            # One pick complete — clear per-pick state and loop.
            self.active_target_pos = None
            self.active_target_class = None
            self.active_place_slot = None
            self._enter('NEXT_TARGET')
            return

        if s == 'ALL_DONE':
            self._enter('ORIENT_DOWN_AT_WAIT')
            return

        if s == 'ORIENT_DOWN_AT_WAIT':
            self.orient_down_at_wait()
            return

    # ------------------------------------------------------------------
    # Per-pick selection
    # ------------------------------------------------------------------

    def _select_next_target(self):
        if not self.command_queue:
            self.get_logger().info('Command queue empty — cycle complete.')
            self._enter('ALL_DONE')
            return

        next_class = self.command_queue[0]
        match = next((m for m in self.identified if m['class'] == next_class), None)
        if match is None:
            self.get_logger().error(
                f'No identified can of class "{next_class}" — skipping.'
            )
            self.command_queue.pop(0)
            return  # stay in NEXT_TARGET; tick will re-enter

        slot = self._next_free_slot()
        if slot is None:
            self.get_logger().error(
                'Place zone full — aborting remaining picks. '
                'Send /clear_place_zone to reset.'
            )
            self.command_queue = []
            self._enter('ALL_DONE')
            return

        place_xyz = self._slot_xyz(slot)
        cx, cy, cz = match['pos']
        if not self._compute_pick_ik(cx, cy, cz, place_xyz):
            self.get_logger().error(
                f'IK failed for can "{next_class}" at ({cx:.3f},{cy:.3f},{cz:.3f}) '
                f'-> slot {slot} {place_xyz}. Skipping this can.'
            )
            self.command_queue.pop(0)
            self.identified = [m for m in self.identified if m is not match]
            return

        # Reserve the slot (only marked filled after RELEASE).
        self.active_place_slot = slot
        self.active_target_pos = (cx, cy, cz)
        self.active_target_class = next_class
        self.active_target_frame = 'base_link'
        # Once committed, drop from queue + identified so the next pick
        # picks a different physical can if multiple of same class.
        self.command_queue.pop(0)
        self.identified = [m for m in self.identified if m is not match]

        self._publish_active_target()
        self.get_logger().info(
            f'Next pick: class="{next_class}" pos=({cx:.3f},{cy:.3f},{cz:.3f}) '
            f'-> slot {slot} ({place_xyz[0]:.3f},{place_xyz[1]:.3f})'
        )
        self._enter('PLAN_TO_PREGRASP')

    def _mark_active_slot_filled(self):
        if self.active_place_slot is None:
            return
        self.place_slots_filled[self.active_place_slot] = True
        self.get_logger().info(
            f'Slot {self.active_place_slot} marked filled. '
            f'Slots: {self.place_slots_filled}'
        )


def main(args=None):
    rclpy.init(args=args)
    node = PickPlaceManagerNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    executor.shutdown()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
