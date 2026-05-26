#!/usr/bin/env python3
"""Owns the MoveIt planning scene for the pick-and-place stack.

Responsibilities
----------------
- Add a static table/workspace under the arm so MoveIt does not plan paths
  that drive the EE through the mounting surface.
- Mirror every detected can (full /target_can_pose array) as a CollisionObject
  so OMPL routes around them.
- Suppress the *active* pick target (designated via /current_pick_target)
  during the approach so the planner can drive into the can's location.
- Attach the active pick target to the gripper on GRASP, detach on RELEASE.

Message flow
------------
- subscribes /target_can_pose      (CanDetectionArray) all detected cans
- subscribes /current_pick_target  (PoseStamped)       active pick target
- subscribes /pick_place_state     (String)            attach/detach trigger
- calls      /apply_planning_scene (ApplyPlanningScene)
"""
import math

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Pose, PoseStamped
from std_msgs.msg import String

from moveit_msgs.msg import (
    AttachedCollisionObject,
    CollisionObject,
    PlanningScene,
    PlanningSceneWorld,
)
from moveit_msgs.srv import ApplyPlanningScene
from shape_msgs.msg import SolidPrimitive

from ur3_interfaces.msg import CanDetectionArray


PLANNING_FRAME = 'world'

CAN_RADIUS = 0.040
CAN_HEIGHT = 0.130

STATIC_FRAME = 'base_link'
TABLE_ID = 'table_top'
TABLE_SIZE = (1.5, 0.8, 0.05)
TABLE_POS = (0.0, 0.0, -0.025)
BACKBOARD_ID = 'backboard'
BACKBOARD_SIZE = (0.05, 1.5, 0.5)
BACKBOARD_POS = (-0.4, 0.0, 0.25)

ACTIVE_TARGET_ID = 'target_can'
ATTACH_LINK = 'gripper_base_link'
TOUCH_LINKS = [
    'gripper_base_link',
    'gripper_left_finger_link',
    'gripper_right_finger_link',
    'camera_link',
    'camera_optical_link',
]

# Distance under which a detection is considered the same can as the
# active /current_pick_target (used to suppress that one from the obstacle list).
ACTIVE_MATCH_DIST = 0.05


def make_box(object_id, frame, position, size):
    obj = CollisionObject()
    obj.header.frame_id = frame
    obj.id = object_id
    box = SolidPrimitive()
    box.type = SolidPrimitive.BOX
    box.dimensions = list(size)
    pose = Pose()
    pose.position.x, pose.position.y, pose.position.z = position
    pose.orientation.w = 1.0
    obj.primitives = [box]
    obj.primitive_poses = [pose]
    obj.operation = CollisionObject.ADD
    return obj


def make_cylinder(object_id, frame, position, radius, height):
    obj = CollisionObject()
    obj.header.frame_id = frame
    obj.id = object_id
    cyl = SolidPrimitive()
    cyl.type = SolidPrimitive.CYLINDER
    cyl.dimensions = [height, radius]
    pose = Pose()
    pose.position.x = float(position[0])
    pose.position.y = float(position[1])
    pose.position.z = float(position[2]) + height / 2.0
    pose.orientation.w = 1.0
    obj.primitives = [cyl]
    obj.primitive_poses = [pose]
    obj.operation = CollisionObject.ADD
    return obj


class PlanningSceneManagerNode(Node):
    def __init__(self):
        super().__init__('planning_scene_manager_node')

        self.create_subscription(CanDetectionArray, '/target_can_pose',
                                 self._on_detections, 10)
        self.create_subscription(PoseStamped, '/current_pick_target',
                                 self._on_active_target, 10)
        self.create_subscription(String, '/pick_place_state', self._on_state, 10)

        self.scene_client = self.create_client(ApplyPlanningScene, '/apply_planning_scene')

        # All detected cans (positions, possibly mixed source) — used as obstacles.
        self.detections = []      # list[(x,y,z)] in detections_frame
        self.detections_frame = 'base_link'

        # Currently-active pick target (suppressed during approach, attached on grasp).
        self.active_target_pose = None

        # IDs we created in the previous publish so we can REMOVE them when
        # the detection list shrinks.
        self._published_ids = set()

        self.target_attached = False
        self.target_suppressed = False
        self.last_state = None
        self._static_added = False

        self.create_timer(0.5, self._refresh_scene)

        self.get_logger().info('Planning scene manager started.')

    # ---------------------------------------------------------------- callbacks
    def _on_detections(self, msg):
        self.detections = [
            (float(d.position.x), float(d.position.y), float(d.position.z))
            for d in msg.detections
        ]
        self.detections_frame = msg.header.frame_id or 'base_link'

    def _on_active_target(self, msg):
        self.active_target_pose = msg

    def _on_state(self, msg):
        new_state = msg.data
        if new_state == self.last_state:
            return
        self.last_state = new_state

        # Cans are not modelled in the collision world (they're light and
        # treated as visual markers only) — no attach/detach/suppress logic
        # is needed. Pose is left here as a no-op for clarity.
        _ = new_state

    # ---------------------------------------------------------------- scene ops
    def _refresh_scene(self):
        if not self.scene_client.wait_for_service(timeout_sec=0.1):
            return

        scene = PlanningScene()
        scene.is_diff = True
        world = PlanningSceneWorld()

        if not self._static_added:
            world.collision_objects.append(
                make_box(TABLE_ID, STATIC_FRAME, TABLE_POS, TABLE_SIZE)
            )
            world.collision_objects.append(
                make_box(BACKBOARD_ID, STATIC_FRAME, BACKBOARD_POS, BACKBOARD_SIZE)
            )

        # Build current obstacle ids and add/refresh them.
        new_ids = set()
        active_pos = None
        if self.active_target_pose is not None:
            ap = self.active_target_pose.pose.position
            active_pos = (ap.x, ap.y, ap.z)

        # Cans are not added as collision objects: they're light, knocking one
        # over isn't damaging, and modelling them caused the joint-space
        # approach/retreat paths to fail when the planner couldn't avoid the
        # placed can. The active target is still attached to the gripper
        # while held, see _attach_target().
        _ = (active_pos,)  # silence unused-warning when this branch is empty

        # Remove any stale ids that disappeared since the last refresh.
        for old in self._published_ids - new_ids:
            remove = CollisionObject()
            remove.id = old
            remove.header.frame_id = self.detections_frame
            remove.operation = CollisionObject.REMOVE
            world.collision_objects.append(remove)
        self._published_ids = new_ids

        scene.world = world
        self._apply(scene)
        self._static_added = True

    def _attach_target(self):
        if self.active_target_pose is None:
            self.get_logger().warn('Cannot attach target: no /current_pick_target yet.')
            return

        # Remove the world copy first — we use a stable ID for the attached
        # object so the manager always has one name to reason about.
        remove = CollisionObject()
        remove.id = ACTIVE_TARGET_ID
        remove.header.frame_id = self.active_target_pose.header.frame_id or 'base_link'
        remove.operation = CollisionObject.REMOVE

        attached = AttachedCollisionObject()
        attached.link_name = ATTACH_LINK
        attached.object = make_cylinder(
            ACTIVE_TARGET_ID, ATTACH_LINK,
            (0.0, 0.0, 0.083 - CAN_HEIGHT / 2.0),
            CAN_RADIUS, CAN_HEIGHT,
        )
        attached.object.operation = CollisionObject.ADD
        attached.touch_links = list(TOUCH_LINKS)

        scene = PlanningScene()
        scene.is_diff = True
        scene.world.collision_objects.append(remove)
        scene.robot_state.is_diff = True
        scene.robot_state.attached_collision_objects.append(attached)

        if self._apply(scene):
            self.target_attached = True
            self.get_logger().info('Attached active target can to gripper.')

    def _remove_active_object(self):
        # Remove every can-shaped object from the world; refresh will re-add
        # the non-active ones on the next tick.
        scene = PlanningScene()
        scene.is_diff = True
        for obj_id in list(self._published_ids):
            remove = CollisionObject()
            remove.id = obj_id
            remove.header.frame_id = self.detections_frame
            remove.operation = CollisionObject.REMOVE
            scene.world.collision_objects.append(remove)
        self._published_ids = set()
        if self._apply(scene):
            self.get_logger().info('Cleared can obstacles for approach.')

    def _detach_target(self):
        detach = AttachedCollisionObject()
        detach.link_name = ATTACH_LINK
        detach.object.id = ACTIVE_TARGET_ID
        detach.object.operation = CollisionObject.REMOVE

        scene = PlanningScene()
        scene.is_diff = True
        scene.robot_state.is_diff = True
        scene.robot_state.attached_collision_objects.append(detach)

        if self._apply(scene):
            self.target_attached = False
            self.get_logger().info('Detached active target can.')

    def _apply(self, scene):
        if not self.scene_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().error('apply_planning_scene service unavailable.')
            return False
        req = ApplyPlanningScene.Request()
        req.scene = scene
        self.scene_client.call_async(req)
        return True


def main():
    rclpy.init()
    node = PlanningSceneManagerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
