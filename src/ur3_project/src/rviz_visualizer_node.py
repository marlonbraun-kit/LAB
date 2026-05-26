#!/usr/bin/env python3
import math
import numpy as np

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Point, PoseStamped, TransformStamped
from std_msgs.msg import ColorRGBA
from trajectory_msgs.msg import JointTrajectory
from visualization_msgs.msg import Marker, MarkerArray

import tf2_ros

from ur3_interfaces.msg import CanDetectionArray


# UR3e nominal DH (ur_description/config/ur3e/default_kinematics.yaml).
# Must match pick_place_manager_node.py and the URDF ur_type.
UR3_DH = [
    (0.0,       0.15185, math.pi / 2.0),
    (-0.24355,  0.0,     0.0),
    (-0.2132,   0.0,     0.0),
    (0.0,       0.13105, math.pi / 2.0),
    (0.0,       0.08535, -math.pi / 2.0),
    (0.0,       0.0921,  0.0),
]

PICK_ZONE = ( 0.2, -0.3, 0.0)
PLACE_ZONE = (0.2,  0.3, 0.0)
ZONE_SIZE = (0.3, 0.3, 0.001)

# Match pick_place_manager_node's PLACE_SLOT_OFFSETS layout exactly so the
# visualised slots line up with where the robot actually places cans.
PLACE_GRID_SPACING = 0.12
PLACE_SLOT_OFFSETS = [
    (-PLACE_GRID_SPACING / 2.0, -PLACE_GRID_SPACING / 2.0),  # back-left
    (-PLACE_GRID_SPACING / 2.0, +PLACE_GRID_SPACING / 2.0),  # back-right
    (+PLACE_GRID_SPACING / 2.0, -PLACE_GRID_SPACING / 2.0),  # front-left
    (+PLACE_GRID_SPACING / 2.0, +PLACE_GRID_SPACING / 2.0),  # front-right
]
SLOT_MARKER_SIZE = (0.08, 0.08, 0.001)

APPROACH_OFFSET_X = 0.1
BASE_FRAME = 'base_link'

RZ_FIX = np.array([
    [-1, 0, 0],
    [ 0,-1, 0],
    [ 0, 0, 1]
])


def dh_transform(a, d, alpha, theta):
    ct, st = math.cos(theta), math.sin(theta)
    ca, sa = math.cos(alpha), math.sin(alpha)
    return np.array([
        [ct, -st * ca,  st * sa, a * ct],
        [st,  ct * ca, -ct * sa, a * st],
        [0.0, sa,       ca,      d],
        [0.0, 0.0,      0.0,     1.0],
    ])


def fk(q):
    T = np.eye(4)
    for (a, d, alpha), theta in zip(UR3_DH, q):
        T = T @ dh_transform(a, d, alpha, theta)

    T[:3, 3] = RZ_FIX @ T[:3, 3]
    T[:3, :3] = RZ_FIX @ T[:3, :3]

    return T


def rgba(r, g, b, a=1.0):
    c = ColorRGBA()
    c.r, c.g, c.b, c.a = float(r), float(g), float(b), float(a)
    return c


class RvizVisualizerNode(Node):
    def __init__(self):
        super().__init__('rviz_visualizer_node')

        self.marker_pub = self.create_publisher(MarkerArray, '/visualization_markers', 10)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        self.create_subscription(CanDetectionArray, '/target_can_pose',
                                 self._on_detections, 10)
        self.create_subscription(PoseStamped, '/current_pick_target',
                                 self._on_active_target, 10)
        self.create_subscription(
            JointTrajectory,
            '/joint_trajectory_controller/joint_trajectory',
            self._on_trajectory,
            10,
        )

        self.latest_detections = []
        self.detections_frame = BASE_FRAME
        self.active_target = None
        self.latest_trajectory_points = []

        self.create_timer(0.2, self._publish_all)

    def _on_detections(self, msg):
        self.detections_frame = msg.header.frame_id or BASE_FRAME
        self.latest_detections = list(msg.detections)

    def _on_active_target(self, msg):
        self.active_target = msg

    def _on_trajectory(self, msg):
        pts = []
        for pt in msg.points:
            if len(pt.positions) < 6:
                continue
            T = fk(np.asarray(pt.positions[:6], dtype=float))
            p = Point()
            p.x, p.y, p.z = T[:3, 3]
            pts.append(p)
        self.latest_trajectory_points = pts

    def _publish_all(self):
        now = self.get_clock().now().to_msg()

        markers = MarkerArray()
        markers.markers.append(self._zone_marker(0, 'pick_zone', PICK_ZONE, (0.0, 1.0, 0.0, 1), now))
        markers.markers.append(self._zone_marker(1, 'place_zone', PLACE_ZONE, (0.0, 0.2, 1.0, 1), now))
        markers.markers.append(self._table_marker(now))
        markers.markers.append(self._backboard_marker(now))

        for i, (dx, dy) in enumerate(PLACE_SLOT_OFFSETS):
            markers.markers.append(self._slot_marker(i, dx, dy, now))

        for i, det in enumerate(self.latest_detections):
            markers.markers.append(self._can_marker(i, det, now))

        if self.latest_trajectory_points:
            markers.markers.append(self._trajectory_marker(now))

        self.marker_pub.publish(markers)
        self._broadcast_tfs(now)

    def _zone_marker(self, mid, ns, position, color, stamp):
        m = Marker()
        m.header.frame_id = BASE_FRAME
        m.header.stamp = stamp
        m.ns = ns
        m.id = mid
        m.type = Marker.CUBE
        m.action = Marker.ADD
        m.pose.position.x, m.pose.position.y, m.pose.position.z = position
        m.pose.orientation.w = 1.0
        m.scale.x, m.scale.y, m.scale.z = ZONE_SIZE
        m.color = rgba(*color)
        return m

    def _slot_marker(self, idx, dx, dy, stamp):
        m = Marker()
        m.header.frame_id = BASE_FRAME
        m.header.stamp = stamp
        m.ns = 'place_slots'
        m.id = 100 + idx
        m.type = Marker.CUBE
        m.action = Marker.ADD
        m.pose.position.x = PLACE_ZONE[0] + dx
        m.pose.position.y = PLACE_ZONE[1] + dy
        m.pose.position.z = 0.001
        m.pose.orientation.w = 1.0
        m.scale.x, m.scale.y, m.scale.z = SLOT_MARKER_SIZE
        m.color = rgba(0.0, 0.4, 1.0, 0.6)
        return m

    def _table_marker(self, stamp):
        m = Marker()
        m.header.frame_id = BASE_FRAME
        m.header.stamp = stamp
        m.ns = 'table'
        m.id = 10
        m.type = Marker.CUBE
        m.action = Marker.ADD
        m.pose.position.x, m.pose.position.y, m.pose.position.z = 0.0, 0.0, -0.025
        m.pose.orientation.w = 1.0
        m.scale.x, m.scale.y, m.scale.z = 0.8, 1.5, -0.05
        m.color = rgba(1.0, 1.0, 1.0, 1.0)
        return m

    def _backboard_marker(self, stamp):
        m = Marker()
        m.header.frame_id = BASE_FRAME
        m.header.stamp = stamp
        m.ns = 'backboard'
        m.id = 11
        m.type = Marker.CUBE
        m.pose.position.x = -0.4
        m.pose.position.y = 0.0
        m.pose.position.z = 0.25
        m.pose.orientation.w = 1.0
        m.scale.x, m.scale.y, m.scale.z = 0.05, 1.5, 0.5
        m.color = rgba(1.0, 1.0, 1.0, 1.0)
        return m

    def _can_marker(self, idx, det, stamp):
        m = Marker()
        m.header.frame_id = self.detections_frame
        m.header.stamp = stamp
        m.ns = 'cans'
        m.id = 40 + idx
        m.type = Marker.CYLINDER
        m.pose.position = det.position
        m.pose.orientation.w = 1.0
        m.scale.x = 0.06
        m.scale.y = 0.06
        m.scale.z = 0.12
        # Different shade per source so top vs front scans are visible.
        src = (det.source or '').lower()
        if src.startswith('front'):
            m.color = rgba(0.9, 0.4, 0.1, 0.7)
        else:
            m.color = rgba(0.9, 0.1, 0.1, 0.7)
        return m

    def _trajectory_marker(self, stamp):
        m = Marker()
        m.header.frame_id = BASE_FRAME
        m.header.stamp = stamp
        m.ns = 'trajectory'
        m.id = 50
        m.type = Marker.LINE_STRIP
        m.scale.x = 0.01
        m.color = rgba(1.0, 1.0, 0.0, 0.9)
        m.points = list(self.latest_trajectory_points)
        return m

    def _broadcast_tfs(self, stamp):
        tfs = [
            self._make_tf('pick_zone', PICK_ZONE, stamp),
            self._make_tf('place_zone', PLACE_ZONE, stamp),
        ]

        if self.active_target is not None:
            p = self.active_target.pose.position
            tfs.append(self._make_tf('can_active', (p.x, p.y, p.z), stamp))
            tfs.append(self._make_tf(
                'approach_point',
                (p.x - APPROACH_OFFSET_X, p.y, p.z),
                stamp,
            ))

        self.tf_broadcaster.sendTransform(tfs)

    def _make_tf(self, child, xyz, stamp):
        t = TransformStamped()
        t.header.stamp = stamp
        t.header.frame_id = BASE_FRAME
        t.child_frame_id = child
        t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = map(float, xyz)
        t.transform.rotation.w = 1.0
        return t


def main():
    rclpy.init()
    node = RvizVisualizerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
