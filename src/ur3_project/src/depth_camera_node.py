#!/usr/bin/env python3
"""Simulated camera source for at-home testing.

Publishes synthetic CanDetectionArray messages on /target_can_pose so the
manager state machine can run end-to-end without the real camera/recognition
nodes.  The publisher alternates between a "top" scan (positions only — what
the localisation node will publish) and a "front" scan (positions plus
class_name — what the identification node will publish).

For real-robot runs this node is *not* started — see pick_place_moveit.launch.py.
The teammates' camera nodes will publish the real messages instead.

Override behaviour:
  - parameters fake_can_*_x/y/z and fake_can_*_class set the simulated cans
  - or just `ros2 topic pub ... /target_can_pose ur3_interfaces/msg/...`
    yourself and ignore this node
"""
import numpy as np

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Point
from sensor_msgs.msg import Image
from std_msgs.msg import Header

from ur3_interfaces.msg import CanDetection, CanDetectionArray


class DepthCameraNode(Node):
    IMAGE_W = 160
    IMAGE_H = 120

    # Default fake cans laid out inside the pickup zone (around 0.30, 0.20).
    DEFAULT_CANS = [
        ('coke',  (0.30,  0.20, 0.06)),
        ('mahou', (0.36,  0.16, 0.06)),
        ('fanta', (0.24,  0.24, 0.06)),
    ]

    def __init__(self):
        super().__init__('depth_camera_node')

        # Each entry can be overridden via parameters fake_can_<i>_{class,x,y,z}.
        self._cans = []
        for i, (cls, (x, y, z)) in enumerate(self.DEFAULT_CANS):
            self.declare_parameter(f'fake_can_{i}_class', cls)
            self.declare_parameter(f'fake_can_{i}_x', float(x))
            self.declare_parameter(f'fake_can_{i}_y', float(y))
            self.declare_parameter(f'fake_can_{i}_z', float(z))
            self._cans.append(i)

        self.declare_parameter('publish_rate_hz', 1.0)
        self.declare_parameter('frame_id', 'base_link')

        self.detection_pub = self.create_publisher(
            CanDetectionArray, '/target_can_pose', 10
        )
        self.image_pub = self.create_publisher(Image, '/camera/depth/image_raw', 10)

        rate = float(self.get_parameter('publish_rate_hz').value)
        # Two phases per period: top, then front.
        self._phase = 'top'
        self.create_timer(1.0 / max(rate, 0.1), self._publish)

        self.get_logger().info(
            'Fake depth/camera source started '
            f'({len(self._cans)} cans, alternating top/front scans).'
        )

    def _build_detections(self, source):
        frame_id = str(self.get_parameter('frame_id').value)
        msg = CanDetectionArray()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = frame_id

        for i in self._cans:
            cls = str(self.get_parameter(f'fake_can_{i}_class').value)
            x = float(self.get_parameter(f'fake_can_{i}_x').value)
            y = float(self.get_parameter(f'fake_can_{i}_y').value)
            z = float(self.get_parameter(f'fake_can_{i}_z').value)
            det = CanDetection()
            det.header = msg.header
            det.id = f'fake_{i}'
            det.class_name = cls if source == 'front' else ''
            det.confidence = 1.0
            det.position = Point(x=x, y=y, z=z)
            det.source = source
            msg.detections.append(det)

        return msg, frame_id

    def _publish(self):
        msg, frame_id = self._build_detections(self._phase)
        self.detection_pub.publish(msg)
        self.image_pub.publish(self._make_depth_image(frame_id))
        # Alternate every tick.
        self._phase = 'front' if self._phase == 'top' else 'top'

    def _make_depth_image(self, frame_id):
        rows = np.tile(np.linspace(64, 200, self.IMAGE_W, dtype=np.uint8), (self.IMAGE_H, 1))
        img = Image()
        img.header.stamp = self.get_clock().now().to_msg()
        img.header.frame_id = frame_id
        img.height = self.IMAGE_H
        img.width = self.IMAGE_W
        img.encoding = 'mono8'
        img.is_bigendian = 0
        img.step = self.IMAGE_W
        img.data = rows.tobytes()
        return img


def main(args=None):
    rclpy.init(args=args)
    node = DepthCameraNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
