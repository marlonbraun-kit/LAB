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
from std_msgs.msg import Header, Bool
from cv_bridge import CvBridge
import cv2
import mediapipe as mp
import math

class SafetyShieldNode(Node):
    def __init__(self):
        super().__init__('safety_shield_node')
        
        # Subscriber (Raw video input)
        self.subscription = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 10)
        
        # Declare parameter with default value
        self.declare_parameter('mirror_image', True)
            
        # Publisher 1 (Boolean output for the robot)
        self.safety_pub = self.create_publisher(Bool, '/target_can_pose', 10)
        
        # Publisher 2 (Processed video output)
        self.image_pub = self.create_publisher(Image, '/camera/depth/image_raw', 10)
            
        self.cv_bridge = CvBridge()
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Allow up to 2 hands
        self.hands = self.mp_hands.Hands(model_complexity=0, max_num_hands=2)
        
        self.PROXIMITY_THRESHOLD = 150
        self.get_logger().info('Safety shield activated. Streaming video...')

    def image_callback(self, msg):
        try:
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception: return

        # Read parameter and mirror the image if necessary
        mirror = self.get_parameter('mirror_image').value
        if mirror:
            cv_image = cv2.flip(cv_image, 1) # Horizontal flip (mirror)

        # MediaPipe requires RGB images for processing
        image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)
        
        safety_msg = Bool()
        safety_msg.data = False 
        
        # Default text (Safe)
        texto_estado = "ESTADO: SEGURO"
        color_texto = (0, 255, 0) # Green in BGR

        if results.multi_hand_landmarks:
            # Iterate over all detected hands
            for hand in results.multi_hand_landmarks:
                
                # Draw skeleton of the hand
                self.mp_drawing.draw_landmarks(
                    cv_image, hand, self.mp_hands.HAND_CONNECTIONS)
                
                # Calculate distance of the hand
                p0 = hand.landmark[0]
                p9 = hand.landmark[9]
                h, w, _ = cv_image.shape
                dist_px = math.sqrt(((p0.x - p9.x)*w)**2 + ((p0.y - p9.y)*h)**2)

                # If any hand exceeds the threshold, trigger danger
                if dist_px > self.PROXIMITY_THRESHOLD:
                    safety_msg.data = True
                    texto_estado = "PELIGRO: PARADA!"
                    color_texto = (0, 0, 255) # Red in BGR
                
        # Write text on the image
        cv2.putText(cv_image, texto_estado, (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, color_texto, 3, cv2.LINE_AA)

        # Publish data
        self.safety_pub.publish(safety_msg)
        
        try:
            # Publish the final video with drawings
            ros_image_msg = self.cv_bridge.cv2_to_imgmsg(cv_image, encoding="bgr8")
            self.image_pub.publish(ros_image_msg)
        except Exception as e:
            pass

def main(args=None):
    rclpy.init(args=args)
    node = SafetyShieldNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
