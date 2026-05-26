#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import pyrealsense2 as rs
import numpy as np
import cv2
import math
import os
import mediapipe as mp
from ultralytics import YOLO

# Mensajes de tu proyecto
from ur3_interfaces.msg import CanDetection, CanDetectionArray
from std_msgs.msg import Float32

# Hand-proximity tuning: pixel distance between MediaPipe landmark 0 (wrist)
# and 9 (middle-finger MCP), at or above which the hand is treated as
# critically close (proximity scalar -> 0.0). Below the threshold the value
# scales linearly back up to 1.0 (no hand visible).
HAND_PROXIMITY_THRESHOLD_PX = 150.0

class NativeVisionNode(Node):
    def __init__(self):
        super().__init__('native_vision_node')
        
        # Publicador de ROS 2 (Queue Size = 1)
        self.pub = self.create_publisher(CanDetectionArray, '/front_detections', 1)

        # Hand-proximity safety: pick_place_manager consumes /human_proximity
        # (Float32 in [0, 1], 0.0 = danger) and halves motion speed below
        # HUMAN_PROXIMITY_THRESHOLD = 0.5.
        self.proximity_pub = self.create_publisher(Float32, '/human_proximity', 10)
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(model_complexity=0, max_num_hands=2)

        # Cargar YOLO
        self.get_logger().info("Cargando modelo YOLO...")
        model_path = os.path.expanduser('~/ros2_ws/LAB/best.pt')
        self.model = YOLO(model_path)
        # YOLO class indices published on /front_detections
        # 0: beer, 1: coke, 2: lemon, 3: orange
        self.nombres_permitidos = {0: "beer", 1: "coke", 2: "lemon", 3: "orange"}

        # Iniciar Cámara SR305
        self.get_logger().info("Iniciando cámara SR305 en Linux...")
        self.pipe = rs.pipeline()
        self.cfg = rs.config()
        self.cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        
        self.profile = self.pipe.start(self.cfg)
        self.align = rs.align(rs.stream.color)
        
        # Timer a 30Hz para procesar fotogramas
        self.timer = self.create_timer(1.0 / 30.0, self.process_frame)
        self.get_logger().info("¡Nodo de Visión Nativo funcionando a 30 FPS!")

    def process_frame(self):
        try:
            # MODIFICACIÓN CLAVE: Se añade un timeout bajo (100ms) para no bloquear ROS 2
            # Si no llega un frame a tiempo, simplemente pasará al siguiente ciclo del timer
            frames = self.pipe.wait_for_frames(100)
        except RuntimeError:
            # Esto evita que el nodo se cuelgue si la cámara "parpadea" o se retrasa
            return 
        except Exception as e:
            self.get_logger().error(f"Error de hardware en cámara: {e}")
            return

        try:
            aligned_frames = self.align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            if not depth_frame or not color_frame:
                return

            color_image = np.asanyarray(color_frame.get_data())
            intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics

            # The camera is mounted sideways on the gripper AND the identify
            # pose adds another +90° rotation about wrist_3, so the raw 640x480
            # frame is rotated 180° relative to gravity. Rotate 180° (= two
            # 90° CW rotations) to put cans upright for YOLO and for display.
            # IMPORTANT: depth_frame and `intrinsics` are NOT rotated — bounding-
            # box centres are mapped back to the original-frame pixel before
            # the depth lookup + deprojection, so the published 3D position is
            # still in the unrotated camera_optical_link frame.
            H_orig, W_orig = color_image.shape[:2]   # 480, 640
            color_rot = cv2.rotate(color_image, cv2.ROTATE_180)

            # conf: minimum class-probability for a detection to be kept.
            #       Lower => more (including weaker) detections survive.
            # iou:  IoU threshold used by Non-Maximum Suppression. When two
            #       boxes overlap by *more* than this, NMS keeps only the
            #       higher-confidence one. Lower => stricter dedup (fewer
            #       overlapping boxes). False positives are already rare on
            #       this scene, so we err on the permissive side for `conf`.
            results = self.model.track(
                color_rot, persist=True, verbose=False,
                conf=0.15, iou=0.45,
            )

            # Crear el mensaje ROS 2
            msg_array = CanDetectionArray()
            timestamp = self.get_clock().now().to_msg()
            msg_array.header.stamp = timestamp
            msg_array.header.frame_id = "camera_optical_link"
            detections_list = []

            for result in results:
                if result.boxes:
                    for box in result.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        x_center_rot = int((x1 + x2) / 2)
                        y_center_rot = int((y1 + y2) / 2)

                        # Inverse of ROTATE_180: a pixel at (xr, yr) in the
                        # rotated image came from (W_orig-1-xr, H_orig-1-yr)
                        # in the original.
                        x_center = W_orig - 1 - x_center_rot
                        y_center = H_orig - 1 - y_center_rot
                        x_center = max(0, min(W_orig - 1, x_center))
                        y_center = max(0, min(H_orig - 1, y_center))

                        track_id = "0"
                        if box.id is not None:
                            track_id = str(int(box.id[0]))

                        clase_id = int(box.cls[0])

                        label = self.nombres_permitidos.get(clase_id, "unknown")
                        if label == "unknown":
                            continue

                        distance = depth_frame.get_distance(x_center, y_center)

                        if distance > 0:
                            # Push the point from the front face of the can to
                            # its centre. Depth reads the nearest surface (the
                            # front), so add the can radius along the camera
                            # optical +Z axis (= viewing direction) to land on
                            # the can centre.
                            CAN_RADIUS_M = 0.033
                            distance_to_center = distance + CAN_RADIUS_M
                            spatial_coords = rs.rs2_deproject_pixel_to_point(
                                intrinsics, [x_center, y_center], distance_to_center
                            )
                            real_x, real_y, real_z = spatial_coords

                            det = CanDetection()
                            det.header.stamp = timestamp
                            det.header.frame_id = "camera_optical_link"
                            det.id = track_id
                            det.class_id = clase_id
                            det.confidence = float(box.conf[0])
                            det.position.x = real_x
                            det.position.y = real_y
                            det.position.z = real_z
                            det.source = "front"

                            detections_list.append(det)

                            cv2.rectangle(color_rot, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(color_rot, f"{label} {distance:.2f}m", (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            msg_array.detections = detections_list
            self.pub.publish(msg_array)

            # ---- Hand-proximity safety overlay ----------------------------
            # Run MediaPipe on the same (rotated) RGB frame YOLO just saw, so
            # hands and cans share a single display window. Landmark spread
            # (wrist <-> middle-finger MCP, in pixels) is a cheap proxy for
            # hand-to-camera distance.
            # Isolated try/except: a MediaPipe failure must NOT prevent the
            # cv2.imshow call below — otherwise the camera window goes dark.
            try:
                hand_rgb = cv2.cvtColor(color_rot, cv2.COLOR_BGR2RGB)
                hand_results = self.hands.process(hand_rgb)
                h_img, w_img = color_rot.shape[:2]
                max_spread_px = 0.0
                if hand_results.multi_hand_landmarks:
                    for hand in hand_results.multi_hand_landmarks:
                        self.mp_drawing.draw_landmarks(
                            color_rot, hand, self.mp_hands.HAND_CONNECTIONS
                        )
                        p0 = hand.landmark[0]
                        p9 = hand.landmark[9]
                        spread = math.hypot(
                            (p0.x - p9.x) * w_img, (p0.y - p9.y) * h_img
                        )
                        if spread > max_spread_px:
                            max_spread_px = spread

                if HAND_PROXIMITY_THRESHOLD_PX <= 0.0:
                    proximity = 1.0
                else:
                    proximity = 1.0 - min(
                        max_spread_px / HAND_PROXIMITY_THRESHOLD_PX, 1.0
                    )
                self.proximity_pub.publish(Float32(data=float(proximity)))

                if proximity < 0.5:
                    overlay_label, overlay_color = 'PELIGRO: PARADA!', (0, 0, 255)
                else:
                    overlay_label, overlay_color = 'ESTADO: SEGURO', (0, 255, 0)
                cv2.putText(
                    color_rot, f'{overlay_label}  ({proximity:.2f})',
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, overlay_color, 2,
                    cv2.LINE_AA,
                )
            except Exception as e:
                self.get_logger().warn(f'Hand detection failed: {e}')

            cv2.imshow("Visor SR305 (Linux Nativo)", color_rot)
            cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f"Error procesando frame de vision: {e}")

    def destroy_node(self):
        self.get_logger().info("Deteniendo cámara y cerrando ventanas...")
        self.pipe.stop()
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = NativeVisionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()