#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import pyrealsense2 as rs
import numpy as np
import cv2
import os
from ultralytics import YOLO

# Mensajes de tu proyecto
from ur3_interfaces.msg import CanDetection, CanDetectionArray

class NativeVisionNode(Node):
    def __init__(self):
        super().__init__('native_vision_node')
        
        # Publicador de ROS 2 (Queue Size = 1)
        self.pub = self.create_publisher(CanDetectionArray, '/front_detections', 1)
        
        # Cargar YOLO
        self.get_logger().info("Cargando modelo YOLO...")
        model_path = os.path.expanduser('~/ros2_ws/LAB/best.pt')
        self.model = YOLO(model_path)
        self.nombres_permitidos = {0: "coke", 1: "pepsi", 2: "sprite"}

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

            # The camera is mounted sideways on the gripper, so the raw 640x480
            # frame is rotated relative to gravity. Rotate 90° CW to a 480x640
            # portrait image for YOLO (trained on upright cans) and for display.
            # IMPORTANT: depth_frame and `intrinsics` are NOT rotated — bounding-
            # box centres are mapped back to the original-frame pixel before
            # the depth lookup + deprojection, so the published 3D position is
            # still in the unrotated camera_optical_link frame.
            H_orig, W_orig = color_image.shape[:2]   # 480, 640
            color_rot = cv2.rotate(color_image, cv2.ROTATE_90_CLOCKWISE)

            results = self.model.track(color_rot, persist=True, verbose=False)

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

                        # Inverse of ROTATE_90_CLOCKWISE: a pixel at (xr, yr)
                        # in the rotated image came from (yr, H_orig-1-xr) in
                        # the original.
                        x_center = y_center_rot
                        y_center = H_orig - 1 - x_center_rot
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
                            spatial_coords = rs.rs2_deproject_pixel_to_point(
                                intrinsics, [x_center, y_center], distance
                            )
                            real_x, real_y, real_z = spatial_coords

                            det = CanDetection()
                            det.header.stamp = timestamp
                            det.header.frame_id = "camera_optical_link"
                            det.id = track_id
                            det.class_name = label
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