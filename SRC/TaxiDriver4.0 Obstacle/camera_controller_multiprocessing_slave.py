#!/usr/bin/env python3
"""
Camera Controller Slave for Multiprocessin        self.color_bounds = {
            'green': {'lower': np.array([35, 80, 30]), 'upper': np.array([150, 255, 255])},
            'red': {'lower1': np.array([0, 120, 70]), 'upper1': np.array([10, 255, 255]),
                    'lower2': np.array([170, 120, 70]), 'upper2': np.array([180, 255, 255])}
        }gration with TaxiDriver
Detects objects and shares detection data through shared memory
"""

import cv2
import sys
import os
import time
import numpy as np
import multiprocessing
import threading
from datetime import datetime

# Set display for OpenCV (optional for headless operation)
os.environ['DISPLAY'] = ':0'

# Import relay control functions    
try:
    from relay_control import setup_gpio, activate_relay, cleanup
    RELAY_AVAILABLE = True
except ImportError as e:
    RELAY_AVAILABLE = False
    print(f"Warning: Could not import relay control: {e}")

# Camera configuration
camera_index = 0
MAX_CAMERA_RETRIES = 10
RETRY_DELAY = 2

def try_open_camera(index, attempt=1):
    """Attempt to open camera with retry logic and relay activation"""
    print(f"Camera attempt {attempt}: Trying to open camera with index {index}...")
    cap = cv2.VideoCapture(index)
    
    if cap.isOpened():
        print(f"✓ Camera {index} opened successfully on attempt {attempt}!")
        return cap
    else:
        print(f"✗ Failed to open camera {index} on attempt {attempt}")
        cap.release()
        
        if attempt < MAX_CAMERA_RETRIES and RELAY_AVAILABLE:
            print("Attempting to reset camera using relay...")
            try:
                setup_gpio()
                activate_relay()
                cleanup()
                print(f"Relay activation completed. Waiting {RETRY_DELAY} seconds before retry...")
                time.sleep(RETRY_DELAY)
            except Exception as e:
                print(f"Error during relay activation: {e}")
                if 'cleanup' in globals():
                    cleanup()
        
        return None

class CameraControllerSlave:
    def __init__(self, width=640, height=480):
        self.frame_width = width
        self.frame_height = height
        self.current_hsv_frame = None
        self.camera_fov = 140
        
        # Morphological kernel for image processing
        self.morph_kernel = np.ones((5, 5), np.uint8)
        
        
        # Color detection bounds
        self.color_bounds = {
            'green': {'lower': np.array([35, 80, 60]), 'upper': np.array([150, 255, 255])},
            'red': {'lower1': np.array([0, 120, 70]), 'upper1': np.array([10, 255, 255]),
                    'lower2': np.array([170, 120, 70]), 'upper2': np.array([180, 255, 255])}
        }
        
        # Calibration points for angle estimation
        self.calibration_points = [
            (100, -45), (134, -40), (166, -35), (194, -30), (222, -25),
            (232, -20), (256, -15), (286, -10), (312, -5), (352, 0),
            (379, 5), (405, 10), (436, 15), (461, 20), (489, 25),
            (518, 30), (539, 35), (589, 40), (594, 45),
        ]
        self.calibration_points.sort(key=lambda x: x[0])
    
    def _estimate_horizontal_angle(self, object_center_x):
        """Estimate angle using linear interpolation between calibration points"""
        if len(self.calibration_points) < 2:
            return None
        
        # Handle edge cases
        if object_center_x <= self.calibration_points[0][0]:
            return self.calibration_points[0][1]
        if object_center_x >= self.calibration_points[-1][0]:
            return self.calibration_points[-1][1]
        
        # Find the two calibration points to interpolate between
        for i in range(len(self.calibration_points) - 1):
            x1, y1 = self.calibration_points[i]
            x2, y2 = self.calibration_points[i + 1]
            
            if x1 <= object_center_x <= x2:
                if x2 == x1:  # Avoid division by zero
                    return y1
                
                interpolated_angle = y1 + (y2 - y1) * (object_center_x - x1) / (x2 - x1)
                return round(interpolated_angle, 1)
        
        return None
    
    def _process_contours(self, mask, color_name, line_y):
        """Process contours and return detection data"""
        detections = []
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            if cv2.contourArea(cnt) > 150:
                x, y, w, h = cv2.boundingRect(cnt)
                if y < line_y < y + h:
                    rect = cv2.minAreaRect(cnt)
                    rotation_angle = int(rect[2])
                    object_center_x = x + w // 2
                    bearing_angle = self._estimate_horizontal_angle(object_center_x)
                    
                    detection = {
                        'color': color_name,
                        'center_x': object_center_x,
                        'center_y': y + h // 2,
                        'bearing': bearing_angle,
                        'area': cv2.contourArea(cnt),
                        'bbox': (x, y, w, h),
                        'rotation': rotation_angle,
                        'timestamp': time.time()
                    }
                    detections.append(detection)
        
        return detections
    
    def process_frame(self, frame):
        """Process a single frame and return all detections"""
        if frame is None:
            return []
        
        halfway_line_y = self.frame_height // 2 - 20
        
        # Convert to HSV for color detection
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        self.current_hsv_frame = hsv_frame
        
        all_detections = []
        
        # Process each color
        for color_name, bounds in self.color_bounds.items():
            if color_name == 'red':
                mask1 = cv2.inRange(hsv_frame, bounds['lower1'], bounds['upper1'])
                mask2 = cv2.inRange(hsv_frame, bounds['lower2'], bounds['upper2'])
                final_mask = cv2.bitwise_or(mask1, mask2)
            else:
                final_mask = cv2.inRange(hsv_frame, bounds['lower'], bounds['upper'])
            
            # Morphological operations to clean up the mask
            final_mask = cv2.erode(final_mask, self.morph_kernel, iterations=1)
            final_mask = cv2.dilate(final_mask, self.morph_kernel, iterations=1)
            
            # Process contours and get detections
            detections = self._process_contours(final_mask, color_name, halfway_line_y)
            all_detections.extend(detections)
        
        return all_detections

def camera_controller_process(shared_detections, shared_detection_count, shutdown_flag):
    """
    Camera controller process for multiprocessing communication.
    Shares detected objects with the main taxi driver process.
    Saves every 10th annotated frame.
    """
    print("Camera controller process started")
    
    # Try to open camera
    cap = None
    for attempt in range(1, MAX_CAMERA_RETRIES + 1):
        cap = try_open_camera(camera_index, attempt)
        if cap is not None:
            break
        
        if attempt < MAX_CAMERA_RETRIES:
            print(f"Retrying camera in {RETRY_DELAY} seconds...")
            time.sleep(RETRY_DELAY)
    
    if cap is None:
        print(f"Failed to open camera after {MAX_CAMERA_RETRIES} attempts")
        return
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera initialized: {width}x{height}")
    
    # Create camera controller
    controller = CameraControllerSlave(width, height)
    
    frame_count = 0
    save_frame_interval = 10  # Save every 10th frame
    
    # Create camera_captures directory if it doesn't exist
    import os
    os.makedirs("camera_captures", exist_ok=True)
    
    try:
        while shutdown_flag.value == 0:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                time.sleep(0.1)
                continue
            
            # Flip frame if needed
            frame = cv2.flip(frame, -1)
            
            # Process frame for detections
            detections = controller.process_frame(frame)
            
            # Create annotated frame for saving
            annotated_frame = frame.copy()
            halfway_line_y = height // 2 - 20
            
            # Draw halfway line
            cv2.line(annotated_frame, (0, halfway_line_y), (width, halfway_line_y), (255, 255, 0), 2)
            
            # Annotate detections
            for detection in detections:
                x, y, w, h = detection['bbox']
                center_x = detection['center_x']
                center_y = detection['center_y']
                color_name = detection['color']
                bearing = detection['bearing']
                area = detection['area']
                
                # Get HSV values at the center of the detected object
                hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                hsv_pixel = hsv_frame[center_y, center_x]
                h_val, s_val, v_val = hsv_pixel
                
                # Choose annotation color based on detected object color
                if color_name == 'green':
                    annotation_color = (0, 255, 0)  # Green
                elif color_name == 'red':
                    annotation_color = (0, 0, 255)  # Red
                else:
                    annotation_color = (255, 255, 255)  # White
                
                # Draw bounding box
                cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), annotation_color, 2)
                
                # Draw center point
                cv2.circle(annotated_frame, (center_x, center_y), 5, annotation_color, -1)
                
                # Add text annotations including HSV values
                text_lines = [
                    f"{color_name.upper()}",
                    f"Bearing: {bearing:.1f}°" if bearing is not None else "Bearing: N/A",
                    f"Area: {area:.0f}",
                    f"Pos: ({center_x}, {center_y})",
                    f"HSV: ({h_val}, {s_val}, {v_val})"
                ]
                
                # Draw text with background for better visibility
                text_y = y - 10
                for i, text in enumerate(text_lines):
                    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                    text_bg_y = text_y - text_size[1] - 5
                    
                    # Draw background rectangle
                    cv2.rectangle(annotated_frame, 
                                (x, text_bg_y), 
                                (x + text_size[0] + 5, text_y + 5), 
                                (0, 0, 0), -1)
                    
                    # Draw text
                    cv2.putText(annotated_frame, text, (x + 2, text_y), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, annotation_color, 1)
                    
                    text_y -= (text_size[1] + 8)
            
            # Add frame information
            info_text = f"Frame: {frame_count} | Objects: {len(detections)} | Time: {time.strftime('%H:%M:%S')}"
            cv2.putText(annotated_frame, info_text, (10, height - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Save every 10th annotated frame
            if frame_count % save_frame_interval == 0:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"camera_captures/annotated_frame_{timestamp}_{frame_count:06d}.jpg"
                cv2.imwrite(filename, annotated_frame)
                print(f"Saved annotated frame: {filename}")
            
            # Update shared memory with detection data
            with shared_detection_count.get_lock():
                shared_detection_count.value = min(len(detections), len(shared_detections))
                
                for i in range(shared_detection_count.value):
                    detection = detections[i]
                    # Pack detection data into shared array
                    # Format: [color_id, center_x, center_y, bearing, area, bbox_x, bbox_y, bbox_w, bbox_h, timestamp]
                    color_id = {'green': 1, 'red': 2}.get(detection['color'], 0)
                    
                    shared_detections[i * 10 + 0] = color_id
                    shared_detections[i * 10 + 1] = detection['center_x']
                    shared_detections[i * 10 + 2] = detection['center_y']
                    shared_detections[i * 10 + 3] = detection['bearing'] if detection['bearing'] is not None else -999
                    shared_detections[i * 10 + 4] = detection['area']
                    shared_detections[i * 10 + 5] = detection['bbox'][0]  # x
                    shared_detections[i * 10 + 6] = detection['bbox'][1]  # y
                    shared_detections[i * 10 + 7] = detection['bbox'][2]  # w
                    shared_detections[i * 10 + 8] = detection['bbox'][3]  # h
                    shared_detections[i * 10 + 9] = detection['timestamp']
            
            frame_count += 1
            
            # Print detections periodically
            if frame_count % 30 == 0 and detections:
                print(f"Frame {frame_count}: Found {len(detections)} objects")
                for det in detections:
                    print(f"  {det['color']} at {det['bearing']}° (area: {det['area']:.0f})")
            
            # Control frame rate
            time.sleep(0.033)  # ~30 FPS
            
    except Exception as e:
        print(f"Error in camera controller process: {e}")
    finally:
        if cap is not None:
            cap.release()
        if RELAY_AVAILABLE:
            try:
                cleanup()
            except:
                pass
        print("Camera controller process stopped")

# Utility functions for decoding shared detection data
def decode_detection_data(shared_detections, detection_count):
    """Decode shared detection data into a list of dictionaries"""
    detections = []
    color_map = {1: 'green', 2: 'red'}
    
    for i in range(detection_count):
        base_idx = i * 10
        color_id = int(shared_detections[base_idx + 0])
        
        if color_id in color_map:
            detection = {
                'color': color_map[color_id],
                'center_x': int(shared_detections[base_idx + 1]),
                'center_y': int(shared_detections[base_idx + 2]),
                'bearing': shared_detections[base_idx + 3] if shared_detections[base_idx + 3] != -999 else None,
                'area': shared_detections[base_idx + 4],
                'bbox': (
                    int(shared_detections[base_idx + 5]),
                    int(shared_detections[base_idx + 6]),
                    int(shared_detections[base_idx + 7]),
                    int(shared_detections[base_idx + 8])
                ),
                'timestamp': shared_detections[base_idx + 9]
            }
            detections.append(detection)
    
    return detections

if __name__ == '__main__':
    # Test the camera controller in standalone mode
    print("Testing Camera Controller Slave...")
    
    # Create shared memory for testing
    max_detections = 20
    shared_detections = multiprocessing.Array('d', max_detections * 10)
    shared_detection_count = multiprocessing.Value('i', 0)
    shutdown_flag = multiprocessing.Value('i', 0)
    
    try:
        camera_controller_process(shared_detections, shared_detection_count, shutdown_flag)
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
        shutdown_flag.value = 1
