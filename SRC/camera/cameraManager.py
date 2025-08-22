# -*- coding: utf-8 -*-
import RPi.GPIO as GPIO
import time
import cv2
import subprocess
import os
import numpy as np

class USBCamera:
    """
    Controls a USB camera, performs object detection with angle estimation,
    and allows for interactive tuning by clicking the video stream.
    """

    def __init__(self, gpio_pin=26, camera_index=0, init_duration=1):
        self.gpio_pin = gpio_pin
        self.camera_index = camera_index
        self.init_duration = init_duration
        self.camera = None
        self.vendor_id = None
        self.product_id = None
        self.current_hsv_frame = None
        # To hold the lookup table for angle estimation
        self.angle_lookup_table = None
        self.camera_fov = 140 # Camera Field of View in degrees

        self.color_bounds = {
            'green': {'lower': np.array([35, 100, 100]), 'upper': np.array([85, 255, 255])},
            'red': {'lower1': np.array([0, 120, 70]), 'upper1': np.array([10, 255, 255]),
                    'lower2': np.array([170, 120, 70]), 'upper2': np.array([180, 255, 255])},
            'pink': {'lower': np.array([140, 100, 100]), 'upper': np.array([170, 255, 255])}
        }

        try:
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(self.gpio_pin, GPIO.OUT)
            GPIO.output(self.gpio_pin, GPIO.LOW)
            print("GPIO setup complete.")
            self._toggle_relay()
            self.vendor_id, self.product_id = self._get_camera_ids()
            if not self.vendor_id or not self.product_id:
                raise RuntimeError("Could not find a USB device with 'camera' in its name.")
            print(f"Successfully found camera with ID {self.vendor_id}:{self.product_id}")
        except Exception as e:
            print(f"An error occurred during initialization: {e}")
            GPIO.cleanup()
            raise

    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and self.current_hsv_frame is not None:
            hsv_pixel = self.current_hsv_frame[y, x]
            print(f"Clicked at (x={x}, y={y}) - HSV Value: {hsv_pixel}")

    def _generate_angle_lookup_table(self, frame_width):
        """
        Creates a lookup table that maps pixel x-coordinates to a non-linear angle.
        """
        self.angle_lookup_table = []
        weights = [0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.4, 1.3, 1.2, 1.1, 1.0, 0.9, 0.8]
        total_weight = sum(weights)
        
        angle_step = self.camera_fov / len(weights)
        current_pixel = 0
        
        for i, weight in enumerate(weights):
            angle = -self.camera_fov / 2 + (i * angle_step)
            pixel_chunk = (frame_width / total_weight) * weight
            
            start_pixel = int(current_pixel)
            end_pixel = int(current_pixel + pixel_chunk)
            self.angle_lookup_table.append({
                'angle': int(angle + angle_step / 2),
                'range': (start_pixel, end_pixel)
            })
            current_pixel += pixel_chunk
        print("Generated Angle Lookup Table:")
        for entry in self.angle_lookup_table:
            print(f"  Angle: {entry['angle']}deg -> Pixel Range: {entry['range']}")

    def _estimate_horizontal_angle(self, object_center_x):
        """Finds the bearing angle of an object based on its x-coordinate."""
        if self.angle_lookup_table is None:
            return None
        for entry in self.angle_lookup_table:
            if entry['range'][0] <= object_center_x < entry['range'][1]:
                return entry['angle']
        return None

    def _toggle_relay(self):
        print(f"Activating relay on GPIO {self.gpio_pin} for camera mode switch...")
        GPIO.output(self.gpio_pin, GPIO.HIGH)
        time.sleep(self.init_duration)
        GPIO.output(self.gpio_pin, GPIO.LOW)
        print("Relay deactivated. Giving OS time to detect devices...")
        time.sleep(2)

    def _get_camera_ids(self):
        print("\n--- Searching for camera on USB bus... ---")
        try:
            lsusb_output = subprocess.check_output(['lsusb']).decode('utf-8')
            camera_line = next((line for line in lsusb_output.split('\n') if 'camera' in line.lower()), None)
            if not camera_line: return None, None
            print(f"Found device line: '{camera_line}'")
            parts = camera_line.split()
            vid_pid = parts[5]
            return vid_pid.split(':')
        except Exception as e:
            print(f"An error occurred while getting camera IDs: {e}")
            return None, None

    def _probe_video_driver(self):
        if not self.vendor_id or not self.product_id: return False
        print(f"Forcing kernel to probe for driver for device {self.vendor_id}:{self.product_id}...")
        try:
            subprocess.run(['modprobe', 'uvcvideo'], check=True, capture_output=True)
            driver_path = '/sys/bus/usb/drivers/uvcvideo/new_id'
            with open(driver_path, 'w') as f: f.write(f"{self.vendor_id} {self.product_id}")
            print("Driver probe command sent. Waiting 2 seconds for device to stabilize...")
            time.sleep(2)
            return True
        except (subprocess.CalledProcessError, IOError, FileNotFoundError) as e:
            print(f"Could not probe driver: {e}")
            return False

    def _process_contours(self, display_frame, mask, color_name, line_y):
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) > 300:
                x, y, w, h = cv2.boundingRect(cnt)
                if y < line_y < y + h:
                    rect = cv2.minAreaRect(cnt)
                    rotation_angle = int(rect[2])
                    
                    object_center_x = x + w // 2
                    bearing_angle = self._estimate_horizontal_angle(object_center_x)
                    
                    cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
                    
                    if bearing_angle is not None:
                        label = f"{color_name} | Bearing: {bearing_angle}deg | Rot: {rotation_angle}deg"
                    else:
                        label = f"{color_name} | Rot: {rotation_angle}deg"
                        
                    cv2.putText(display_frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                    print(f"FLAGGED: {color_name} object crossing the line.")

    def start_stream(self, max_retries=30):
        for attempt in range(max_retries):
            print(f"--- Connection Attempt {attempt + 1}/{max_retries} ---")
            self._probe_video_driver()
            self.camera = cv2.VideoCapture(self.camera_index)
            if self.camera and self.camera.isOpened():
                print(f"Successfully opened camera stream using index {self.camera_index}.")
                break
            else:
                if self.camera: self.camera.release()
                if attempt < max_retries - 1:
                    print("Failed to open camera stream. Waiting 1 second before retrying...")
                    time.sleep(1)
        else:
            print(f"Error: Could not open camera stream after {max_retries} attempts.")
            self.stop_stream()
            return
            
        window_name = 'Object Detection Stream'
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, self._mouse_callback)

        print("\nStarting camera stream... Press 'q' to quit.")
        try:
            while True:
                ret, frame = self.camera.read()
                if not ret: break

                frame = cv2.flip(frame, -1)
                display_frame = frame.copy()

                height, width, _ = frame.shape
                
                if self.angle_lookup_table is None:
                    self._generate_angle_lookup_table(width)

                halfway_line_y = height // 2
                hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                self.current_hsv_frame = hsv_frame

                for color_name, bounds in self.color_bounds.items():
                    if color_name == 'red':
                        mask1 = cv2.inRange(hsv_frame, bounds['lower1'], bounds['upper1'])
                        mask2 = cv2.inRange(hsv_frame, bounds['lower2'], bounds['upper2'])
                        final_mask = cv2.bitwise_or(mask1, mask2)
                    else:
                        final_mask = cv2.inRange(hsv_frame, bounds['lower'], bounds['upper'])
                    
                    kernel = np.ones((5, 5), np.uint8)
                    final_mask = cv2.erode(final_mask, kernel, iterations=1)
                    final_mask = cv2.dilate(final_mask, kernel, iterations=1)
                    
                    self._process_contours(display_frame, final_mask, color_name.capitalize(), halfway_line_y)
                    cv2.imshow(f"{color_name.capitalize()} Mask", final_mask)
                
                cv2.line(display_frame, (0, halfway_line_y), (width, halfway_line_y), (255, 255, 0), 2)
                cv2.imshow(window_name, display_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'): break
        finally:
            self.stop_stream()

    def stop_stream(self):
        if self.camera is not None and self.camera.isOpened(): self.camera.release()
        cv2.destroyAllWindows()
        GPIO.cleanup()
        print("Camera stream stopped and all resources released.")

    def __del__(self):
        self.stop_stream()

if __name__ == '__main__':
    RELAY_PIN = 26
    CAMERA_INDEX = 0
    print("--- USB Camera Controller ---")
    print("NOTE: This script must be run with 'sudo' using the venv python.")
    try:
        my_camera = USBCamera(gpio_pin=RELAY_PIN, camera_index=CAMERA_INDEX)
        my_camera.start_stream()
    except Exception as e:
        print(f"Program failed to start due to a critical error: {e}")