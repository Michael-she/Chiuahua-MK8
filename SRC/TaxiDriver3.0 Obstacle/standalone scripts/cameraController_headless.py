import cv2
import sys
import os
import time
import numpy as np
from datetime import datetime
from threading import Thread
from queue import Queue

# Import relay control functions
try:
    from relay_control import setup_gpio, activate_relay, cleanup
    RELAY_AVAILABLE = True
    print("Relay control module imported successfully")
except ImportError as e:
    RELAY_AVAILABLE = False
    print(f"Warning: Could not import relay control: {e}")

# --- OPTIMIZATION: Thread-Safe Camera Stream Class ---
class WebcamStream:
    def __init__(self, src=0, width=640, height=480):
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        while not self.stopped:
            if not self.grabbed:
                self.stop()
            else:
                (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        self.stream.release()

# Import relay control functions
try:
    from relay_control import setup_gpio, activate_relay, cleanup
    RELAY_AVAILABLE = True
    print("Relay control module imported successfully")
except ImportError as e:
    RELAY_AVAILABLE = False
    print(f"Warning: Could not import relay control: {e}")

# Adjust the camera_index if you have multiple cameras or if 0 doesn't work.
camera_index = 0 
MAX_CAMERA_RETRIES = 3
RETRY_DELAY = 2  # seconds between retries

def try_open_camera(index, attempt=1):
    """Attempt to open camera with retry logic and relay activation"""
    print(f"Attempt {attempt}: Trying to open camera with index {index}...")
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

# Try to open camera with relay reset capability
cap = None
for attempt in range(1, MAX_CAMERA_RETRIES + 1):
    cap = try_open_camera(camera_index, attempt)
    if cap is not None:
        break
    
    if attempt < MAX_CAMERA_RETRIES:
        print(f"Retrying in {RETRY_DELAY} seconds...")
        time.sleep(RETRY_DELAY)

if cap is None:
    print(f"Error: Could not open camera with index {camera_index} after {MAX_CAMERA_RETRIES} attempts.")
    print("Available video devices:")
    import subprocess
    try:
        result = subprocess.run(['ls', '/dev/video*'], capture_output=True, text=True, shell=True)
        print(result.stdout)
    except:
        print("Could not list video devices")
    sys.exit(1)

print(f"Camera {camera_index} opened successfully!")

# Advanced camera class with object detection and angle estimation (Headless Version)
class USBCameraControllerHeadless:
    def __init__(self, camera_source, width=640, height=480):
        self.camera_source = camera_source
        self.frame_width = width
        self.frame_height = height
        self.current_hsv_frame = None
        self.angle_lookup_table = None
        self.camera_fov = 140
        
        # Morphological kernel for image processing
        self.morph_kernel = np.ones((5, 5), np.uint8)
        
        # Color detection bounds
        self.color_bounds = {
            'green': {'lower': np.array([35, 100, 100]), 'upper': np.array([85, 255, 255])},
            'red': {'lower1': np.array([0, 120, 70]), 'upper1': np.array([10, 255, 255]),
                    'lower2': np.array([170, 120, 70]), 'upper2': np.array([180, 255, 255])},
            'pink': {'lower': np.array([140, 100, 100]), 'upper': np.array([170, 255, 255])}
        }
        
        # Use threaded camera stream
        self.camera_stream = WebcamStream(src=camera_source, width=width, height=height).start()
        time.sleep(2.0)  # Allow camera to initialize
        
        self._generate_angle_lookup_table(self.frame_width)
        
        # Create output directory
        self.output_dir = "camera_captures"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
    
    def _generate_angle_lookup_table(self, frame_width):
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
        print("Generated Angle Lookup Table.")
    
    def _estimate_horizontal_angle(self, object_center_x):
        if self.angle_lookup_table is None:
            return None
        for entry in self.angle_lookup_table:
            if entry['range'][0] <= object_center_x < entry['range'][1]:
                return entry['angle']
        return None
    
    def _process_contours(self, display_frame, mask, color_name, line_y):
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
                    
                    # Draw on display frame
                    cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
                    label = f"{color_name} | Bearing: {bearing_angle}deg" if bearing_angle is not None else color_name
                    cv2.putText(display_frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                    
                    # Store detection info
                    detections.append({
                        'color': color_name,
                        'center_x': object_center_x,
                        'bearing': bearing_angle,
                        'area': cv2.contourArea(cnt),
                        'bbox': (x, y, w, h)
                    })
        return detections
    
    def start_stream(self, max_frames=300, save_every_n_frames=30):
        halfway_line_y = self.frame_height // 2
        
        print("Starting headless camera stream with object detection...")
        print(f"Saving annotated captures to {self.output_dir}/ directory")
        print("Press Ctrl+C to stop...")
        
        frame_count = 0
        total_detections = {'green': 0, 'red': 0, 'pink': 0}
        
        try:
            while True:
                frame = self.camera_stream.read()
                if frame is None:
                    continue
                
                # Flip frame (optional, remove if not needed)
                frame = cv2.flip(frame, -1)
                display_frame = frame.copy()
                
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
                    detections = self._process_contours(display_frame, final_mask, color_name.capitalize(), halfway_line_y)
                    all_detections.extend(detections)
                    total_detections[color_name] += len(detections)
                
                # Draw horizontal reference line
                cv2.line(display_frame, (0, halfway_line_y), (self.frame_width, halfway_line_y), (255, 255, 0), 2)
                
                # Save annotated frame periodically or when detections are found
                if frame_count % save_every_n_frames == 0 or len(all_detections) > 0:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"{self.output_dir}/capture_{timestamp}_{frame_count:06d}.jpg"
                    cv2.imwrite(filename, display_frame)
                    
                    if len(all_detections) > 0:
                        detection_info = ", ".join([f"{d['color']}@{d['bearing']}°" for d in all_detections])
                        print(f"Frame {frame_count}: {len(all_detections)} objects detected - {detection_info}")
                        print(f"  Saved: {filename}")
                    else:
                        print(f"Saved frame: {filename}")
                
                frame_count += 1
                
                # Break after max_frames for testing
                if frame_count >= max_frames:
                    print("Test completed")
                    break
                    
        except KeyboardInterrupt:
            print("\nCapture stopped by user")
        finally:
            self.stop_stream()
            
        print(f"\nDetection Summary:")
        print(f"Total frames processed: {frame_count}")
        for color, count in total_detections.items():
            print(f"  {color.capitalize()} objects detected: {count}")
    
    def stop_stream(self):
        if hasattr(self, 'camera_stream') and self.camera_stream is not None:
            self.camera_stream.stop()
        print("Camera stream stopped.")

# Create and start the advanced camera controller
try:
    camera_controller = USBCameraControllerHeadless(camera_index)  # Pass camera index, not camera object
    camera_controller.start_stream()
except KeyboardInterrupt:
    print("\nProgram interrupted by user")
except Exception as e:
    print(f"Error: {e}")
finally:
    if 'cap' in locals() and cap is not None:
        cap.release()
    if RELAY_AVAILABLE:
        try:
            cleanup()
        except:
            pass
