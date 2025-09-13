import cv2
import sys
import os
import time
import numpy as np
from datetime import datetime

# Set display for OpenCV
os.environ['DISPLAY'] = ':0'

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
MAX_CAMERA_RETRIES = 5
RETRY_DELAY = 5

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

# Set camera properties for better performance
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

# Get actual camera properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

print(f"Camera resolution: {width}x{height}")
print(f"Camera FPS: {fps}")

# Advanced camera class with object detection and angle estimation (Display Version)
class USBCameraControllerDisplay:
    def __init__(self, camera_cap, width=640, height=480):
        self.camera_cap = camera_cap
        self.frame_width = width
        self.frame_height = height
        self.current_hsv_frame = None
        self.angle_lookup_table = None
        self.camera_fov = 140
        
        # Morphological kernel for image processing
        self.morph_kernel = np.ones((5, 5), np.uint8)
        
        # Color detection bounds
        self.color_bounds = {
            'green': {'lower': np.array([35, 80, 30]), 'upper': np.array([150, 255, 255])},
            'red': {'lower1': np.array([0, 120, 70]), 'upper1': np.array([10, 255, 255]),
                    'lower2': np.array([160, 120, 70]), 'upper2': np.array([180, 255, 255])}
        }
        
        self._generate_angle_lookup_table(self.frame_width)
        
        # Print calibration template for easy setup
        self.print_calibration_template()
    
    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and self.current_hsv_frame is not None:
            hsv_pixel = self.current_hsv_frame[y, x]
            # Also show the estimated angle for calibration purposes
            estimated_angle = self._estimate_horizontal_angle(x)
            print(f"Clicked at (x={x}, y={y}) - HSV Value: {hsv_pixel} - Estimated Angle: {estimated_angle}°")
            print(f"Calibration point: ({x}, YOUR_MEASURED_ANGLE_HERE),")
    
    def print_calibration_template(self):
        """
        Print a template for easy copy-paste calibration.
        """
        print("\n" + "="*60)
        print("CALIBRATION TEMPLATE - Replace YOUR_ANGLE with measured values")
        print("="*60)
        template_points = [
            (0, "YOUR_ANGLE"),
            (64, "YOUR_ANGLE"), 
            (128, "YOUR_ANGLE"),
            (192, "YOUR_ANGLE"),
            (256, "YOUR_ANGLE"),
            (320, "0.0"),  # Center should be 0
            (384, "YOUR_ANGLE"),
            (448, "YOUR_ANGLE"),
            (512, "YOUR_ANGLE"),
            (576, "YOUR_ANGLE"),
            (640, "YOUR_ANGLE"),
        ]
        
        print("self.calibration_points = [")
        for pixel, angle in template_points:
            print(f"    ({pixel}, {angle}),")
        print("]")
        print("="*60 + "\n")
    
    def _generate_angle_lookup_table(self, frame_width):
        """
        Generate calibration-based lookup table for linear interpolation.
        Fill in your calibrated pixel-to-degree measurements in the calibration_points list.
        """
        # CALIBRATION TABLE - Fill in your measured values here
        # Format: (pixel_x, angle_degrees)
        # Add as many calibration points as you want for better accuracy
        self.calibration_points = [
            (100, -45),
            (134, -40),
            (166, -35),
            (194, -30),
            (222, -25),
            (232, -20),
            (256, -15),
            (286, -10),
            (312, -5),
            (352, 0),   # Center point
            (379, 5),
            (405, 10),
            (436, 15),
            (461, 20),
            (489, 25),
            (518, 30),
            (539, 35),
            (589, 40),
            (594, 45),
        ]
        
        # Sort calibration points by pixel position (just in case)
        self.calibration_points.sort(key=lambda x: x[0])
        
        print("Calibration-based angle lookup table generated.")
        print("Calibration points:")
        for pixel, angle in self.calibration_points:
            print(f"  Pixel {pixel:3d} -> {angle:+6.1f}°")
    
    def _estimate_horizontal_angle(self, object_center_x):
        """
        Estimate angle using linear interpolation between calibration points.
        """
        if not hasattr(self, 'calibration_points') or len(self.calibration_points) < 2:
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
                # Linear interpolation: y = y1 + (y2-y1) * (x-x1)/(x2-x1)
                if x2 == x1:  # Avoid division by zero
                    return y1
                
                interpolated_angle = y1 + (y2 - y1) * (object_center_x - x1) / (x2 - x1)
                return round(interpolated_angle, 1)
        
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
                    detection_info = {
                        'color': color_name,
                        'center_x': object_center_x,
                        'bearing': bearing_angle,
                        'area': cv2.contourArea(cnt),
                        'bbox': (x, y, w, h)
                    }
                    
                    # Add HSV values if available
                    if self.current_hsv_frame is not None:
                        center_y = y + h // 2
                        hsv_values = self.current_hsv_frame[center_y, object_center_x]
                        detection_info['hsv'] = tuple(hsv_values)
                    
                    detections.append(detection_info)
                    
                    # Get HSV values at the center of the detected object
                    if self.current_hsv_frame is not None:
                        center_y = y + h // 2
                        hsv_values = self.current_hsv_frame[center_y, object_center_x]
                        h_val, s_val, v_val = hsv_values
                        hsv_str = f"HSV({h_val}, {s_val}, {v_val})"
                    else:
                        hsv_str = "HSV(N/A)"
                    
                    # Get the area from the detection info
                    area = detection_info['area']
                    
                    # Print detection to console with HSV values
                    print(f"Detected {color_name} object at bearing {bearing_angle}° (center_x: {object_center_x}, area: {area}) - {hsv_str}")
        return detections
    
    def start_stream_display(self):
        halfway_line_y = self.frame_height // 2 - 20
        
        # Test if display is available
        try:
            window_name = 'USB Camera Feed with Object Detection'
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.setMouseCallback(window_name, self._mouse_callback)
            print("Display window created successfully!")
        except Exception as e:
            print(f"Cannot create display window: {e}")
            print("Falling back to headless mode...")
            return self.start_stream_headless()
        
        print("Starting camera stream with object detection... Press 'q' to quit.")
        print("Click on the video to see HSV values at that point.")
        
        frame_count = 0
        total_detections = {'green': 0, 'red': 0}
        
        try:
            while True:
                ret, frame = self.camera_cap.read()
                if not ret:
                    print("Error: Failed to grab frame.")
                    break
                
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
                
                # Add frame info
                cv2.putText(display_frame, f"Frame: {frame_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(display_frame, f"Detections: {len(all_detections)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Display the frame
                cv2.imshow(window_name, display_frame)
                
                frame_count += 1
                
                # Press 'q' to exit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        except KeyboardInterrupt:
            print("\nStream stopped by user")
        except Exception as e:
            print(f"Error during display: {e}")
        finally:
            cv2.destroyAllWindows()
            
        print(f"\nDetection Summary:")
        print(f"Total frames processed: {frame_count}")
        for color, count in total_detections.items():
            print(f"  {color.capitalize()} objects detected: {count}")
    
    def start_stream_headless(self):
        # Fallback headless mode
        print("Running in headless mode - saving frames instead of displaying")
        output_dir = "camera_captures"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        halfway_line_y = self.frame_height // 2 - 50
        frame_count = 0
        
        try:
            for i in range(100):  # Capture 100 frames
                ret, frame = self.camera_cap.read()
                if not ret:
                    break
                    
                frame = cv2.flip(frame, -1)
                display_frame = frame.copy()
                hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                self.current_hsv_frame = hsv_frame
                
                # Process detections
                for color_name, bounds in self.color_bounds.items():
                    if color_name == 'red':
                        mask1 = cv2.inRange(hsv_frame, bounds['lower1'], bounds['upper1'])
                        mask2 = cv2.inRange(hsv_frame, bounds['lower2'], bounds['upper2'])
                        final_mask = cv2.bitwise_or(mask1, mask2)
                    else:
                        final_mask = cv2.inRange(hsv_frame, bounds['lower'], bounds['upper'])
                    
                    final_mask = cv2.erode(final_mask, self.morph_kernel, iterations=1)
                    final_mask = cv2.dilate(final_mask, self.morph_kernel, iterations=1)
                    self._process_contours(display_frame, final_mask, color_name.capitalize(), halfway_line_y)
                
                cv2.line(display_frame, (0, halfway_line_y), (self.frame_width, halfway_line_y), (255, 255, 0), 2)
                
                if frame_count % 10 == 0:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"{output_dir}/capture_{timestamp}_{frame_count:06d}.jpg"
                    cv2.imwrite(filename, display_frame)
                    print(f"Saved: {filename}")
                
                frame_count += 1
                
        except KeyboardInterrupt:
            print("\nHeadless capture stopped")

# Create and start the camera controller
try:
    camera_controller = USBCameraControllerDisplay(cap, width, height)
    camera_controller.start_stream_display()
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

print("Program completed successfully!")
