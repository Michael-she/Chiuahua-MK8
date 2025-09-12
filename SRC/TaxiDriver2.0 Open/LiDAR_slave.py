# lidar_main.py
# This script is the final, high-performance implementation based on all our findings.
# It uses a checksum-agnostic reader in a separate thread and a fast Pygame renderer.

import serial
import RPi.GPIO as GPIO
import time
import struct
import math
import pygame
import threading
import collections
import multiprocessing
import ctypes

# --- Thread-Safe Shared Data Structure ---
# A deque is a thread-safe, fast way to append/pop from both ends
lidar_points = collections.deque(maxlen=450) # Store ~5 full rotations of data
filtered_points = collections.deque(maxlen=450) # Store filtered points without lonely points
stop_event = threading.Event()

# --- Filtering Configuration ---
LONELY_POINT_THRESHOLD_MM = 50  # Points without neighbors within 50mm are considered lonely
MAX_ANGLE_FROM_ZERO_DEG = 120  # Filter out points more than 120 degrees from 0 degrees

# --- LiDAR and GPIO Configuration ---
PWM_PIN = 18
SERIAL_PORT = '/dev/serial0'
BAUD_RATE = 230400
MOTOR_DUTY_CYCLE = 65  # A medium-high speed for stable 10Hz rotation

# --- Filtering Function ---
def filter_lonely_points(points):
    """Filter out lonely points and points outside the angular range."""
    if len(points) < 2:
        return list(points)
    
    points_list = list(points)
    filtered = []
    
    for i, (angle_rad, distance) in enumerate(points_list):
        # Convert angle to degrees for easier comparison
        angle_deg = math.degrees(angle_rad) % 360
        
        # Normalize angle to [-180, 180] range for easier calculation from 0 degrees
        if angle_deg > 180:
            angle_deg -= 360
        
        # Filter out points that are more than 120 degrees away from 0 degrees
        if abs(angle_deg) > MAX_ANGLE_FROM_ZERO_DEG:
            continue
        
        # Check for lonely points (existing logic)
        has_neighbor = False
        
        # Convert current point to Cartesian coordinates
        x1 = distance * math.sin(angle_rad)
        y1 = distance * math.cos(angle_rad)
        
        # Check distance to all other points
        for j, (angle2, distance2) in enumerate(points_list):
            if i == j:
                continue
                
            # Convert comparison point to Cartesian coordinates
            x2 = distance2 * math.sin(angle2)
            y2 = distance2 * math.cos(angle2)
            
            # Calculate Euclidean distance between points
            distance_between = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            
            if distance_between <= LONELY_POINT_THRESHOLD_MM:
                has_neighbor = True
                break
        
        # Only keep points that have at least one neighbor and are within angular range
        if has_neighbor:
            filtered.append((angle_rad, distance))
    
    return filtered

# --- Pygame Configuration ---
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 800
# We will scale the visualization so 4 meters (4000mm) is the edge of the screen
MAX_DISTANCE_MM = 4000
SCALE_FACTOR = MAX_DISTANCE_MM / (SCREEN_WIDTH / 2 - 20) # a little padding
BACKGROUND_COLOR = (0, 0, 0)
POINT_COLOR = (100, 200, 255)

# --- LiDAR Reader Thread ---
def lidar_reader_thread():
    """Reads data from LiDAR and puts valid points into the shared deque."""
    global packets_processed_per_second
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1)
    except serial.SerialException as e:
        print(f"FATAL: Could not open serial port: {e}")
        stop_event.set(); return

    # Start Motor
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
    GPIO.setup(PWM_PIN, GPIO.OUT)
    motor_pwm = GPIO.PWM(PWM_PIN, 1000)
    motor_pwm.start(MOTOR_DUTY_CYCLE)
    print("Motor started. Reader thread running.")

    packet_buffer = bytearray()
    while not stop_event.is_set():
        try:
            data = ser.read(4096)
            if not data:
                time.sleep(0.001)
                continue
            packet_buffer.extend(data)
            
            while len(packet_buffer) >= 47:
                header_index = packet_buffer.find(0x54)
                if header_index == -1:
                    packet_buffer.clear(); break
                if header_index > 0:
                    packet_buffer = packet_buffer[header_index:]
                if len(packet_buffer) < 47: break
                
                # We assume the packet is valid (checksum-agnostic)
                packet = packet_buffer[:47]
                
                # Parse the data from the packet
                start_angle = struct.unpack('<H', bytes(packet[4:6]))[0]
                end_angle = struct.unpack('<H', bytes(packet[42:44]))[0]
                start_angle_deg = start_angle / 100.0
                end_angle_deg = end_angle / 100.0
                angle_diff = end_angle_deg - start_angle_deg
                if angle_diff < 0: angle_diff += 360.0

                for i in range(12):
                    offset = 6 + (i * 3)
                    distance_mm = struct.unpack('<H', bytes(packet[offset:offset+2]))[0]
                    confidence = packet[offset+2]
                    
                    if distance_mm > 0 and confidence > 180:
                        angle_step = angle_diff / 11 if 11 > 0 else 0
                        current_angle_deg = start_angle_deg + i * angle_step
                        # Add to our shared data structure for the main thread to use
                        lidar_points.append((math.radians(current_angle_deg % 360), distance_mm))
                
                packet_buffer = packet_buffer[47:]
        except Exception as e:
            print(f"Error in reader thread: {e}")
            break

    motor_pwm.stop()
    GPIO.cleanup()
    ser.close()
    print("Reader thread has stopped.")

# --- Point Filtering Thread ---
def point_filter_thread():
    """Continuously filters lonely points from the raw LiDAR data."""
    print("Point filter thread running.")
    
    while not stop_event.is_set():
        try:
            # Get current points and filter them
            if len(lidar_points) > 0:
                filtered = filter_lonely_points(lidar_points)
                
                # Update the filtered points deque
                filtered_points.clear()
                filtered_points.extend(filtered)
            
            # Run filtering at a reasonable rate (10Hz)
            time.sleep(0.1)
            
        except Exception as e:
            print(f"Error in filter thread: {e}")
            time.sleep(0.1)
    
    print("Filter thread has stopped.")

# --- Multiprocessing LiDAR Process ---
def lidar_data_process(shared_points_x, shared_points_y, shared_points_count, shutdown_flag):
    """
    LiDAR data collection process for multiprocessing communication.
    Shares filtered Cartesian coordinates with the main taxi driver process.
    """
    print("LiDAR data process started")
    
    # Local data structures for this process
    local_lidar_points = collections.deque(maxlen=450)
    local_filtered_points = collections.deque(maxlen=450)
    local_stop_event = threading.Event()
    
    def local_lidar_reader_thread():
        """Local LiDAR reader thread for the multiprocessing version"""
        try:
            ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1)
        except serial.SerialException as e:
            print(f"FATAL: Could not open serial port: {e}")
            local_stop_event.set()
            return

        # Start Motor
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        GPIO.setup(PWM_PIN, GPIO.OUT)
        motor_pwm = GPIO.PWM(PWM_PIN, 1000)
        motor_pwm.start(MOTOR_DUTY_CYCLE)
        print("LiDAR motor started in data process")

        packet_buffer = bytearray()
        while not local_stop_event.is_set() and shutdown_flag.value == 0:
            try:
                data = ser.read(4096)
                if not data:
                    time.sleep(0.001)
                    continue
                packet_buffer.extend(data)
                
                while len(packet_buffer) >= 47:
                    header_index = packet_buffer.find(0x54)
                    if header_index == -1:
                        packet_buffer.clear()
                        break
                    if header_index > 0:
                        packet_buffer = packet_buffer[header_index:]
                    if len(packet_buffer) < 47:
                        break
                    
                    packet = packet_buffer[:47]
                    
                    # Parse the data from the packet
                    start_angle = struct.unpack('<H', bytes(packet[4:6]))[0]
                    end_angle = struct.unpack('<H', bytes(packet[42:44]))[0]
                    start_angle_deg = start_angle / 100.0
                    end_angle_deg = end_angle / 100.0
                    angle_diff = end_angle_deg - start_angle_deg
                    if angle_diff < 0:
                        angle_diff += 360.0

                    for i in range(12):
                        offset = 6 + (i * 3)
                        distance_mm = struct.unpack('<H', bytes(packet[offset:offset+2]))[0]
                        confidence = packet[offset+2]
                        
                        if distance_mm > 0 and confidence > 180:
                            angle_step = angle_diff / 11 if 11 > 0 else 0
                            current_angle_deg = start_angle_deg + i * angle_step
                            local_lidar_points.append((math.radians(current_angle_deg % 360), distance_mm))
                    
                    packet_buffer = packet_buffer[47:]
            except Exception as e:
                print(f"Error in LiDAR reader thread: {e}")
                break

        motor_pwm.stop()
        GPIO.cleanup()
        ser.close()
        print("LiDAR reader thread stopped")

    def local_filter_thread():
        """Local filtering thread for the multiprocessing version"""
        while not local_stop_event.is_set() and shutdown_flag.value == 0:
            try:
                if len(local_lidar_points) > 0:
                    filtered = filter_lonely_points(local_lidar_points)
                    local_filtered_points.clear()
                    local_filtered_points.extend(filtered)
                    
                    # Convert to Cartesian and update shared memory
                    cartesian_points = []
                    for angle_rad, distance_mm in filtered:
                        x = distance_mm * math.sin(angle_rad)
                        y = distance_mm * math.cos(angle_rad)
                        cartesian_points.append((x, y))
                    
                    # Update shared arrays (limit to array size)
                    num_points = min(len(cartesian_points), len(shared_points_x))
                    shared_points_count.value = num_points
                    
                    for i in range(num_points):
                        shared_points_x[i] = cartesian_points[i][0]
                        shared_points_y[i] = cartesian_points[i][1]
                
                time.sleep(0.1)  # 10Hz update rate
                
            except Exception as e:
                print(f"Error in LiDAR filter thread: {e}")
                time.sleep(0.1)

    # Start local threads
    reader_thread = threading.Thread(target=local_lidar_reader_thread)
    filter_thread = threading.Thread(target=local_filter_thread)
    
    reader_thread.daemon = True
    filter_thread.daemon = True
    
    reader_thread.start()
    filter_thread.start()
    
    # Keep the process alive until shutdown
    try:
        while shutdown_flag.value == 0:
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    
    print("LiDAR data process shutting down...")
    local_stop_event.set()
    reader_thread.join(timeout=2)
    filter_thread.join(timeout=2)
    print("LiDAR data process stopped")

# --- Main Pygame Loop ---
def main():
    """Starts threads and runs the Pygame loop."""
    # Start the data reader thread
    reader = threading.Thread(target=lidar_reader_thread)
    reader.daemon = True
    reader.start()
    
    # Start the point filtering thread
    filter_thread = threading.Thread(target=point_filter_thread)
    filter_thread.daemon = True
    filter_thread.start()

    # Pygame setup
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("High-Performance LiDAR Scan")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 30)
    
    center_x, center_y = SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2

    running = True
    while running and reader.is_alive() and filter_thread.is_alive():
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False

        # --- Drawing ---
        screen.fill(BACKGROUND_COLOR)
        
        # Draw concentric circles for distance reference
        for r in range(1000, MAX_DISTANCE_MM + 1, 1000):
            radius_px = int(r / SCALE_FACTOR)
            pygame.draw.circle(screen, (50, 50, 50), (center_x, center_y), radius_px, 1)
        
        # Draw angular filtering boundaries (120 degrees from 0)
        max_radius = int(MAX_DISTANCE_MM / SCALE_FACTOR)
        # Left boundary (-120 degrees)
        angle_left = math.radians(-MAX_ANGLE_FROM_ZERO_DEG)
        x_left = center_x + int(max_radius * math.sin(angle_left))
        y_left = center_y - int(max_radius * math.cos(angle_left))
        pygame.draw.line(screen, (255, 100, 100), (center_x, center_y), (x_left, y_left), 2)
        
        # Right boundary (+120 degrees)
        angle_right = math.radians(MAX_ANGLE_FROM_ZERO_DEG)
        x_right = center_x + int(max_radius * math.sin(angle_right))
        y_right = center_y - int(max_radius * math.cos(angle_right))
        pygame.draw.line(screen, (255, 100, 100), (center_x, center_y), (x_right, y_right), 2)
        
        # Draw center line (0 degrees)
        pygame.draw.line(screen, (100, 255, 100), (center_x, center_y), (center_x, center_y - max_radius), 2)

        # Get a copy of the filtered points to avoid thread issues during iteration
        points_to_draw = list(filtered_points)
        raw_points_count = len(lidar_points)
        
        for angle_rad, distance_mm in points_to_draw:
            if distance_mm > MAX_DISTANCE_MM: continue
            
            # Convert polar (angle, distance) to Cartesian (x, y)
            x = distance_mm / SCALE_FACTOR * math.sin(angle_rad)
            y = distance_mm / SCALE_FACTOR * math.cos(angle_rad)
            
            # Pygame's y-axis is inverted, so we subtract y from the center
            screen_x = center_x + int(x)
            screen_y = center_y - int(y)

            # Draw a small circle for each point
            pygame.draw.circle(screen, POINT_COLOR, (screen_x, screen_y), 2)
        
        # Display performance stats
        fps_text = font.render(f'FPS: {int(clock.get_fps())}', True, (255, 255, 0))
        raw_points_text = font.render(f'Raw Points: {raw_points_count}', True, (255, 255, 0))
        filtered_points_text = font.render(f'Filtered Points: {len(points_to_draw)}', True, (255, 255, 0))
        screen.blit(fps_text, (10, 10))
        screen.blit(raw_points_text, (10, 40))
        screen.blit(filtered_points_text, (10, 70))

        pygame.display.flip()
        clock.tick(60) # Limit to 60 FPS

    # --- Cleanup ---
    print("Main loop finished. Stopping threads...")
    stop_event.set()
    reader.join()
    filter_thread.join()
    pygame.quit()
    print("Program finished cleanly.")

if __name__ == '__main__':
    main()
