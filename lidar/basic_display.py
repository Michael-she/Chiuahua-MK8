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

# --- Thread-Safe Shared Data Structure ---
# A deque is a thread-safe, fast way to append/pop from both ends
lidar_points = collections.deque(maxlen=2000) # Store ~5 full rotations of data
stop_event = threading.Event()

# --- LiDAR and GPIO Configuration ---
PWM_PIN = 18
SERIAL_PORT = '/dev/serial0'
BAUD_RATE = 230400
MOTOR_DUTY_CYCLE = 65  # A medium-high speed for stable 10Hz rotation

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

# --- Main Pygame Loop ---
def main():
    """Starts threads and runs the Pygame loop."""
    # Start the data reader thread
    reader = threading.Thread(target=lidar_reader_thread)
    reader.daemon = True
    reader.start()

    # Pygame setup
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("High-Performance LiDAR Scan")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 30)

    center_x, center_y = SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2

    running = True
    while running and reader.is_alive():
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False

        # --- Drawing ---
        screen.fill(BACKGROUND_COLOR)

        # Draw concentric circles for distance reference
        for r in range(1000, MAX_DISTANCE_MM + 1, 1000):
            radius_px = int(r / SCALE_FACTOR)
            pygame.draw.circle(screen, (50, 50, 50), (center_x, center_y), radius_px, 1)

        # Get a copy of the points to avoid thread issues during iteration
        points_to_draw = list(lidar_points)

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
        points_text = font.render(f'Points: {len(points_to_draw)}', True, (255, 255, 0))
        screen.blit(fps_text, (10, 10))
        screen.blit(points_text, (10, 40))

        pygame.display.flip()
        clock.tick(60) # Limit to 60 FPS

    # --- Cleanup ---
    print("Main loop finished. Stopping reader thread...")
    stop_event.set()
    reader.join()
    pygame.quit()
    print("Program finished cleanly.")

if __name__ == '__main__':
    main()
