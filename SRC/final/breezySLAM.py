# run_slam_final.py
# Final version with correct API usage for BreezySLAM

import threading
import collections
import time
import math
import pygame
import numpy as np
import serial
import RPi.GPIO as GPIO
import struct

# --- Import Your Hardware Drivers and SLAM ---
from n20_motor import N20Motor
import board
import busio
from adafruit_bno08x.i2c import BNO08X_I2C
from adafruit_bno08x import BNO_REPORT_ROTATION_VECTOR

from breezyslam.algorithms import RMHC_SLAM
from breezyslam.sensors import Laser

# --- Hardware and SLAM Configuration ---
WHEEL_DIAMETER_M = 0.020
MOTOR_FWD_PIN = 20
MOTOR_REV_PIN = 16
MOTOR_ENCODER_A_PIN = 27 # Using a safe pin
MOTOR_PULSES_PER_REVOLUTION = 200
LIDAR_SERIAL_PORT = '/dev/serial0'
LIDAR_BAUD_RATE = 230400
LIDAR_PWM_PIN = 18
LIDAR_MOTOR_DUTY_CYCLE = 65
MAP_SIZE_PIXELS = 800
MAP_SIZE_METERS = 20
LIDAR_MAX_RANGE_MM = 3000

# --- Thread-Safe Shared Data ---
lidar_points_polar = collections.deque(maxlen=450)
stop_event = threading.Event()
yaw_lock = threading.Lock()
current_yaw_deg = 0.0
yaw_offset = 0.0

# --- ALL THREADS AND HELPER FUNCTIONS ARE UNCHANGED ---
# (Omitting them here for brevity, paste your working versions in)
def gyro_reader_thread():
    # ... (Your working gyro_reader_thread code) ...
    global current_yaw_deg, yaw_offset
    try:
        i2c = busio.I2C(board.SCL, board.SDA)
        bno = BNO08X_I2C(i2c)
        bno.enable_feature(BNO_REPORT_ROTATION_VECTOR)
    except Exception as e:
        print(f"FATAL: Could not initialize BNO08x: {e}")
        stop_event.set(); return

    print("Calibrating gyroscope... Do not move the robot.")
    time.sleep(1.5)
    try:
        quat_initial = bno.quaternion
        yaw_offset = calculate_yaw(quat_initial)
        print(f"Gyro calibrated. Initial yaw offset: {yaw_offset:.2f} degrees.")
    except Exception as e:
        print(f"WARNING: Could not calibrate gyro: {e}. Using zero offset.")
        yaw_offset = 0.0

    while not stop_event.is_set():
        try:
            quat = bno.quaternion
            raw_yaw = calculate_yaw(quat)
            with yaw_lock:
                calibrated_yaw = raw_yaw - yaw_offset
                current_yaw_deg = (calibrated_yaw + 180) % 360 - 180
        except (RuntimeError, KeyError):
            time.sleep(0.01)
            continue
        time.sleep(0.01)

def calculate_yaw(quaternion):
    # ... (Your working calculate_yaw code) ...
    q_i, q_j, q_k, q_real = quaternion
    siny_cosp = 2 * (q_real * q_k + q_i * q_j)
    cosy_cosp = 1 - 2 * (q_j * q_j + q_k * q_k)
    return math.degrees(math.atan2(siny_cosp, cosy_cosp))

def lidar_reader_thread():
    # ... (Your working lidar_reader_thread code) ...
    try:
        ser = serial.Serial(LIDAR_SERIAL_PORT, LIDAR_BAUD_RATE, timeout=0.1)
    except serial.SerialException as e:
        print(f"FATAL: Could not open LiDAR serial port: {e}")
        stop_event.set(); return

    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
    GPIO.setup(LIDAR_PWM_PIN, GPIO.OUT)
    motor_pwm = GPIO.PWM(LIDAR_PWM_PIN, 1000)
    motor_pwm.start(LIDAR_MOTOR_DUTY_CYCLE)
    print("LiDAR motor started. Reader thread running.")

    packet_buffer = bytearray()
    while not stop_event.is_set():
        try:
            data = ser.read(4096)
            if not data: time.sleep(0.001); continue
            packet_buffer.extend(data)
            while len(packet_buffer) >= 47:
                header_index = packet_buffer.find(b'\x54')
                if header_index == -1: packet_buffer.clear(); break
                if header_index > 0: packet_buffer = packet_buffer[header_index:]
                if len(packet_buffer) < 47: break
                packet = packet_buffer[:47]
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
                    if confidence > 180:
                        angle_step = angle_diff / 11 if 11 > 0 else 0
                        current_angle_deg = start_angle_deg + i * angle_step
                        lidar_points_polar.append((math.radians(current_angle_deg % 360), distance_mm))
                packet_buffer = packet_buffer[47:]
        except Exception as e:
            print(f"Error in LiDAR reader thread: {e}"); break
    motor_pwm.stop()
    GPIO.cleanup(LIDAR_PWM_PIN)
    ser.close()
    print("LiDAR reader thread has stopped.")
    
class RealOdometry:
    # ... (Your working RealOdometry class code) ...
    def __init__(self, motor, initial_x_m, initial_y_m):
        self.motor = motor
        self.x_m = initial_x_m
        self.y_m = initial_y_m
        self.theta_deg = 0.0
        self.wheel_circumference_m = WHEEL_DIAMETER_M * math.pi
        self.last_revolutions = self.motor.get_revolutions()

    def get_odometry(self):
        current_revolutions = self.motor.get_revolutions()
        with yaw_lock:
            self.theta_deg = current_yaw_deg
        delta_revolutions = current_revolutions - self.last_revolutions
        distance_traveled_m = delta_revolutions * self.wheel_circumference_m
        theta_rad = math.radians(self.theta_deg)
        self.x_m += distance_traveled_m * math.cos(theta_rad)
        self.y_m += distance_traveled_m * math.sin(theta_rad)
        self.last_revolutions = current_revolutions
        return [self.x_m, self.y_m, self.theta_deg]

# --- Main Application (with corrected API calls) ---
# --- Main Application (with correct scan padding logic) ---
def main():
    pygame.init()
    screen = pygame.display.set_mode((MAP_SIZE_PIXELS, MAP_SIZE_PIXELS))
    pygame.display.set_caption("BreezySLAM - Use Arrow Keys to Move")
    font = pygame.font.SysFont(None, 30)
    map_surface = pygame.Surface((MAP_SIZE_PIXELS, MAP_SIZE_PIXELS))

    map_bytes = bytearray(MAP_SIZE_PIXELS * MAP_SIZE_PIXELS)
    
    motor = N20Motor(fwd_pin=MOTOR_FWD_PIN, rev_pin=MOTOR_REV_PIN, enc_a_pin=MOTOR_ENCODER_A_PIN, pulses_per_revolution=MOTOR_PULSES_PER_REVOLUTION)
    
    try:
        gyro_thread = threading.Thread(target=gyro_reader_thread)
        gyro_thread.daemon = True
        gyro_thread.start()
        lidar_thread = threading.Thread(target=lidar_reader_thread)
        lidar_thread.daemon = True
        lidar_thread.start()

        print("Waiting for hardware to initialize and calibrate...")
        time.sleep(3)

        # We promise a 360-point scan. This is the correct size.
        SCAN_SIZE = 360
        slam = RMHC_SLAM(Laser(SCAN_SIZE, 7, 0, 0), MAP_SIZE_PIXELS, MAP_SIZE_METERS)
        odometry = RealOdometry(motor, MAP_SIZE_METERS / 2, MAP_SIZE_METERS / 2)
        motor_speed = 0
        
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE): running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP: motor_speed = 40
                    elif event.key == pygame.K_DOWN: motor_speed = -40
                if event.type == pygame.KEYUP:
                    if event.key in [pygame.K_UP, pygame.K_DOWN]: motor_speed = 0
            motor.set_speed(motor_speed)

            current_odometry = odometry.get_odometry()

            # --- SCAN PADDING LOGIC ---
            
            # 1. Create an empty scan of the correct size
            scan_distances_mm = [0] * SCAN_SIZE
            
            points_to_process = list(lidar_points_polar)
            for angle_rad, distance_mm in points_to_process:
                if 0 < distance_mm < LIDAR_MAX_RANGE_MM:
                    # 2. Calculate the correct index (0-359) for this measurement
                    index = int(math.degrees(angle_rad)) % SCAN_SIZE
                    # 3. Place the measurement in its slot
                    scan_distances_mm[index] = distance_mm

            # 4. Deliver the perfectly-sized list to SLAM
            slam.update(scan_distances_mm, current_odometry)

            x_m, y_m, theta_deg = slam.getpos()
            slam.getmap(map_bytes)

            gray_map = np.frombuffer(map_bytes, dtype=np.uint8).reshape((MAP_SIZE_PIXELS, MAP_SIZE_PIXELS))
            display_map = np.stack([gray_map]*3, axis=-1)

            pygame.surfarray.blit_array(map_surface, display_map)
            screen.blit(map_surface, (0, 0))
            
            robot_x_pix = int(x_m / MAP_SIZE_METERS * MAP_SIZE_PIXELS)
            robot_y_pix = int(y_m / MAP_SIZE_METERS * MAP_SIZE_PIXELS)
            pygame.draw.circle(screen, (255, 0, 0), (robot_x_pix, robot_y_pix), 5)
            theta_rad = math.radians(theta_deg)
            end_x = robot_x_pix + 15 * math.cos(theta_rad)
            end_y = robot_y_pix + 15 * math.sin(theta_rad)
            pygame.draw.line(screen, (255, 0, 0), (robot_x_pix, robot_y_pix), (end_x, end_y), 2)
            
            pose_text = f"Pose: ({x_m:.2f}m, {y_m:.2f}m, {theta_deg:.1f}°)"
            text_surface = font.render(pose_text, True, (255, 255, 0))
            screen.blit(text_surface, (10, 10))
            pygame.display.flip()
            
    finally:
        # ... The rest of the finally block is correct ...
        print("Stopping all processes and cleaning up...")
        stop_event.set()
        motor.stop()
        motor.cleanup()
        
        if 'lidar_thread' in locals() and lidar_thread.is_alive(): lidar_thread.join()
        if 'gyro_thread' in locals() and gyro_thread.is_alive(): gyro_thread.join()

        if 'slam' in locals():
            with open('slam_map_final.pgm', 'wb') as f:
                pgm_header = f"P5\n{MAP_SIZE_PIXELS} {MAP_SIZE_PIXELS}\n255\n".encode()
                f.write(pgm_header)
                f.write(map_bytes)
            print("Map saved to slam_map_final.pgm")
        
        pygame.quit()
        print("Program finished cleanly.")

        
if __name__ == '__main__':
    # Paste your other functions (gyro, lidar, odometry class) here
    main()