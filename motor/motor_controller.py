# breezy.py - The Complete, Fully Assembled Master Script

import threading
import collections
import time
import math
import pygame
import numpy as np
import serial
import RPi.GPIO as GPIO
import struct
import random

from n20_motor import N20Motor
import board
import busio
from adafruit_bno08x.i2c import BNO08X_I2C
from adafruit_bno08x import BNO_REPORT_ROTATION_VECTOR

# --- CONFIGURATION ---
# Drive Motor
WHEEL_DIAMETER_M = 0.020
MOTOR_FWD_PIN = 20
MOTOR_REV_PIN = 16
MOTOR_ENCODER_A_PIN = 13
MOTOR_PULSES_PER_REVOLUTION = 200
DRIVE_MOTOR_SPEED = 40

# Steering Servo
SERVO_PIN = 17
SERVO_ZERO_POINT_DEG = 82
STEERING_ANGLE_DEG = 30

# LIDAR and Filtering
LIDAR_SERIAL_PORT = '/dev/serial0'
LIDAR_BAUD_RATE = 230400
LIDAR_PWM_PIN = 18
LIDAR_MOTOR_DUTY_CYCLE = 65
ISOLATION_DISTANCE_MM = 60.0
ISOLATION_MIN_NEIGHBORS = 3

# SLAM and Map
MAP_SIZE_METERS = 20
MAP_RESOLUTION_M_PER_PIX = 0.05
MAP_SIZE_PIXELS = int(MAP_SIZE_METERS / MAP_RESOLUTION_M_PER_PIX)
LIDAR_MAX_RANGE_M = 3.0

# Particle Filter
NUM_PARTICLES = 100
ODOMETRY_MOTION_NOISE = 0.01 # meters
ODOMETRY_TURN_NOISE = 0.01 # radians

# --- Thread-Safe Shared Data ---
lidar_points_polar = collections.deque(maxlen=500)
stop_event = threading.Event()
yaw_lock = threading.Lock()
current_yaw_deg = 0.0
yaw_offset = 0.0

# --- THREADS AND HELPER FUNCTIONS ---
def gyro_reader_thread():
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
    q_i, q_j, q_k, q_real = quaternion
    siny_cosp = 2 * (q_real * q_k + q_i * q_j)
    cosy_cosp = 1 - 2 * (q_j * q_j + q_k * q_k)
    return math.degrees(math.atan2(siny_cosp, cosy_cosp))

def lidar_reader_thread():
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
    try:
        motor_pwm.stop()
        GPIO.cleanup(LIDAR_PWM_PIN)
        ser.close()
    except Exception as e:
        print(f"Ignoring harmless error during LiDAR thread cleanup: {e}")
    print("LiDAR reader thread has stopped.")

def filter_isolated_points(cartesian_points, min_neighbors, isolation_radius):
    if not cartesian_points: return []
    grid_cell_size = isolation_radius
    grid = collections.defaultdict(list)
    for i, point in enumerate(cartesian_points):
        cell = (int(point[0] // grid_cell_size), int(point[1] // grid_cell_size))
        grid[cell].append(i)
    filtered_points = []
    isolation_radius_sq = isolation_radius ** 2
    for i, p1 in enumerate(cartesian_points):
        neighbor_count = 0
        p1_cell = (int(p1[0] // grid_cell_size), int(p1[1] // grid_cell_size))
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                check_cell = (p1_cell[0] + dx, p1_cell[1] + dy)
                if check_cell in grid:
                    for neighbor_idx in grid[check_cell]:
                        p2 = cartesian_points[neighbor_idx]
                        dist_sq = (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2
                        if dist_sq < isolation_radius_sq:
                            neighbor_count += 1
        if neighbor_count >= min_neighbors:
            filtered_points.append(p1)
    return filtered_points

# --- CLASSES ---
class Steering:
    def __init__(self, pin, zero_point):
        self.pin = pin
        self.zero_point = zero_point
        self.last_signal_time = 0
        self.signal_release_delay = 0.25 # seconds
        GPIO.setup(self.pin, GPIO.OUT)
        self.pwm = GPIO.PWM(self.pin, 50)
        self.pwm.start(0)
        self.set_angle(0)

    def set_angle(self, relative_angle):
        target_angle = self.zero_point + relative_angle
        target_angle = max(0, min(180, target_angle))
        duty_cycle = 2.0 + (target_angle / 18.0)
        self.pwm.ChangeDutyCycle(duty_cycle)
        self.last_signal_time = time.time()

    def release_if_idle(self):
        if time.time() - self.last_signal_time > self.signal_release_delay:
            self.pwm.ChangeDutyCycle(0)

    def cleanup(self):
        self.pwm.stop()

class Odometry:
    def __init__(self, motor):
        self.motor = motor
        self.wheel_circumference_m = WHEEL_DIAMETER_M * math.pi
        with yaw_lock: # Initialize with current yaw
            self.last_theta_rad = math.radians(-current_yaw_deg)
        self.last_revolutions = self.motor.get_revolutions()

    def get_motion_delta(self):
        current_revolutions = self.motor.get_revolutions()
        local_current_yaw_deg = 0.0
        with yaw_lock:
            local_current_yaw_deg = current_yaw_deg
        
        current_theta_rad = math.radians(-local_current_yaw_deg)
        
        delta_revolutions = current_revolutions - self.last_revolutions
        delta_dist_m = delta_revolutions * self.wheel_circumference_m
        
        delta_theta_rad = current_theta_rad - self.last_theta_rad
        if delta_theta_rad > math.pi: delta_theta_rad -= 2 * math.pi
        if delta_theta_rad < -math.pi: delta_theta_rad += 2 * math.pi
        
        self.last_revolutions = current_revolutions
        self.last_theta_rad = current_theta_rad
        
        return (delta_dist_m, delta_theta_rad)

class ParticleFilter:
    def __init__(self, num_particles, map_size_m):
        self.num_particles = num_particles
        self.particles = np.zeros((num_particles, 3))
        self.particles[:, 0] = map_size_m / 2
        self.particles[:, 1] = map_size_m / 2
        self.weights = np.ones(num_particles) / num_particles

    def predict(self, motion_delta):
        delta_dist, delta_theta = motion_delta
        noisy_dist = delta_dist + np.random.randn(self.num_particles) * ODOMETRY_MOTION_NOISE
        noisy_theta = delta_theta + np.random.randn(self.num_particles) * ODOMETRY_TURN_NOISE
        
        self.particles[:, 2] += noisy_theta # Update angle first
        self.particles[:, 0] += noisy_dist * np.cos(self.particles[:, 2])
        self.particles[:, 1] += noisy_dist * np.sin(self.particles[:, 2])

    def weigh(self, scan_data, occupancy_map):
        for i, particle in enumerate(self.particles):
            self.weights[i] = occupancy_map.score_scan(particle, scan_data)
        
        total_weight = np.sum(self.weights)
        if total_weight > 0:
            self.weights /= total_weight
        else:
            self.weights.fill(1.0 / self.num_particles)

    def resample(self):
        indices = np.random.choice(np.arange(self.num_particles), size=self.num_particles, p=self.weights)
        self.particles = self.particles[indices]
        self.weights.fill(1.0 / self.num_particles)

    def get_pose(self):
        mean_pose = np.mean(self.particles, axis=0)
        return tuple(mean_pose)

class OccupancyGridSLAM:
    def __init__(self):
        self.map = np.zeros((MAP_SIZE_PIXELS, MAP_SIZE_PIXELS), dtype=np.int8)

    def to_map_coords(self, x_m, y_m):
        x_pix = int(x_m / MAP_RESOLUTION_M_PER_PIX)
        y_pix = int(y_m / MAP_RESOLUTION_M_PER_PIX)
        return x_pix, y_pix

    def update(self, robot_pose_m_rad, scan_data):
        rx, ry, rtheta = robot_pose_m_rad
        for angle_rad, distance_mm in scan_data:
            dist_m = distance_mm / 1000.0
            if 0.1 < dist_m < LIDAR_MAX_RANGE_M:
                scan_angle_abs = rtheta + angle_rad
                wall_x = rx + dist_m * math.cos(scan_angle_abs)
                wall_y = ry + dist_m * math.sin(scan_angle_abs)
                wall_x_pix, wall_y_pix = self.to_map_coords(wall_x, wall_y)
                if 0 <= wall_x_pix < MAP_SIZE_PIXELS and 0 <= wall_y_pix < MAP_SIZE_PIXELS:
                    self.map[wall_y_pix, wall_x_pix] = 1

    def score_scan(self, particle_pose, scan_data):
        px, py, ptheta = particle_pose
        score = 1.0
        for angle_rad, distance_mm in scan_data:
            dist_m = distance_mm / 1000.0
            if 0.1 < dist_m < LIDAR_MAX_RANGE_M:
                scan_angle_abs = ptheta + angle_rad
                hit_x = px + dist_m * math.cos(scan_angle_abs)
                hit_y = py + dist_m * math.sin(scan_angle_abs)
                hit_x_pix, hit_y_pix = self.to_map_coords(hit_x, hit_y)
                if 0 <= hit_x_pix < MAP_SIZE_PIXELS and 0 <= hit_y_pix < MAP_SIZE_PIXELS:
                    if self.map[hit_y_pix, hit_x_pix] == 1:
                        score += 5.0
                    elif self.map[hit_y_pix, hit_x_pix] == 0:
                        score += 1.0
                else:
                    score *= 0.1
        return score

    def get_map_for_display(self):
        display_map = np.zeros((MAP_SIZE_PIXELS, MAP_SIZE_PIXELS, 3), dtype=np.uint8)
        display_map[self.map == 1] = [0, 0, 0]
        display_map[self.map == 0] = [128, 128, 128]
        return display_map
        
# --- Main Application ---
def main():
    pygame.init()
    screen = pygame.display.set_mode((MAP_SIZE_PIXELS, MAP_SIZE_PIXELS))
    pygame.display.set_caption("Particle Filter SLAM")
    font = pygame.font.SysFont(None, 24)
    map_surface = pygame.Surface((MAP_SIZE_PIXELS, MAP_SIZE_PIXELS))
    
    motor = N20Motor(fwd_pin=MOTOR_FWD_PIN, rev_pin=MOTOR_REV_PIN, enc_a_pin=MOTOR_ENCODER_A_PIN, pulses_per_revolution=MOTOR_PULSES_PER_REVOLUTION)
    steering = Steering(pin=SERVO_PIN, zero_point=SERVO_ZERO_POINT_DEG)

    try:
        motor.reset_encoder()

        gyro_thread = threading.Thread(target=gyro_reader_thread)
        gyro_thread.daemon = True; gyro_thread.start()
        lidar_thread = threading.Thread(target=lidar_reader_thread)
        lidar_thread.daemon = True; lidar_thread.start()
        print("Waiting for hardware to initialize and calibrate...")
        time.sleep(3)
        
        slam = OccupancyGridSLAM()
        odometry = Odometry(motor)
        particle_filter = ParticleFilter(NUM_PARTICLES, MAP_SIZE_METERS)
        motor_speed = 0
        steering_angle = 0
        
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE): running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_w: motor_speed = DRIVE_MOTOR_SPEED
                    elif event.key == pygame.K_s: motor_speed = -DRIVE_MOTOR_SPEED
                    elif event.key == pygame.K_a: steering_angle = -STEERING_ANGLE_DEG
                    elif event.key == pygame.K_d: steering_angle = STEERING_ANGLE_DEG
                if event.type == pygame.KEYUP:
                    if event.key in [pygame.K_w, pygame.K_s]: motor_speed = 0
                    elif event.key in [pygame.K_a, pygame.K_d]: steering_angle = 0
            
            motor.set_speed(motor_speed)
            steering.set_angle(steering_angle)
            steering.release_if_idle()

            # --- SLAM PIPELINE ---
            motion_delta = odometry.get_motion_delta()
            particle_filter.predict(motion_delta)

            points_to_process = list(lidar_points_polar)
            cartesian_points = []
            for angle_rad, distance_mm in points_to_process:
                if 0 < distance_mm < (LIDAR_MAX_RANGE_M * 1000):
                    x = distance_mm * math.sin(angle_rad)
                    y = distance_mm * math.cos(angle_rad)
                    cartesian_points.append((x, y))
            filtered_cartesian_points = filter_isolated_points(cartesian_points, ISOLATION_MIN_NEIGHBORS, ISOLATION_DISTANCE_MM)
            filtered_polar_points = []
            for x, y in filtered_cartesian_points:
                dist_mm = math.hypot(x, y)
                angle_rad = math.atan2(x, y)
                filtered_polar_points.append((angle_rad, dist_mm))

            if filtered_polar_points:
                particle_filter.weigh(filtered_polar_points, slam)
                particle_filter.resample()

            current_pose = particle_filter.get_pose()
            slam.update(current_pose, filtered_polar_points)

            # --- Drawing Logic ---
            display_map_data = slam.get_map_for_display()
            pygame.surfarray.blit_array(map_surface, display_map_data)
            screen.blit(map_surface, (0, 0))
            
            x_m, y_m, theta_rad = current_pose
            robot_x_pix, robot_y_pix = slam.to_map_coords(x_m, y_m)
            pygame.draw.circle(screen, (255, 0, 0), (robot_x_pix, robot_y_pix), 5)
            end_x = robot_x_pix + 15 * math.cos(theta_rad)
            end_y = robot_y_pix + 15 * math.sin(theta_rad)
            pygame.draw.line(screen, (255, 0, 0), (robot_x_pix, robot_y_pix), (end_x, end_y), 2)
            
            theta_deg = math.degrees(theta_rad)
            pose_text = f"Pose: ({x_m:.2f}m, {y_m:.2f}m, {theta_deg:.1f}°)"
            text_surface = font.render(pose_text, True, (255, 255, 0))
            screen.blit(text_surface, (10, 10))
            pygame.display.flip()
            
    finally:
        print("Stopping all processes and cleaning up...")
        stop_event.set()
        motor.stop()
        motor.cleanup()
        if 'lidar_thread' in locals() and lidar_thread.is_alive(): lidar_thread.join()
        if 'gyro_thread' in locals() and gyro_thread.is_alive(): gyro_thread.join()
        if 'slam' in locals():
            with open('slam_map_particle_filter.pgm', 'wb') as f:
                pgm_map = np.full(slam.map.shape, 205, dtype=np.uint8)
                pgm_map[slam.map == 1] = 0
                pgm_header = f"P5\n{MAP_SIZE_PIXELS} {MAP_SIZE_PIXELS}\n255\n".encode()
                f.write(pgm_header)
                f.write(pgm_map.tobytes())
            print("Map saved to slam_map_particle_filter.pgm")
        pygame.quit()
        print("Program finished cleanly.")

if __name__ == '__main__':
    main()