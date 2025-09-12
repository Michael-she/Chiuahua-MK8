#!/usr/bin/env python3
"""
Taxi Driver Main Controller
Controls motor using multiprocessing for communication with motor controller slave
"""

import multiprocessing
import time
import threading
import math
import pygame
import sys
import os
from motor_controller_slave import motor_controller_process
from gyroscope_slave import gyroscope_controller_process
from LiDAR_slave import lidar_data_process
from camera_controller_multiprocessing_slave import camera_controller_process, decode_detection_data

class TaxiDriver:
    def __init__(self):
        """Initialize the Taxi Driver with multiprocessing communication"""
        # Shared variables for motor control
        self.motor_speed = multiprocessing.Value('d', 0.0)  # Speed percentage (0-100)
        self.motor_direction = multiprocessing.Value('i', 0)  # Direction (1=forward, -1=reverse, 0=stop)
        self.encoder_value = multiprocessing.Value('i', 0)  # Encoder count
        self.motor_shutdown_flag = multiprocessing.Value('i', 0)  # Motor shutdown signal
        
        # Shared variables for gyroscope/servo control
        self.target_angle = multiprocessing.Value('d', 0.0)  # Target angle in degrees
        self.current_angle = multiprocessing.Value('d', 0.0)  # Current angle in degrees (absolute, can be > 360°)
        self.servo_position = multiprocessing.Value('i', 94)  # Servo position (64-124)
        self.total_rotations = multiprocessing.Value('i', 0)  # Total rotation count
        self.gyro_shutdown_flag = multiprocessing.Value('i', 0)  # Gyroscope shutdown signal
        
        # Shared variables for LiDAR data (Cartesian coordinates)
        self.max_lidar_points = 450  # Maximum number of LiDAR points
        self.lidar_points_x = multiprocessing.Array('d', self.max_lidar_points)  # X coordinates in mm
        self.lidar_points_y = multiprocessing.Array('d', self.max_lidar_points)  # Y coordinates in mm
        self.lidar_points_count = multiprocessing.Value('i', 0)  # Number of valid points
        self.lidar_shutdown_flag = multiprocessing.Value('i', 0)  # LiDAR shutdown signal
        
        # Shared variables for camera controller data
        self.max_detections = 20  # Maximum number of object detections
        self.camera_detections = multiprocessing.Array('d', self.max_detections * 10)  # Detection data (10 fields per detection)
        self.camera_detection_count = multiprocessing.Value('i', 0)  # Number of current detections
        self.camera_shutdown_flag = multiprocessing.Value('i', 0)  # Camera shutdown signal
        
        # Controller processes
        self.motor_process = None
        self.gyro_process = None
        self.lidar_process = None
        self.camera_process = None
        
        print("Taxi Driver initialized")
    
    def start_motor_controller(self):
        """Start the motor controller process"""
        try:
            print("Starting motor controller process...")
            self.motor_process = multiprocessing.Process(
                target=motor_controller_process,
                args=(self.motor_speed, self.motor_direction, self.encoder_value, self.motor_shutdown_flag)
            )
            self.motor_process.start()
            time.sleep(1)  # Give the process time to initialize
            print("Motor controller process started")
        except Exception as e:
            print(f"Error starting motor controller: {e}")
    
    def start_gyroscope_controller(self):
        """Start the gyroscope controller process"""
        try:
            print("Starting gyroscope controller process...")
            self.gyro_process = multiprocessing.Process(
                target=gyroscope_controller_process,
                args=(self.target_angle, self.current_angle, self.servo_position, self.total_rotations, self.gyro_shutdown_flag)
            )
            self.gyro_process.start()
            time.sleep(2)  # Give the process time to initialize (gyro needs more time)
            print("Gyroscope controller process started")
        except Exception as e:
            print(f"Error starting gyroscope controller: {e}")
    
    def start_lidar_controller(self):
        """Start the LiDAR controller process"""
        try:
            print("Starting LiDAR controller process...")
            self.lidar_process = multiprocessing.Process(
                target=lidar_data_process,
                args=(self.lidar_points_x, self.lidar_points_y, self.lidar_points_count, self.lidar_shutdown_flag)
            )
            self.lidar_process.start()
            time.sleep(1)  # Give the process time to initialize
            print("LiDAR controller process started")
        except Exception as e:
            print(f"Error starting LiDAR controller: {e}")
    
    def start_camera_controller(self):
        """Start the camera controller process"""
        try:
            print("Starting camera controller process...")
            self.camera_process = multiprocessing.Process(
                target=camera_controller_process,
                args=(self.camera_detections, self.camera_detection_count, self.camera_shutdown_flag)
            )
            self.camera_process.start()
            time.sleep(2)  # Give the process time to initialize camera
            print("Camera controller process started")
        except Exception as e:
            print(f"Error starting camera controller: {e}")
    
    def start_all_controllers(self):
        """Start motor, gyroscope, LiDAR, and camera controllers"""
        self.start_motor_controller()
        self.start_gyroscope_controller()
        self.start_lidar_controller()
        self.start_camera_controller()
    
    def stop_motor_controller(self):
        """Stop the motor controller process"""
        try:
            if self.motor_process and self.motor_process.is_alive():
                print("Stopping motor controller process...")
                self.motor_shutdown_flag.value = 1
                self.motor_process.join(timeout=5)
                if self.motor_process.is_alive():
                    print("Force terminating motor controller process...")
                    self.motor_process.terminate()
                    self.motor_process.join()
                print("Motor controller process stopped")
        except Exception as e:
            print(f"Error stopping motor controller: {e}")
    
    def stop_gyroscope_controller(self):
        """Stop the gyroscope controller process"""
        try:
            if self.gyro_process and self.gyro_process.is_alive():
                print("Stopping gyroscope controller process...")
                self.gyro_shutdown_flag.value = 1
                self.gyro_process.join(timeout=5)
                if self.gyro_process.is_alive():
                    print("Force terminating gyroscope controller process...")
                    self.gyro_process.terminate()
                    self.gyro_process.join()
                print("Gyroscope controller process stopped")
        except Exception as e:
            print(f"Error stopping gyroscope controller: {e}")
    
    def stop_lidar_controller(self):
        """Stop the LiDAR controller process"""
        try:
            if self.lidar_process and self.lidar_process.is_alive():
                print("Stopping LiDAR controller process...")
                self.lidar_shutdown_flag.value = 1
                self.lidar_process.join(timeout=5)
                if self.lidar_process.is_alive():
                    print("Force terminating LiDAR controller process...")
                    self.lidar_process.terminate()
                    self.lidar_process.join()
                print("LiDAR controller process stopped")
        except Exception as e:
            print(f"Error stopping LiDAR controller: {e}")
    
    def stop_camera_controller(self):
        """Stop the camera controller process"""
        try:
            if self.camera_process and self.camera_process.is_alive():
                print("Stopping camera controller process...")
                self.camera_shutdown_flag.value = 1
                self.camera_process.join(timeout=5)
                if self.camera_process.is_alive():
                    print("Force terminating camera controller process...")
                    self.camera_process.terminate()
                    self.camera_process.join()
                print("Camera controller process stopped")
        except Exception as e:
            print(f"Error stopping camera controller: {e}")
    
    def stop_all_controllers(self):
        """Stop motor, gyroscope, LiDAR, and camera controllers"""
        self.stop_motor_controller()
        self.stop_gyroscope_controller()
        self.stop_lidar_controller()
        self.stop_camera_controller()
    
    def set_motor_speed(self, speed_percent):
        """
        Set motor speed
        
        Args:
            speed_percent (float): Speed as percentage (0-100)
        """
        try:
            speed_percent = max(0, min(100, speed_percent))
            self.motor_speed.value = speed_percent
            print(f"Motor speed set to {speed_percent}%")
        except Exception as e:
            print(f"Error setting motor speed: {e}")
    
    def set_motor_direction(self, direction):
        """
        Set motor direction
        
        Args:
            direction (int): 1 for forward, -1 for reverse, 0 for stop
        """
        try:
            if direction > 0:
                self.motor_direction.value = 1
                print("Motor direction set to FORWARD")
            elif direction < 0:
                self.motor_direction.value = -1
                print("Motor direction set to REVERSE")
            else:
                self.motor_direction.value = 0
                print("Motor STOPPED")
        except Exception as e:
            print(f"Error setting motor direction: {e}")
    
    def move_forward(self, speed_percent):
        """Move forward at specified speed"""
        self.set_motor_direction(1)
        self.set_motor_speed(speed_percent)
    
    def move_reverse(self, speed_percent):
        """Move reverse at specified speed"""
        self.set_motor_direction(-1)
        self.set_motor_speed(speed_percent)
    
    def stop_motor(self):
        """Stop the motor"""
        self.set_motor_direction(0)
        self.set_motor_speed(0)
    
    def get_encoder_reading(self):
        """
        Get current encoder reading
        
        Returns:
            int: Current encoder count
        """
        try:
            return self.encoder_value.value
        except Exception as e:
            print(f"Error reading encoder: {e}")
            return 0
    
    # Gyroscope/Servo control methods
    def set_target_angle(self, angle):
        """
        Set target angle for steering
        
        Args:
            angle (float): Target angle in degrees
        """
        try:
            self.target_angle.value = angle
            print(f"Target angle set to {angle:.1f}°")
        except Exception as e:
            print(f"Error setting target angle: {e}")
    
    def turn_left(self, degrees):
        """
        Turn left by specified degrees
        
        Args:
            degrees (float): Degrees to turn left
        """
        try:
            new_target = self.target_angle.value + degrees
            self.set_target_angle(new_target)
            print(f"Turning left by {degrees}°, new target angle: {new_target:.1f}°")
        except Exception as e:
            print(f"Error turning left: {e}")
    
    def turn_right(self, degrees):
        """
        Turn right by specified degrees
        
        Args:
            degrees (float): Degrees to turn right
        """
        try:
            new_target = self.target_angle.value - degrees
            self.set_target_angle(new_target)
            print(f"Turning right by {degrees}°, new target angle: {new_target:.1f}°")
        except Exception as e:
            print(f"Error turning right: {e}")

    def get_current_angle(self):
        """
        Get current angle from gyroscope
        
        Returns:
            float: Current angle in degrees
        """
        try:
            return self.current_angle.value
        except Exception as e:
            print(f"Error reading current angle: {e}")
            return 0.0
    
    def get_target_angle(self):
        """
        Get current target angle
        
        Returns:
            float: Target angle in degrees
        """
        try:
            return self.target_angle.value
        except Exception as e:
            print(f"Error reading target angle: {e}")
            return 0.0
    
    def get_servo_position(self):
        """
        Get current servo position
        
        Returns:
            int: Servo position (64-124)
        """
        try:
            return self.servo_position.value
        except Exception as e:
            print(f"Error reading servo position: {e}")
            return 94
    
    def get_total_rotations(self):
        """
        Get total number of complete rotations
        
        Returns:
            int: Number of complete 360° rotations (can be negative)
        """
        try:
            return self.total_rotations.value
        except Exception as e:
            print(f"Error reading total rotations: {e}")
            return 0
    
    def get_absolute_angle(self):
        """
        Get absolute angle including rotations (can exceed ±360°)
        This is the same as get_current_angle() but with a clearer name
        
        Returns:
            float: Absolute angle in degrees
        """
        return self.get_current_angle()
    
    def reset_gyroscope_offset(self):
        """
        Reset the gyroscope angle offset and rotation count
        Note: This would need to be implemented in the gyroscope process
        """
        print("Note: Gyroscope offset reset would need to be implemented with a shared flag")
        # Could add a reset flag to the multiprocessing interface if needed
    
    def get_motor_status(self):
        """
        Get current motor status
        
        Returns:
            dict: Motor status information
        """
        try:
            speed = self.motor_speed.value
            direction = self.motor_direction.value
            encoder = self.encoder_value.value
            
            direction_text = "Stopped"
            if direction == 1:
                direction_text = "Forward"
            elif direction == -1:
                direction_text = "Reverse"
            
            return {
                'speed': speed,
                'direction': direction,
                'direction_text': direction_text,
                'encoder_count': encoder,
                'status': f"{direction_text} at {speed}%" if direction != 0 else "Stopped"
            }
        except Exception as e:
            print(f"Error getting motor status: {e}")
            return {}
    
    def get_gyroscope_status(self):
        """
        Get current gyroscope and servo status
        
        Returns:
            dict: Gyroscope status information
        """
        try:
            current_angle = self.current_angle.value
            target_angle = self.target_angle.value
            servo_position = self.servo_position.value
            total_rotations = self.total_rotations.value
            angle_diff = current_angle - target_angle
            
            # Calculate the wrapped angle (within ±180°)
            wrapped_current = current_angle % 360
            if wrapped_current > 180:
                wrapped_current -= 360
            elif wrapped_current < -180:
                wrapped_current += 360
            
            return {
                'current_angle': current_angle,  # Absolute angle (can be > 360°)
                'wrapped_angle': wrapped_current,  # Angle within ±180°
                'target_angle': target_angle,
                'servo_position': servo_position,
                'angle_difference': angle_diff,
                'total_rotations': total_rotations,
                'status': f"Current: {current_angle:.1f}° (wrapped: {wrapped_current:.1f}°), Target: {target_angle:.1f}°, Servo: {servo_position}, Rotations: {total_rotations}"
            }
        except Exception as e:
            print(f"Error getting gyroscope status: {e}")
            return {}
    
    # LiDAR data access methods
    def get_lidar_points(self):
        """
        Get current filtered LiDAR points in Cartesian coordinates
        
        Returns:
            list: List of (x, y) tuples in millimeters
        """
        try:
            count = self.lidar_points_count.value
            points = []
            for i in range(count):
                x = self.lidar_points_x[i]
                y = self.lidar_points_y[i]
                points.append((x, y))
            return points
        except Exception as e:
            print(f"Error getting LiDAR points: {e}")
            return []
    
    def get_lidar_points_count(self):
        """
        Get the number of current filtered LiDAR points
        
        Returns:
            int: Number of valid LiDAR points
        """
        try:
            return self.lidar_points_count.value
        except Exception as e:
            print(f"Error getting LiDAR points count: {e}")
            return 0
    
    def get_lidar_status(self):
        """
        Get current LiDAR status information
        
        Returns:
            dict: LiDAR status information
        """
        try:
            points_count = self.lidar_points_count.value
            is_active = self.lidar_process and self.lidar_process.is_alive()
            
            return {
                'points_count': points_count,
                'is_active': is_active,
                'max_points': self.max_lidar_points,
                'status': f"Active: {is_active}, Points: {points_count}/{self.max_lidar_points}"
            }
        except Exception as e:
            print(f"Error getting LiDAR status: {e}")
            return {}
    
    # Camera data access methods
    def get_camera_detections(self):
        """
        Get current camera object detections
        
        Returns:
            list: List of detected objects with color, position, and angle information
        """
        try:
            detection_count = self.camera_detection_count.value
            return decode_detection_data(self.camera_detections, detection_count)
        except Exception as e:
            print(f"Error getting camera detections: {e}")
            return []
    
    def get_camera_detection_count(self):
        """
        Get the number of current camera detections
        
        Returns:
            int: Number of detected objects
        """
        try:
            return self.camera_detection_count.value
        except Exception as e:
            print(f"Error getting camera detection count: {e}")
            return 0
    
    def get_camera_status(self):
        """
        Get current camera status information
        
        Returns:
            dict: Camera status information
        """
        try:
            detection_count = self.camera_detection_count.value
            is_active = self.camera_process and self.camera_process.is_alive()
            
            return {
                'detection_count': detection_count,
                'is_active': is_active,
                'max_detections': self.max_detections,
                'status': f"Active: {is_active}, Detections: {detection_count}/{self.max_detections}"
            }
        except Exception as e:
            print(f"Error getting camera status: {e}")
            return {}
    
    def get_objects_by_color(self, color):
        """
        Get all detected objects of a specific color
        
        Args:
            color (str): Color to filter by ('green', 'red', 'pink')
        
        Returns:
            list: List of objects matching the specified color
        """
        try:
            all_detections = self.get_camera_detections()
            return [det for det in all_detections if det['color'].lower() == color.lower()]
        except Exception as e:
            print(f"Error getting objects by color {color}: {e}")
            return []
    
    def get_closest_object_by_angle(self, target_angle, tolerance=5):
        """
        Get the closest object to a specific angle
        
        Args:
            target_angle (float): Target angle in degrees
            tolerance (float): Angle tolerance in degrees
        
        Returns:
            dict: Closest object or None if no objects in range
        """
        try:
            all_detections = self.get_camera_detections()
            objects_in_range = []
            
            for detection in all_detections:
                if detection['bearing'] is not None:
                    angle_diff = abs(detection['bearing'] - target_angle)
                    if angle_diff <= tolerance:
                        objects_in_range.append((detection, angle_diff))
            
            if objects_in_range:
                # Return the object with the smallest angle difference
                objects_in_range.sort(key=lambda x: x[1])
                return objects_in_range[0][0]
            
            return None
        except Exception as e:
            print(f"Error getting closest object by angle: {e}")
            return None
    
    def get_closest_obstacle_distance(self):
        """
        Get the distance to the closest obstacle in front of the vehicle
        
        Returns:
            float: Distance to closest obstacle in mm, or None if no obstacles
        """
        try:
            points = self.get_lidar_points()
            if not points:
                return None
            
            # Only consider points in front of the vehicle (positive Y direction)
            # and within a reasonable forward cone (±30 degrees)
            front_points = []
            for x, y in points:
                if y > 0:  # In front of the vehicle
                    angle = math.atan2(abs(x), y)  # Angle from forward direction
                    if angle <= math.radians(30):  # Within ±30 degrees
                        distance = math.sqrt(x**2 + y**2)
                        front_points.append(distance)
            
            return min(front_points) if front_points else None
        except Exception as e:
            print(f"Error getting closest obstacle distance: {e}")
            return None
    
    def get_distance(self, angle):
        """
        Get the median distance of LiDAR points within ±2.5 degrees of the specified angle
        
        Args:
            angle (float): Target angle in degrees (0° = forward, positive = clockwise)
        
        Returns:
            float: Median distance in mm, or None if no points found in the angle range
        """
        try:
            points = self.get_lidar_points()
            if not points:
                return 9999  # No points available Large Distance
            
            # Convert target angle to radians and normalize
            target_angle_rad = math.radians(angle)
            angle_tolerance_rad = math.radians(2.5)  # ±2.5 degrees
            
            # Collect distances of points within the angle range
            distances_in_range = []
            
            for x, y in points:
                # Calculate the angle of this point
                point_angle_rad = math.atan2(x, y)  # atan2(x, y) for our coordinate system
                
                # Calculate the angular difference
                angle_diff = abs(point_angle_rad - target_angle_rad)
                
                # Handle angle wrapping (e.g., difference between 359° and 1°)
                if angle_diff > math.pi:
                    angle_diff = 2 * math.pi - angle_diff
                
                # Check if the point is within the tolerance
                if angle_diff <= angle_tolerance_rad:
                    distance = math.sqrt(x**2 + y**2)
                    distances_in_range.append(distance)
            
            # Return median distance if we have points in range
            if distances_in_range:
                distances_in_range.sort()
                n = len(distances_in_range)
                if n % 2 == 0:
                    # Even number of points - average of middle two
                    median = (distances_in_range[n//2 - 1] + distances_in_range[n//2]) / 2
                else:
                    # Odd number of points - middle value
                    median = distances_in_range[n//2]
                return median
            else:
                return 9999
                
        except Exception as e:
            print(f"Error getting distance at angle {angle}°: {e}")
            return None
    
    def get_furthest_point(self, requested_angle):
        """
        Get the 95th percentile furthest point within a 30-degree FOV (±15 degrees)
        and normalize it to the requested angle using trigonometry
        
        Args:
            requested_angle (float): Target angle in degrees (0° = forward, positive = clockwise)
        
        Returns:
            float: Normalized distance in mm, or None if no points found
        """
        try:
            points = self.get_lidar_points()
            if not points:
                return None
            
            # Convert requested angle to radians
            requested_angle_rad = math.radians(requested_angle)
            fov_half_angle_rad = math.radians(15)  # ±15 degrees FOV
            
            # Collect points within the 30-degree FOV centered on the requested angle
            points_in_fov = []
            
            for x, y in points:
                # Calculate the angle of this point
                point_angle_rad = math.atan2(x, y)  # atan2(x, y) for our coordinate system
                
                # Calculate the angular difference from the requested angle
                angle_diff = abs(point_angle_rad - requested_angle_rad)
                
                # Handle angle wrapping (e.g., difference between 359° and 1°)
                if angle_diff > math.pi:
                    angle_diff = 2 * math.pi - angle_diff
                
                # Check if the point is within the FOV
                if angle_diff <= fov_half_angle_rad:
                    distance = math.sqrt(x**2 + y**2)
                    angle_deg = math.degrees(point_angle_rad)
                    points_in_fov.append((distance, angle_deg, point_angle_rad))
            
            if not points_in_fov:
                return None
            
            # Sort by distance and get the 95th percentile furthest point
            points_in_fov.sort(key=lambda p: p[0], reverse=True)  # Sort by distance descending
            percentile_95_index = int(0.05 * len(points_in_fov))  # 95th percentile (top 5%)
            furthest_distance, furthest_angle_deg, furthest_angle_rad = points_in_fov[percentile_95_index]
            
            # Normalize the distance to the requested angle using trigonometry
            # Calculate the angle difference between the furthest point and requested angle
            angle_difference = furthest_angle_rad - requested_angle_rad
            
            # Handle angle wrapping
            if angle_difference > math.pi:
                angle_difference -= 2 * math.pi
            elif angle_difference < -math.pi:
                angle_difference += 2 * math.pi
            
            # Use cosine to project the distance onto the requested angle direction
            # This gives us the component of the distance in the requested direction
            normalized_distance = furthest_distance * math.cos(angle_difference)
            
            return abs(normalized_distance)  # Return absolute value for distance
            
        except Exception as e:
            print(f"Error getting furthest point at angle {requested_angle}°: {e}")
            return None
    
    def get_closest_point(self, requested_angle):
        """
        Get the closest point within a 30-degree FOV (±15 degrees)
        and normalize it to the requested angle using trigonometry
        
        Args:
            requested_angle (float): Target angle in degrees (0° = forward, positive = clockwise)
        
        Returns:
            float: Normalized distance in mm, or None if no points found
        """
        try:
            points = self.get_lidar_points()
            if not points:
                return None
            
            # Convert requested angle to radians
            requested_angle_rad = math.radians(requested_angle)
            fov_half_angle_rad = math.radians(15)  # ±15 degrees FOV
            
            # Collect points within the 30-degree FOV centered on the requested angle
            points_in_fov = []
            
            for x, y in points:
                # Calculate the angle of this point
                point_angle_rad = math.atan2(x, y)  # atan2(x, y) for our coordinate system
                
                # Calculate the angular difference from the requested angle
                angle_diff = abs(point_angle_rad - requested_angle_rad)
                
                # Handle angle wrapping (e.g., difference between 359° and 1°)
                if angle_diff > math.pi:
                    angle_diff = 2 * math.pi - angle_diff
                
                # Check if the point is within the FOV
                if angle_diff <= fov_half_angle_rad:
                    distance = math.sqrt(x**2 + y**2)
                    angle_deg = math.degrees(point_angle_rad)
                    points_in_fov.append((distance, angle_deg, point_angle_rad))
            
            if not points_in_fov:
                return None
            
            # Sort by distance and get the closest point
            points_in_fov.sort(key=lambda p: p[0])  # Sort by distance ascending
            closest_distance, closest_angle_deg, closest_angle_rad = points_in_fov[0]
            
            # Normalize the distance to the requested angle using trigonometry
            # Calculate the angle difference between the closest point and requested angle
            angle_difference = closest_angle_rad - requested_angle_rad
            
            # Handle angle wrapping
            if angle_difference > math.pi:
                angle_difference -= 2 * math.pi
            elif angle_difference < -math.pi:
                angle_difference += 2 * math.pi
            
            # Use cosine to project the distance onto the requested angle direction
            # This gives us the component of the distance in the requested direction
            normalized_distance = closest_distance * math.cos(angle_difference)
            
            return abs(normalized_distance)  # Return absolute value for distance
            
        except Exception as e:
            print(f"Error getting closest point at angle {requested_angle}°: {e}")
            return None
    
    def get_full_status(self):
        """
        Get complete system status
        
        Returns:
            dict: Complete system status
        """
        motor_status = self.get_motor_status()
        gyro_status = self.get_gyroscope_status()
        lidar_status = self.get_lidar_status()
        camera_status = self.get_camera_status()
        
        return {
            'motor': motor_status,
            'gyroscope': gyro_status,
            'lidar': lidar_status,
            'camera': camera_status,
            'timestamp': time.time()
        }
    
    def run_test_sequence(self):
        """Run a test sequence to demonstrate motor control"""
        print("\n=== Running Motor Test Sequence ===")
        
        # Reset encoder by getting initial reading
        initial_encoder = self.get_encoder_reading()
        print(f"Initial encoder reading: {initial_encoder}")
        
        # Test forward motion
        print("\n1. Testing FORWARD motion:")
        for speed in [25, 50, 75]:
            start_encoder = self.get_encoder_reading()
            print(f"  Forward at {speed}% - Starting encoder: {start_encoder}")
            self.move_forward(speed)
            time.sleep(3)
            end_encoder = self.get_encoder_reading()
            change = end_encoder - start_encoder
            print(f"    Encoder change: {change} counts")
        
        # Brief stop
        print("\n  Stopping...")
        self.stop_motor()
        time.sleep(2)
        
        # Test reverse motion
        print("\n2. Testing REVERSE motion:")
        for speed in [25, 50, 75]:
            start_encoder = self.get_encoder_reading()
            print(f"  Reverse at {speed}% - Starting encoder: {start_encoder}")
            self.move_reverse(speed)
            time.sleep(3)
            end_encoder = self.get_encoder_reading()
            change = end_encoder - start_encoder
            print(f"    Encoder change: {change} counts")
        
        # Final stop
        print("\n  Final stop...")
        self.stop_motor()
        final_encoder = self.get_encoder_reading()
        total_change = final_encoder - initial_encoder
        print(f"\nTest completed. Total encoder change: {total_change} counts")
    

    def interactive_control(self):
        """Interactive control interface"""
        print("\n=== Taxi Driver Interactive Control ===")
        print("Motor Commands:")
        print("  f <speed>       - Move forward at speed (0-100)")
        print("  r <speed>       - Move reverse at speed (0-100)")
        print("  s <speed>       - Set speed in current direction")
        print("  stop            - Stop motor")
        print()
        print("Steering Commands:")
        print("  angle <degrees> - Set target angle (-180 to 180)")
        print("  left <degrees>  - Turn left by degrees")
        print("  right <degrees> - Turn right by degrees")
        print("  center          - Set target to current angle")
        print("  rotations       - Show total rotations count")
        print("  absolute        - Show absolute angle (can be > 360°)")
        print()
        print("LiDAR Commands:")
        print("  lidar           - Show LiDAR status")
        print("  points          - Show LiDAR point data")
        print("  obstacle        - Show closest obstacle distance")
        print("  distance <angle> - Show median distance at angle (±2.5°)")
        print("  furthest <angle> - Show 95th percentile furthest point (±15°)")
        print("  closest <angle>  - Show closest point (±15°)")
        print()
        print("Camera Commands:")
        print("  camera          - Show camera status")
        print("  detections      - Show all object detections")
        print("  color <color>   - Show objects of specific color (green/red/pink)")
        print("  angle_obj <angle> - Show closest object to angle (±10°)")
        print()
        print("Status Commands:")
        print("  status          - Show motor and encoder status")
        print("  gyro            - Show gyroscope and servo status")
        print("  full            - Show complete system status")
        print("  encoder         - Show encoder reading")
        print("  test            - Run test sequence")
        print("  viz/visualize   - Start pygame object visualization")
        print("  quit            - Exit")
        print()
        
        try:
            while True:
                try:
                    command = input("TaxiDriver> ").strip().lower()
                    
                    if command in ['quit', 'exit', 'q']:
                        break
                    elif command in ['help', 'h', '?']:
                        print("\n=== Available Commands ===")
                        print("Motor Control:")
                        print("  f <speed>     - Move forward at speed %")
                        print("  r <speed>     - Move reverse at speed %")
                        print("  s <speed>     - Set motor speed %")
                        print("  stop          - Stop motor")
                        print("\nSteering Control:")
                        print("  angle <deg>   - Set target angle")
                        print("  left <deg>    - Turn left by degrees")
                        print("  right <deg>   - Turn right by degrees")
                        print("  center        - Center steering to current angle")
                        print("\nLiDAR Commands:")
                        print("  lidar         - Show LiDAR status")
                        print("  points        - Show LiDAR point data")
                        print("  obstacle      - Show closest obstacle distance")
                        print("  distance <angle> - Show median distance at angle (±2.5°)")
                        print("  furthest <angle> - Show 95th percentile furthest point (±15°)")
                        print("  closest <angle>  - Show closest point (±15°)")
                        print("\nCamera Commands:")
                        print("  camera        - Show camera status")
                        print("  detections    - Show all object detections")
                        print("  color <color> - Show objects of specific color")
                        print("  angle_obj <angle> - Show closest object to angle")
                        print("\nStatus Commands:")
                        print("  status        - Show motor status")
                        print("  gyro          - Show gyroscope status")
                        print("  full          - Show complete system status")
                        print("  encoder       - Show encoder reading")
                        print("  rotations     - Show total rotations")
                        print("  absolute      - Show absolute angle")
                        print("\nOther:")
                        print("  test          - Run test sequence")
                        print("  viz/visualize - Start pygame object visualization")
                        print("  help/h/?      - Show this help")
                        print("  quit/exit/q   - Exit program")
                        print()
                    elif command == 'stop':
                        self.stop_motor()
                    elif command == 'status':
                        status = self.get_motor_status()
                        print(f"Motor Status: {status.get('status', 'Unknown')}")
                        print(f"Speed: {status.get('speed', 0)}%")
                        print(f"Direction: {status.get('direction_text', 'Unknown')}")
                        print(f"Encoder: {status.get('encoder_count', 0)} counts")
                    elif command == 'gyro':
                        status = self.get_gyroscope_status()
                        print(f"Gyroscope Status: {status.get('status', 'Unknown')}")
                        print(f"Absolute Angle: {status.get('current_angle', 0):.1f}°")
                        print(f"Wrapped Angle: {status.get('wrapped_angle', 0):.1f}°")
                        print(f"Target Angle: {status.get('target_angle', 0):.1f}°")
                        print(f"Servo Position: {status.get('servo_position', 94)}")
                        print(f"Total Rotations: {status.get('total_rotations', 0)}")
                        print(f"Angle Difference: {status.get('angle_difference', 0):.1f}°")
                    elif command == 'lidar':
                        status = self.get_lidar_status()
                        print(f"LiDAR Status: {status.get('status', 'Unknown')}")
                        print(f"Points Count: {status.get('points_count', 0)}")
                        print(f"Is Active: {status.get('is_active', False)}")
                        closest = self.get_closest_obstacle_distance()
                        if closest is not None:
                            print(f"Closest Obstacle: {closest:.1f} mm ({closest/1000:.2f} m)")
                        else:
                            print("No obstacles detected in front")
                    elif command == 'points':
                        points = self.get_lidar_points()
                        count = len(points)
                        print(f"LiDAR Points ({count} total):")
                        if count > 0:
                            # Show first 10 points as example
                            for i, (x, y) in enumerate(points[:10]):
                                print(f"  Point {i+1}: ({x:.1f}, {y:.1f}) mm")
                            if count > 10:
                                print(f"  ... and {count-10} more points")
                        else:
                            print("  No points available")
                    elif command == 'obstacle':
                        closest = self.get_closest_obstacle_distance()
                        if closest is not None:
                            print(f"Closest obstacle distance: {closest:.1f} mm ({closest/1000:.2f} m)")
                        else:
                            print("No obstacles detected in forward direction")
                    elif command.startswith('distance '):
                        try:
                            angle = float(command.split()[1])
                            distance = self.get_distance(angle)
                            if distance is not None:
                                print(f"Median distance at {angle}° (±2.5°): {distance:.1f} mm ({distance/1000:.2f} m)")
                            else:
                                print(f"No LiDAR points found at {angle}° (±2.5°)")
                        except (IndexError, ValueError):
                            print("Usage: distance <angle_in_degrees>")
                            print("Example: distance 0    (forward)")
                            print("Example: distance 45   (45° clockwise)")
                            print("Example: distance -30  (30° counter-clockwise)")
                    elif command.startswith('furthest '):
                        try:
                            angle = float(command.split()[1])
                            distance = self.get_furthest_point(angle)
                            if distance is not None:
                                print(f"95th percentile furthest point at {angle}° (±15°): {distance:.1f} mm ({distance/1000:.2f} m)")
                            else:
                                print(f"No LiDAR points found at {angle}° (±15°)")
                        except (IndexError, ValueError):
                            print("Usage: furthest <angle_in_degrees>")
                            print("Example: furthest 0    (forward)")
                            print("Example: furthest 45   (45° clockwise)")
                            print("Example: furthest -30  (30° counter-clockwise)")
                    elif command.startswith('closest '):
                        try:
                            angle = float(command.split()[1])
                            distance = self.get_closest_point(angle)
                            if distance is not None:
                                print(f"Closest point at {angle}° (±15°): {distance:.1f} mm ({distance/1000:.2f} m)")
                            else:
                                print(f"No LiDAR points found at {angle}° (±15°)")
                        except (IndexError, ValueError):
                            print("Usage: closest <angle_in_degrees>")
                            print("Example: closest 0    (forward)")
                            print("Example: closest 45   (45° clockwise)")
                            print("Example: closest -30  (30° counter-clockwise)")
                    elif command == 'camera':
                        status = self.get_camera_status()
                        print(f"Camera Status: {status.get('status', 'Unknown')}")
                        print(f"Detections: {status.get('detection_count', 0)}")
                        print(f"Is Active: {status.get('is_active', False)}")
                    elif command == 'detections':
                        detections = self.get_camera_detections()
                        if detections:
                            print(f"Found {len(detections)} object(s):")
                            for i, det in enumerate(detections):
                                bearing_str = f"{det['bearing']:.1f}°" if det['bearing'] is not None else "Unknown"
                                print(f"  {i+1}. {det['color'].title()} object at {bearing_str}")
                                print(f"     Position: ({det['center_x']}, {det['center_y']})")
                                print(f"     Area: {det['area']:.0f} pixels")
                                print(f"     BBox: {det['bbox']}")
                        else:
                            print("No objects detected")
                    elif command.startswith('color '):
                        try:
                            color = command.split()[1]
                            objects = self.get_objects_by_color(color)
                            if objects:
                                print(f"Found {len(objects)} {color} object(s):")
                                for i, obj in enumerate(objects):
                                    bearing_str = f"{obj['bearing']:.1f}°" if obj['bearing'] is not None else "Unknown"
                                    print(f"  {i+1}. {color.title()} at {bearing_str} (area: {obj['area']:.0f})")
                            else:
                                print(f"No {color} objects detected")
                        except (IndexError, ValueError):
                            print("Usage: color <color_name>")
                            print("Example: color green")
                            print("Example: color red")
                            print("Example: color pink")
                    elif command.startswith('angle_obj '):
                        try:
                            angle = float(command.split()[1])
                            obj = self.get_closest_object_by_angle(angle)
                            if obj:
                                bearing_str = f"{obj['bearing']:.1f}°" if obj['bearing'] is not None else "Unknown"
                                print(f"Closest object to {angle}°:")
                                print(f"  Color: {obj['color'].title()}")
                                print(f"  Bearing: {bearing_str}")
                                print(f"  Position: ({obj['center_x']}, {obj['center_y']})")
                                print(f"  Area: {obj['area']:.0f} pixels")
                            else:
                                print(f"No objects found near {angle}° (±10°)")
                        except (IndexError, ValueError):
                            print("Usage: angle_obj <angle_in_degrees>")
                            print("Example: angle_obj 0    (forward)")
                            print("Example: angle_obj 45   (45° clockwise)")
                            print("Example: angle_obj -30  (30° counter-clockwise)")
                    elif command == 'full':
                        status = self.get_full_status()
                        motor = status.get('motor', {})
                        gyro = status.get('gyroscope', {})
                        lidar = status.get('lidar', {})
                        camera = status.get('camera', {})
                        print(f"=== Full System Status ===")
                        print(f"Motor: {motor.get('status', 'Unknown')}")
                        print(f"Encoder: {motor.get('encoder_count', 0)} counts")
                        print(f"Gyroscope: {gyro.get('status', 'Unknown')}")
                        print(f"Angle Diff: {gyro.get('angle_difference', 0):.1f}°")
                        print(f"LiDAR: {lidar.get('status', 'Unknown')}")
                        closest = self.get_closest_obstacle_distance()
                        if closest is not None:
                            print(f"Closest Obstacle: {closest:.1f} mm ({closest/1000:.2f} m)")
                        else:
                            print("No front obstacles detected")
                        print(f"Camera: {camera.get('status', 'Unknown')}")
                        detections = self.get_camera_detections()
                        if detections:
                            print(f"Objects detected: {len(detections)}")
                            for det in detections[:3]:  # Show first 3 detections
                                bearing_str = f"{det['bearing']:.1f}°" if det['bearing'] is not None else "Unknown"
                                print(f"  {det['color'].title()} at {bearing_str}")
                        else:
                            print("No objects detected")
                    elif command == 'encoder':
                        encoder = self.get_encoder_reading()
                        print(f"Encoder reading: {encoder} counts")
                    elif command == 'center':
                        current = self.get_current_angle()
                        self.set_target_angle(current)
                        print(f"Target angle centered to current angle: {current:.1f}°")
                    elif command == 'rotations':
                        rotations = self.get_total_rotations()
                        print(f"Total rotations: {rotations}")
                    elif command == 'absolute':
                        absolute_angle = self.get_absolute_angle()
                        wrapped_angle = absolute_angle % 360
                        if wrapped_angle > 180:
                            wrapped_angle -= 360
                        elif wrapped_angle < -180:
                            wrapped_angle += 360
                        print(f"Absolute angle: {absolute_angle:.1f}°")
                        print(f"Wrapped angle: {wrapped_angle:.1f}°")
                    elif command == 'test':
                        self.run_test_sequence()
                    elif command == 'viz' or command == 'visualize':
                        print("Starting object visualization...")
                        self.start_object_visualization()
                    elif command.startswith('f '):
                        try:
                            speed = float(command.split()[1])
                            self.move_forward(speed)
                        except (IndexError, ValueError):
                            print("Usage: f <speed>")
                    elif command.startswith('r '):
                        try:
                            speed = float(command.split()[1])
                            self.move_reverse(speed)
                        except (IndexError, ValueError):
                            print("Usage: r <speed>")
                    elif command.startswith('s '):
                        try:
                            speed = float(command.split()[1])
                            self.set_motor_speed(speed)
                        except (IndexError, ValueError):
                            print("Usage: s <speed>")
                    elif command.startswith('angle '):
                        try:
                            angle = float(command.split()[1])
                            self.set_target_angle(angle)
                        except (IndexError, ValueError):
                            print("Usage: angle <degrees>")
                    elif command.startswith('left '):
                        try:
                            degrees = float(command.split()[1])
                            current_target = self.get_target_angle()
                            new_target = current_target + degrees
                            self.set_target_angle(new_target)
                            print(f"Turned left {degrees}° (new target: {new_target:.1f}°)")
                        except (IndexError, ValueError):
                            print("Usage: left <degrees>")
                    elif command.startswith('right '):
                        try:
                            degrees = float(command.split()[1])
                            current_target = self.get_target_angle()
                            new_target = current_target - degrees
                            self.set_target_angle(new_target)
                            print(f"Turned right {degrees}° (new target: {new_target:.1f}°)")
                        except (IndexError, ValueError):
                            print("Usage: right <degrees>")
                    else:
                        print("Unknown command. Type 'quit' to exit.")
                        
                except KeyboardInterrupt:
                    print("\nInterrupted by user")
                    break
                except Exception as e:
                    print(f"Error processing command: {e}")
                    
        except KeyboardInterrupt:
            print("\nExiting...")
    
    def start_object_visualization(self):
        """
        Start pygame visualization showing relative positions of detected objects
        Combines camera angle data with LiDAR distance measurements
        """
        print("Starting object visualization...")
        
        # Pygame configuration
        SCREEN_WIDTH = 800
        SCREEN_HEIGHT = 800
        MAX_DISTANCE_MM = 4000  # 4 meters max display range
        SCALE_FACTOR = MAX_DISTANCE_MM / (SCREEN_WIDTH / 2 - 50)
        
        # Colors
        BACKGROUND_COLOR = (0, 0, 0)
        GRID_COLOR = (50, 50, 50)
        ROBOT_COLOR = (255, 255, 255)
        LIDAR_POINT_COLOR = (100, 200, 255)
        
        # Object colors
        OBJECT_COLORS = {
            'green': (0, 255, 0),
            'red': (255, 0, 0),
            'pink': (255, 0, 255)
        }
        
        try:
            pygame.init()
            screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            pygame.display.set_caption("TaxiDriver Object Visualization")
            clock = pygame.time.Clock()
            font = pygame.font.SysFont(None, 24)
            
            center_x, center_y = SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2
            
            print("Visualization started. Press 'q' or ESC to exit.")
            
            running = True
            while running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key in [pygame.K_ESCAPE, pygame.K_q]):
                        running = False
                
                # Clear screen
                screen.fill(BACKGROUND_COLOR)
                
                # Draw grid circles for distance reference
                for r in range(1000, MAX_DISTANCE_MM + 1, 1000):
                    radius_px = int(r / SCALE_FACTOR)
                    pygame.draw.circle(screen, GRID_COLOR, (center_x, center_y), radius_px, 1)
                    # Distance labels
                    label = font.render(f"{r//1000}m", True, GRID_COLOR)
                    screen.blit(label, (center_x + radius_px - 15, center_y - 10))
                
                # Draw angle reference lines
                for angle_deg in range(0, 360, 30):
                    angle_rad = math.radians(angle_deg)
                    end_x = center_x + int((SCREEN_WIDTH // 2 - 50) * math.sin(angle_rad))
                    end_y = center_y - int((SCREEN_WIDTH // 2 - 50) * math.cos(angle_rad))
                    pygame.draw.line(screen, GRID_COLOR, (center_x, center_y), (end_x, end_y), 1)
                
                # Draw forward direction indicator
                pygame.draw.line(screen, (0, 255, 0), (center_x, center_y), (center_x, center_y - 100), 3)
                forward_label = font.render("FORWARD", True, (0, 255, 0))
                screen.blit(forward_label, (center_x - 30, center_y - 130))
                
                # Draw LiDAR points
                lidar_points = self.get_lidar_points()
                for x, y in lidar_points:
                    distance = math.sqrt(x**2 + y**2)
                    if distance <= MAX_DISTANCE_MM:
                        screen_x = center_x + int(x / SCALE_FACTOR)
                        screen_y = center_y - int(y / SCALE_FACTOR)
                        pygame.draw.circle(screen, LIDAR_POINT_COLOR, (screen_x, screen_y), 1)
                
                # Get camera detections and correlate with LiDAR distance
                camera_detections = self.get_camera_detections()
                detected_objects = []
                
                for detection in camera_detections:
                    if detection['bearing'] is not None:
                        angle = detection['bearing']
                        
                        # Get LiDAR distance at this angle
                        lidar_distance = self.get_distance(angle)
                        
                        if lidar_distance is not None and lidar_distance != 9999 and lidar_distance <= MAX_DISTANCE_MM:
                            # Convert polar coordinates to screen coordinates
                            angle_rad = math.radians(angle)
                            x_mm = lidar_distance * math.sin(angle_rad)
                            y_mm = lidar_distance * math.cos(angle_rad)
                            
                            screen_x = center_x + int(x_mm / SCALE_FACTOR)
                            screen_y = center_y - int(y_mm / SCALE_FACTOR)
                            
                            # Get color for this object
                            color = OBJECT_COLORS.get(detection['color'], (255, 255, 255))
                            
                            # Draw object
                            pygame.draw.circle(screen, color, (screen_x, screen_y), 8)
                            pygame.draw.circle(screen, (255, 255, 255), (screen_x, screen_y), 8, 2)
                            
                            # Draw angle line from robot to object
                            pygame.draw.line(screen, color, (center_x, center_y), (screen_x, screen_y), 1)
                            
                            # Label with distance and angle
                            label_text = f"{detection['color']}: {lidar_distance:.0f}mm, {angle:.1f}°"
                            label = font.render(label_text, True, color)
                            label_x = screen_x + 10
                            label_y = screen_y - 10
                            
                            # Keep labels on screen
                            if label_x + label.get_width() > SCREEN_WIDTH:
                                label_x = screen_x - label.get_width() - 10
                            if label_y < 0:
                                label_y = screen_y + 20
                            
                            screen.blit(label, (label_x, label_y))
                            
                            detected_objects.append({
                                'color': detection['color'],
                                'angle': angle,
                                'distance': lidar_distance,
                                'screen_pos': (screen_x, screen_y)
                            })
                
                # Draw robot (center circle)
                pygame.draw.circle(screen, ROBOT_COLOR, (center_x, center_y), 10)
                pygame.draw.circle(screen, (0, 0, 0), (center_x, center_y), 10, 2)
                
                # Status information
                status_y = 10
                status_texts = [
                    f"Objects detected: {len(detected_objects)}",
                    f"LiDAR points: {self.get_lidar_points_count()}",
                    f"Camera detections: {self.get_camera_detection_count()}",
                ]
                
                for text in status_texts:
                    label = font.render(text, True, (255, 255, 255))
                    screen.blit(label, (10, status_y))
                    status_y += 25
                
                # Object summary
                if detected_objects:
                    summary_y = status_y + 10
                    summary_label = font.render("Detected Objects:", True, (255, 255, 255))
                    screen.blit(summary_label, (10, summary_y))
                    summary_y += 25
                    
                    for obj in detected_objects[:5]:  # Show first 5 objects
                        obj_text = f"  {obj['color']}: {obj['distance']:.0f}mm @ {obj['angle']:.1f}°"
                        obj_label = font.render(obj_text, True, OBJECT_COLORS.get(obj['color'], (255, 255, 255)))
                        screen.blit(obj_label, (10, summary_y))
                        summary_y += 20
                
                # Instructions
                instruction_text = "Press 'q' or ESC to exit"
                instruction_label = font.render(instruction_text, True, (200, 200, 200))
                screen.blit(instruction_label, (10, SCREEN_HEIGHT - 30))
                
                pygame.display.flip()
                clock.tick(30)  # 30 FPS
                
        except Exception as e:
            print(f"Error in visualization: {e}")
        finally:
            pygame.quit()
            print("Visualization stopped")
    
    def cleanup(self):
        """Clean up resources"""
        print("Cleaning up Taxi Driver...")
        self.stop_motor()
        time.sleep(0.5)
        self.stop_all_controllers()

def main():
    """Main function"""
    taxi = None
    try:
        print("=== Taxi Driver 2.0 ===")
        taxi = TaxiDriver()
        
        # Check command line arguments
        if len(sys.argv) > 1 and sys.argv[1] in ['viz', 'visualize', 'visual']:
            print("Starting in visualization mode...")
            taxi.start_all_controllers()
            time.sleep(5)  # Wait for controllers to initialize
            print("Starting object visualization...")
            taxi.start_object_visualization()
        else:
            print("Starting in interactive mode...")
            print("Use 'viz' command to start visualization")
            taxi.start_all_controllers()
            time.sleep(3)  # Give controllers time to initialize
            taxi.interactive_control()
        
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if taxi:
            taxi.cleanup()

if __name__ == "__main__":
    main()