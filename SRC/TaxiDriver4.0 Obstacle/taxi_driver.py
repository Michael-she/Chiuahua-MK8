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
import pigpio
import cv2
import numpy as np
from datetime import datetime
from motor_controller_slave import motor_controller_process
from gyroscope_slave import gyroscope_controller_process
from LiDAR_slave import lidar_data_process
from camera_controller_multiprocessing_slave import camera_controller_process, decode_detection_data

# Servo configuration for pin 26
SERVO_PIN_26 = 26           # GPIO pin for additional servo (BCM numbering)
SERVO_MIN_PULSE = 500      # Minimum pulse width in microseconds (1ms)
SERVO_MAX_PULSE = 2500      # Maximum pulse width in microseconds (2ms)
SERVO_CENTER_POSITION = 90  # Center position in degrees
SERVO_MIN_ANGLE = 0         # Minimum servo angle in degrees
SERVO_MAX_ANGLE = 180       # Maximum servo angle in degrees

# Button configuration for pin 23
BUTTON_PIN_23 = 23          # GPIO pin for button (BCM numbering)
BUTTON_DEBOUNCE_TIME = 0.05 # Debounce time in seconds (50ms)
# Note: 5V is applied to pin 23 when button is pressed (HIGH = pressed, LOW = released)

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
        self.forward_offset_angle = 0.0  # Forward offset angle for relative angle calculations
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
        
        # Servo control for pin 26
        self.servo_pin_26 = None
        self.servo_angle = 0  # Current servo angle
        self._servo_lock = threading.Lock()
        
        # Button control for pin 23
        self.button_pin_23 = None
        self.button_last_state = 0  # Assume button is not pressed initially (LOW when released, HIGH when pressed)
        self.button_last_time = 0   # Last time button state was checked
        self._button_lock = threading.Lock()
        
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
        
        # Set forward offset angle after gyroscope is initialized
        print("Setting forward offset angle...")
        time.sleep(1)  # Allow gyroscope to stabilize
        self.set_forward_offset_angle()
    
    def set_forward_offset_angle(self):
        """Set the forward offset angle to the current gyroscope angle for relative angle calculations"""
        try:
            current_angle = self.get_current_angle()
            self.forward_offset_angle = current_angle
            print(f"Forward offset angle set to: {self.forward_offset_angle:.1f}°")
        except Exception as e:
            print(f"Error setting forward offset angle: {e}")
            self.forward_offset_angle = 0.0
    
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
    
    def wait_for_turn_completion(self, tolerance=4.0):
        self.set_motor_speed(100)
        angle_diff = abs(self.get_current_angle() - self.get_target_angle())
        while angle_diff > tolerance:
            time.sleep(0.1)
            angle_diff = abs(self.get_current_angle() - self.get_target_angle())
            print(f"waiting for turn completion: Current: {self.get_current_angle():.1f}°, Target: {self.get_target_angle():.1f}°, Diff: {angle_diff:.1f}°")
        self.set_motor_speed(99)
        return

    # Gyroscope/Servo control methods
    def set_target_angle(self, angle):
        """
        Set target angle for steering
        
        Args:
            angle (float): Target angle in degrees
        """
        try:
            self.target_angle.value = angle + self.forward_offset_angle
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
            return self.current_angle.value - self.forward_offset_angle
        except Exception as e:
            print(f"Error reading current angle: {e}")
            return 0.0

    def get_forward_offset_angle(self):
        try:
            return self.forward_offset_angle
        except Exception as e:
            print(f"Error reading initial angle: {e}")
            return 0.0
    
    def reset_forward_offset_angle(self):
        """Reset the forward offset angle to the current gyroscope angle"""
        try:
            current_angle = self.get_current_angle()
            self.forward_offset_angle = current_angle
            print(f"Forward offset angle reset to: {self.forward_offset_angle:.1f}°")
        except Exception as e:
            print(f"Error resetting forward offset angle: {e}")
            self.forward_offset_angle = 0.0
    
    def get_relative_angle(self):
        """
        Get the current angle relative to the forward offset angle
        This gives the angle relative to the car's starting orientation
        
        Returns:
            float: Relative angle in degrees (-180 to 180)
        """
        try:
            current_angle = self.get_current_angle()
            relative_angle = current_angle - self.forward_offset_angle
            
            # Normalize to -180 to 180 range
            while relative_angle > 180:
                relative_angle -= 360
            while relative_angle < -180:
                relative_angle += 360
                
            return relative_angle
        except Exception as e:
            print(f"Error calculating relative angle: {e}")
            return 0.0
    
    def get_target_angle(self):
        """
        Get current target angle
        
        Returns:
            float: Target angle in degrees
        """
        try:
            return self.target_angle.value - self.forward_offset_angle
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
    
    def reset_gyroscope_offset(self):
        """
        Reset the gyroscope angle offset and rotation count
        Note: This would need to be implemented in the gyroscope process
        """
        print("Note: Gyroscope offset reset would need to be implemented with a shared flag")
        # Could add a reset flag to the multiprocessing interface if needed
    
    # Servo control methods for pin 26
    def initialize_servo_pin_26(self):
        """
        Initialize the servo on pin 26
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Initialize pigpio connection
            self.servo_pin_26 = pigpio.pi()
            if not self.servo_pin_26.connected:
                print("Failed to connect to pigpio daemon for servo control")
                return False
            
            # Set initial servo position to center
            self.set_servo_angle_pin_26(SERVO_CENTER_POSITION)
            print(f"Servo on pin 26 initialized at center position ({SERVO_CENTER_POSITION}°)")
            return True
            
        except Exception as e:
            print(f"Error initializing servo on pin 26: {e}")
            return False
    
    def set_servo_angle_pin_26(self, angle):
        """
        Set the angle of the servo connected to pin 26
        
        Args:
            angle (float): Target angle in degrees (0-180)
        """
        try:
            if self.servo_pin_26 is None:
                if not self.initialize_servo_pin_26():
                    print("Cannot set servo angle - initialization failed")
                    return
            
            # Clamp angle to valid range
            angle = max(SERVO_MIN_ANGLE, min(SERVO_MAX_ANGLE, angle))
            
            with self._servo_lock:
                self.servo_angle = angle
            
            # Convert angle to pulse width (same method as gyroscope_slave.py)
            pulse_width = self._angle_to_pulse_width(angle)
            
            # Set servo position using pigpio
            self.servo_pin_26.set_servo_pulsewidth(SERVO_PIN_26, pulse_width)
            print(f"Servo on pin 26 set to {angle}° (pulse width: {pulse_width}μs)")
            
        except Exception as e:
            print(f"Error setting servo angle on pin 26: {e}")
    
    def get_servo_angle_pin_26(self):
        """
        Get the current angle of the servo connected to pin 26
        
        Returns:
            float: Current servo angle in degrees
        """
        try:
            with self._servo_lock:
                return self.servo_angle
        except Exception as e:
            print(f"Error getting servo angle on pin 26: {e}")
            return SERVO_CENTER_POSITION
    
    def _angle_to_pulse_width(self, angle):
        """
        Convert servo angle (0-180 degrees) to pulse width in microseconds
        
        Args:
            angle (float): Servo angle in degrees (0-180)
            
        Returns:
            int: Pulse width in microseconds (1000-2000)
        """
        # Map angle range (0-180) to pulse width range (1000-2000 microseconds)
        normalized_angle = angle / (SERVO_MAX_ANGLE - SERVO_MIN_ANGLE)
        pulse_width = SERVO_MIN_PULSE + normalized_angle * (SERVO_MAX_PULSE - SERVO_MIN_PULSE)
        return int(pulse_width)
    
    def center_servo_pin_26(self):
        """Center the servo on pin 26 to 90 degrees"""
        self.set_servo_angle_pin_26(SERVO_CENTER_POSITION)
    
    def cleanup_servo_pin_26(self):
        """Clean up servo resources for pin 26"""
        try:
            if self.servo_pin_26 is not None:
                # Center servo before shutdown
                center_pulse = self._angle_to_pulse_width(SERVO_CENTER_POSITION)
                self.servo_pin_26.set_servo_pulsewidth(SERVO_PIN_26, center_pulse)
                time.sleep(0.5)
                
                # Turn off servo signal
                self.servo_pin_26.set_servo_pulsewidth(SERVO_PIN_26, 0)
                
                # Disconnect from pigpio daemon
                self.servo_pin_26.stop()
                self.servo_pin_26 = None
                print("Servo on pin 26 cleaned up")
                
        except Exception as e:
            print(f"Error cleaning up servo on pin 26: {e}")
    
    # Button control methods for pin 23
    def initialize_button_pin_23(self):
        """
        Initialize the button on pin 23 (5V applied when pressed)
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Initialize pigpio connection if not already done
            if self.button_pin_23 is None:
                self.button_pin_23 = pigpio.pi()
                if not self.button_pin_23.connected:
                    print("Failed to connect to pigpio daemon for button control")
                    return False
            
            # Set pin as input with pull-down resistor (since 5V is applied when pressed)
            self.button_pin_23.set_mode(BUTTON_PIN_23, pigpio.INPUT)
            self.button_pin_23.set_pull_up_down(BUTTON_PIN_23, pigpio.PUD_DOWN)
            
            # Initialize button state
            with self._button_lock:
                self.button_last_state = self.button_pin_23.read(BUTTON_PIN_23)
                self.button_last_time = time.time()
            
            print(f"Button on pin 23 initialized with pull-down resistor (5V when pressed)")
            return True
            
        except Exception as e:
            print(f"Error initializing button on pin 23: {e}")
            return False
    
    def get_button_state_pin_23(self):
        """
        Get the current state of the button connected to pin 23
        
        Returns:
            int: 1 if button is pressed (HIGH), 0 if button is not pressed (LOW)
                 -1 if error or not initialized
        """
        try:
            if self.button_pin_23 is None:
                if not self.initialize_button_pin_23():
                    print("Cannot read button state - initialization failed")
                    return -1
            
            # Read current button state
            current_state = self.button_pin_23.read(BUTTON_PIN_23)
            return current_state
            
        except Exception as e:
            print(f"Error reading button state on pin 23: {e}")
            return -1
    
    def is_button_pressed_pin_23(self):
        """
        Check if the button connected to pin 23 is currently pressed
        
        Returns:
            bool: True if button is pressed, False if not pressed or error
        """
        state = self.get_button_state_pin_23()
        return state == 1  # Button is pressed when pin reads HIGH (5V applied)
    
    def get_button_state_debounced_pin_23(self):
        """
        Get debounced button state to avoid false readings from electrical noise
        
        Returns:
            int: Debounced button state (1=pressed, 0=not pressed, -1=error)
        """
        try:
            current_time = time.time()
            current_state = self.get_button_state_pin_23()
            
            if current_state == -1:
                return -1
            
            with self._button_lock:
                # Check if enough time has passed since last state change
                if current_time - self.button_last_time >= BUTTON_DEBOUNCE_TIME:
                    if current_state != self.button_last_state:
                        # State has changed and debounce time has passed
                        self.button_last_state = current_state
                        self.button_last_time = current_time
                        return current_state
                    else:
                        # State hasn't changed, return current stable state
                        return current_state
                else:
                    # Not enough time has passed, return last stable state
                    return self.button_last_state
            
        except Exception as e:
            print(f"Error reading debounced button state on pin 23: {e}")
            return -1
    
    def is_button_pressed_debounced_pin_23(self):
        """
        Check if button is pressed using debounced reading
        
        Returns:
            bool: True if button is pressed (debounced), False otherwise
        """
        state = self.get_button_state_debounced_pin_23()
        return state == 1  # Button is pressed when pin reads HIGH (5V applied)
    
    def wait_for_button_press_pin_23(self, timeout=None):
        """
        Wait for button to be pressed (blocking)
        
        Args:
            timeout (float): Maximum time to wait in seconds (None for infinite)
            
        Returns:
            bool: True if button was pressed, False if timeout occurred
        """
        try:
            start_time = time.time()
            
            print(f"Waiting for button press on pin 23{f' (timeout: {timeout}s)' if timeout else ''}...")
            
            while True:
                if self.is_button_pressed_debounced_pin_23():
                    print("Button pressed!")
                    return True
                
                # Check timeout
                if timeout and (time.time() - start_time) >= timeout:
                    print("Button wait timeout")
                    return False
                
                time.sleep(0.01)  # Small delay to prevent excessive CPU usage
                
        except KeyboardInterrupt:
            print("Button wait interrupted by user")
            return False
        except Exception as e:
            print(f"Error waiting for button press: {e}")
            return False
    
    def wait_for_button_release_pin_23(self, timeout=None):
        """
        Wait for button to be released (blocking)
        
        Args:
            timeout (float): Maximum time to wait in seconds (None for infinite)
            
        Returns:
            bool: True if button was released, False if timeout occurred
        """
        try:
            start_time = time.time()
            
            print(f"Waiting for button release on pin 23{f' (timeout: {timeout}s)' if timeout else ''}...")
            
            while True:
                if not self.is_button_pressed_debounced_pin_23():
                    print("Button released!")
                    return True
                
                # Check timeout
                if timeout and (time.time() - start_time) >= timeout:
                    print("Button release wait timeout")
                    return False
                
                time.sleep(0.01)  # Small delay to prevent excessive CPU usage
                
        except KeyboardInterrupt:
            print("Button wait interrupted by user")
            return False
        except Exception as e:
            print(f"Error waiting for button release: {e}")
            return False
    
    def get_button_status_pin_23(self):
        """
        Get comprehensive button status information
        
        Returns:
            dict: Button status information
        """
        try:
            raw_state = self.get_button_state_pin_23()
            debounced_state = self.get_button_state_debounced_pin_23()
            is_pressed = self.is_button_pressed_pin_23()
            is_pressed_debounced = self.is_button_pressed_debounced_pin_23()
            
            with self._button_lock:
                last_time = self.button_last_time
                last_state = self.button_last_state
            
            return {
                'raw_state': raw_state,
                'debounced_state': debounced_state,
                'is_pressed': is_pressed,
                'is_pressed_debounced': is_pressed_debounced,
                'last_state': last_state,
                'last_time': last_time,
                'initialized': self.button_pin_23 is not None,
                'status': f"{'Pressed' if is_pressed_debounced else 'Released'} (raw: {'HIGH' if raw_state == 1 else 'LOW' if raw_state == 0 else 'ERROR'})"
            }
            
        except Exception as e:
            print(f"Error getting button status: {e}")
            return {'error': str(e)}
    
    def cleanup_button_pin_23(self):
        """Clean up button resources for pin 23"""
        try:
            if self.button_pin_23 is not None:
                # Reset pin to default state (turn off pull-down resistor)
                self.button_pin_23.set_pull_up_down(BUTTON_PIN_23, pigpio.PUD_OFF)
                
                # Disconnect from pigpio daemon
                self.button_pin_23.stop()
                self.button_pin_23 = None
                print("Button on pin 23 cleaned up")
                
        except Exception as e:
            print(f"Error cleaning up button on pin 23: {e}")
    
    # Camera testing methods
    def test_camera_pixel_rgb(self, x=320, y=240):
        """
        Test camera by capturing a frame and getting RGB values at specified pixel coordinates
        Uses the same camera stream as the block detection (temporarily pauses camera controller)
        
        Args:
            x (int): X coordinate (default: center of 640px width)
            y (int): Y coordinate (default: center of 480px height)
            
        Returns:
            tuple: (r, g, b, success) where r,g,b are RGB values and success is boolean
        """
        # Check if camera controller is running
        camera_was_running = self.camera_process and self.camera_process.is_alive()
        
        if camera_was_running:
            print("Temporarily pausing camera controller for testing...")
            self.stop_camera_controller()
            time.sleep(1)  # Give time for camera to be released
        
        try:
            # Try to open camera (same as camera controller)
            cap = None
            for attempt in range(3):  # Try 3 times
                cap = cv2.VideoCapture(0)
                if cap.isOpened():
                    break
                cap.release()
                time.sleep(0.5)
            
            if not cap.isOpened():
                print("Failed to open camera for testing")
                return (0, 0, 0, False)
            
            # Set camera resolution (same as in camera controller)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            # Get actual dimensions
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"Camera test resolution: {width}x{height}")
            
            # Validate coordinates
            x = max(0, min(width - 1, x))
            y = max(0, min(height - 1, y))
            
            # Capture frame
            print("Capturing test frame...")
            ret, frame = cap.read()
            
            if not ret or frame is None:
                print("Failed to capture frame")
                cap.release()
                return (0, 0, 0, False)
            
            # Flip frame (same as in camera controller)
            frame = cv2.flip(frame, -1)
            
            # Get pixel RGB values (OpenCV uses BGR format)
            bgr_pixel = frame[y, x]  # Note: numpy indexing is [row, col] = [y, x]
            b, g, r = bgr_pixel  # Convert BGR to RGB
            
            print(f"Pixel at ({x}, {y}): RGB({r}, {g}, {b})")
            
            # Save test frame with marker at test pixel
            cv2.circle(frame, (x, y), 5, (0, 255, 255), 2)  # Yellow circle
            cv2.putText(frame, f"RGB({r},{g},{b})", (x+10, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            # Save to camera_captures folder
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"camera_captures/test_pixel_{timestamp}.jpg"
            
            # Create directory if it doesn't exist
            os.makedirs("camera_captures", exist_ok=True)
            cv2.imwrite(filename, frame)
            print(f"Test frame saved as: {filename}")
            
            cap.release()
            result = (int(r), int(g), int(b), True)
            
        except Exception as e:
            print(f"Error testing camera pixel: {e}")
            if 'cap' in locals() and cap is not None:
                cap.release()
            result = (0, 0, 0, False)
        
        finally:
            # Restart camera controller if it was running
            if camera_was_running:
                print("Restarting camera controller...")
                time.sleep(0.5)  # Brief pause before restart
                self.start_camera_controller()
        
        return result
    
    def test_camera_center_pixel(self):
        """Test camera by getting RGB values at the center pixel"""
        return self.test_camera_pixel_rgb(320, 240)
    
    def test_camera_multiple_pixels(self, coordinates=None):
        """
        Test camera by getting RGB values at multiple pixel coordinates
        Uses the same camera stream as the block detection (temporarily pauses camera controller)
        
        Args:
            coordinates (list): List of (x, y) tuples. If None, uses default test points
            
        Returns:
            list: List of dictionaries with pixel info
        """
        if coordinates is None:
            # Default test coordinates (center and corners)
            coordinates = [
                (320, 240),  # Center
                (160, 120),  # Top-left quadrant center
                (480, 120),  # Top-right quadrant center
                (160, 360),  # Bottom-left quadrant center
                (480, 360),  # Bottom-right quadrant center
            ]
        
        # Check if camera controller is running
        camera_was_running = self.camera_process and self.camera_process.is_alive()
        
        if camera_was_running:
            print("Temporarily pausing camera controller for multiple pixel testing...")
            self.stop_camera_controller()
            time.sleep(1)  # Give time for camera to be released
        
        results = []
        cap = None
        
        try:
            # Try to open camera once for all tests
            for attempt in range(3):  # Try 3 times
                cap = cv2.VideoCapture(0)
                if cap.isOpened():
                    break
                cap.release()
                time.sleep(0.5)
            
            if not cap.isOpened():
                print("Failed to open camera for multiple pixel testing")
                return []
            
            # Set camera resolution (same as in camera controller)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            # Get actual dimensions
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            print(f"Testing {len(coordinates)} pixel locations...")
            
            for i, (x, y) in enumerate(coordinates):
                print(f"\nTesting pixel {i+1}/{len(coordinates)} at ({x}, {y})")
                
                # Validate coordinates
                x = max(0, min(width - 1, x))
                y = max(0, min(height - 1, y))
                
                # Capture frame
                ret, frame = cap.read()
                
                if ret and frame is not None:
                    # Flip frame (same as in camera controller)
                    frame = cv2.flip(frame, -1)
                    
                    # Get pixel RGB values (OpenCV uses BGR format)
                    bgr_pixel = frame[y, x]  # Note: numpy indexing is [row, col] = [y, x]
                    b, g, r = bgr_pixel  # Convert BGR to RGB
                    
                    print(f"Pixel at ({x}, {y}): RGB({r}, {g}, {b})")
                    
                    results.append({
                        'coordinates': (x, y),
                        'rgb': (int(r), int(g), int(b)),
                        'success': True,
                        'brightness': (r + g + b) / 3
                    })
                else:
                    print(f"Failed to capture frame for pixel ({x}, {y})")
                    results.append({
                        'coordinates': (x, y),
                        'rgb': (0, 0, 0),
                        'success': False,
                        'brightness': 0
                    })
                
                time.sleep(0.2)  # Small delay between captures
            
        except Exception as e:
            print(f"Error in multiple pixel testing: {e}")
            
        finally:
            if cap is not None:
                cap.release()
            
            # Restart camera controller if it was running
            if camera_was_running:
                print("Restarting camera controller...")
                time.sleep(0.5)  # Brief pause before restart
                self.start_camera_controller()
        
        # Print summary
        print(f"\n=== Camera Test Summary ===")
        for i, result in enumerate(results):
            coord = result['coordinates']
            rgb = result['rgb']
            brightness = result['brightness']
            status = "✓" if result['success'] else "✗"
            print(f"{status} Pixel {i+1} ({coord[0]}, {coord[1]}): RGB{rgb}, Brightness: {brightness:.1f}")
        
        return results
    
    def test_camera_color_detection(self):
        """
        Test camera color detection capabilities by analyzing captured frame
        Uses the same camera stream as the block detection (temporarily pauses camera controller)
        
        Returns:
            dict: Analysis results including color statistics
        """
        # Check if camera controller is running
        camera_was_running = self.camera_process and self.camera_process.is_alive()
        
        if camera_was_running:
            print("Temporarily pausing camera controller for color detection test...")
            self.stop_camera_controller()
            time.sleep(1)  # Give time for camera to be released
        
        try:
            # Try to open camera (same as camera controller)
            cap = None
            for attempt in range(3):  # Try 3 times
                cap = cv2.VideoCapture(0)
                if cap.isOpened():
                    break
                cap.release()
                time.sleep(0.5)
            
            if not cap.isOpened():
                print("Failed to open camera for color detection test")
                return {}
            
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            print("Capturing frame for color analysis...")
            ret, frame = cap.read()
            
            if not ret or frame is None:
                print("Failed to capture frame for color analysis")
                cap.release()
                return {}
            
            # Flip frame (same as in camera controller)
            frame = cv2.flip(frame, -1)
            
            # Convert to HSV for color analysis (same as camera controller)
            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # Color bounds from camera controller (exact same values)
            color_bounds = {
                'green': {'lower': np.array([35, 80, 30]), 'upper': np.array([150, 255, 255])},
                'red': {'lower1': np.array([0, 120, 70]), 'upper1': np.array([10, 255, 255]),
                        'lower2': np.array([170, 120, 70]), 'upper2': np.array([180, 255, 255])}
            }
            
            results = {}
            
            # Analyze each color
            for color_name, bounds in color_bounds.items():
                if color_name == 'red':
                    # Red has two ranges
                    mask1 = cv2.inRange(hsv_frame, bounds['lower1'], bounds['upper1'])
                    mask2 = cv2.inRange(hsv_frame, bounds['lower2'], bounds['upper2'])
                    mask = cv2.bitwise_or(mask1, mask2)
                else:
                    # Single range for other colors
                    mask = cv2.inRange(hsv_frame, bounds['lower'], bounds['upper'])
                
                # Count pixels in color range
                pixel_count = cv2.countNonZero(mask)
                total_pixels = frame.shape[0] * frame.shape[1]
                percentage = (pixel_count / total_pixels) * 100
                
                results[color_name] = {
                    'pixel_count': pixel_count,
                    'percentage': percentage,
                    'detected': pixel_count > 100  # Threshold for detection
                }
                
                print(f"{color_name.title()} pixels: {pixel_count} ({percentage:.2f}%)")
            
            # Save annotated frame
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"camera_captures/color_test_{timestamp}.jpg"
            
            # Add color information to frame
            y_offset = 30
            for color_name, data in results.items():
                text = f"{color_name.title()}: {data['pixel_count']} pixels ({data['percentage']:.1f}%)"
                color = (0, 255, 0) if data['detected'] else (0, 0, 255)
                cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                y_offset += 25
            
            os.makedirs("camera_captures", exist_ok=True)
            cv2.imwrite(filename, frame)
            print(f"Color analysis frame saved as: {filename}")
            
            cap.release()
            result = results
            
        except Exception as e:
            print(f"Error in camera color detection test: {e}")
            if 'cap' in locals() and cap is not None:
                cap.release()
            result = {}
        
        finally:
            # Restart camera controller if it was running
            if camera_was_running:
                print("Restarting camera controller...")
                time.sleep(0.5)  # Brief pause before restart
                self.start_camera_controller()
        
        return result
    
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
        Get current camera object detections ordered by area (largest first)
        
        Returns:
            list: List of detected objects with color, position, and angle information ordered by area descending
        """
        try:
            detection_count = self.camera_detection_count.value
            detections = decode_detection_data(self.camera_detections, detection_count)
            # Sort by area in descending order (largest objects first)
            return sorted(detections, key=lambda x: x.get('area', 0), reverse=True)
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
            color (str): Color to filter by ('green', 'red')
        
        Returns:
            list: List of objects matching the specified color
        """
        try:
            all_detections = self.get_camera_detections()
            return [det for det in all_detections if det['color'].lower() == color.lower()]
        except Exception as e:
            print(f"Error getting objects by color {color}: {e}")
            return []
    
    def get_closest_object_by_angle(self, target_angle, tolerance=10):
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
                        print("Detection within angle:", detection)
                        # Get LiDAR distance at this object's angle
                        lidar_distance = self.get_distance(detection['bearing'])
                        if lidar_distance is not None and lidar_distance != 9999:
                            objects_in_range.append((lidar_distance, detection['bearing']))
            
            if objects_in_range:
                # Return the object with the smallest distance
                print(objects_in_range)
                objects_in_range.sort(key=lambda x: x[0])
                print(objects_in_range)
                print("Closest object: Distance: " + str(objects_in_range[0][0]) + ", Bearing: " + str(objects_in_range[0][1]))
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
            fov_half_angle_rad = math.radians(5)  # ±15 degrees FOV
            
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
        print("  relative        - Show angle relative to starting position")
        print("  reset_offset    - Reset forward offset to current angle")
        print("  offset          - Show forward offset angle")
        print()
        print("LiDAR Commands:")
        print("  lidar           - Show LiDAR status")
        print("  points          - Show LiDAR point data")
        print("  obstacle        - Show closest obstacle distance")
        print("  distance <angle> - Show median distance at angle (±2.5°)")
        print("  furthest <angle> - Show 95th percentile furthest point (±15°)")
        print("  closest <angle>  - Show closest point (±15°)")
        print("  smart <angle>    - Show intelligent distance reading (±5°)")
        print()
        print("Camera Commands:")
        print("  camera          - Show camera status")
        print("  detections      - Show all object detections")
        print("  color <color>   - Show objects of specific color (green/red)")
        print("  angle_obj <angle> - Show closest object to angle (±10°)")
        print("  test_pixel <x> <y> - Test camera by getting RGB at pixel coordinates")
        print("  test_center     - Test camera RGB at center pixel (320, 240)")
        print("  test_pixels     - Test camera RGB at multiple test points")
        print("  test_colors     - Test camera color detection capabilities")
        print()
        print("Servo Commands (Pin 26):")
        print("  servo <angle>   - Set servo angle (0-180 degrees)")
        print("  servo_center    - Center servo to 90 degrees")
        print("  servo_status    - Show servo current angle")
        print("  servo_init      - Initialize servo on pin 26")
        print()
        print("Button Commands (Pin 23):")
        print("  button_state    - Show current button state")
        print("  button_status   - Show detailed button status")
        print("  button_wait     - Wait for button press")
        print("  button_init     - Initialize button on pin 23")
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
                        print("  relative      - Show angle relative to starting position")
                        print("  reset_offset  - Reset forward offset to current angle")
                        print("  offset        - Show forward offset angle")
                        print("\nLiDAR Commands:")
                        print("  lidar         - Show LiDAR status")
                        print("  points        - Show LiDAR point data")
                        print("  obstacle      - Show closest obstacle distance")
                        print("  distance <angle> - Show median distance at angle (±2.5°)")
                        print("  furthest <angle> - Show 95th percentile furthest point (±15°)")
                        print("  closest <angle>  - Show closest point (±15°)")
                        print("  smart <angle>    - Show intelligent distance reading (±5°)")
                        print("\nCamera Commands:")
                        print("  camera        - Show camera status")
                        print("  detections    - Show all object detections")
                        print("  color <color> - Show objects of specific color")
                        print("  angle_obj <angle> - Show closest object to angle")
                        print("  test_pixel <x> <y> - Test camera RGB at pixel coordinates")
                        print("  test_center   - Test camera RGB at center pixel")
                        print("  test_pixels   - Test camera RGB at multiple test points")
                        print("  test_colors   - Test camera color detection capabilities")
                        print("\nServo Commands (Pin 26):")
                        print("  servo <angle> - Set servo angle (0-180 degrees)")
                        print("  servo_center  - Center servo to 90 degrees")
                        print("  servo_status  - Show servo current angle")
                        print("  servo_init    - Initialize servo on pin 26")
                        print("\nButton Commands (Pin 23):")
                        print("  button_state  - Show current button state")
                        print("  button_status - Show detailed button status")
                        print("  button_wait   - Wait for button press")
                        print("  button_init   - Initialize button on pin 23")
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
                        print(f"Relative Angle: {self.get_relative_angle():.1f}° (relative to start)")
                        print(f"Forward Offset: {self.get_forward_offset_angle():.1f}°")
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
                    elif command.startswith('smart '):
                        try:
                            angle = float(command.split()[1])
                            distance, method = self.get_intelligent_distance_reading(angle)
                            if distance is not None:
                                method_text = "furthest point" if method == 'furthest' else "closest point"
                                print(f"Smart reading at {angle}° (±5°): {distance:.1f} mm ({distance/1000:.2f} m)")
                                print(f"Method used: {method_text} ({'objects detected nearby' if method == 'furthest' else 'no objects nearby'})")
                            else:
                                print(f"No LiDAR points found at {angle}° (±5°)")
                        except (IndexError, ValueError):
                            print("Usage: smart <angle_in_degrees>")
                            print("Example: smart 0    (forward)")
                            print("Example: smart 45   (45° clockwise)")
                            print("Example: smart -30  (30° counter-clockwise)")
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
                            print("Example: color red")
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
                    elif command == 'test_center':
                        print("Testing camera at center pixel (320, 240)...")
                        r, g, b, success = self.test_camera_center_pixel()
                        if success:
                            print(f"Center pixel RGB: ({r}, {g}, {b})")
                            brightness = (r + g + b) / 3
                            print(f"Brightness level: {brightness:.1f}")
                        else:
                            print("Failed to capture center pixel")
                    elif command.startswith('test_pixel '):
                        try:
                            parts = command.split()
                            if len(parts) >= 3:
                                x = int(parts[1])
                                y = int(parts[2])
                                print(f"Testing camera at pixel ({x}, {y})...")
                                r, g, b, success = self.test_camera_pixel_rgb(x, y)
                                if success:
                                    print(f"Pixel RGB: ({r}, {g}, {b})")
                                    brightness = (r + g + b) / 3
                                    print(f"Brightness level: {brightness:.1f}")
                                else:
                                    print("Failed to capture pixel")
                            else:
                                print("Usage: test_pixel <x> <y>")
                                print("Example: test_pixel 320 240  (center)")
                                print("Example: test_pixel 100 100  (top-left area)")
                        except (ValueError):
                            print("Invalid pixel coordinates. Use integers.")
                            print("Usage: test_pixel <x> <y>")
                    elif command == 'test_pixels':
                        print("Testing camera at multiple pixel locations...")
                        results = self.test_camera_multiple_pixels()
                        if results:
                            print("Test completed successfully")
                        else:
                            print("Failed to complete pixel tests")
                    elif command == 'test_colors':
                        print("Testing camera color detection capabilities...")
                        results = self.test_camera_color_detection()
                        if results:
                            detected_colors = [color for color, data in results.items() if data['detected']]
                            if detected_colors:
                                print(f"Colors detected: {', '.join(detected_colors)}")
                            else:
                                print("No target colors detected in current view")
                        else:
                            print("Failed to complete color detection test")
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
                    elif command == 'relative':
                        relative_angle = self.get_relative_angle()
                        print(f"Relative angle: {relative_angle:.1f}° (relative to starting position)")
                        print(f"Forward offset angle: {self.get_forward_offset_angle():.1f}°")
                    elif command == 'reset_offset':
                        self.reset_forward_offset_angle()
                    elif command == 'offset':
                        offset_angle = self.get_forward_offset_angle()
                        print(f"Forward offset angle: {offset_angle:.1f}°")
                    elif command == 'servo_init':
                        if self.initialize_servo_pin_26():
                            print("Servo on pin 26 initialized successfully")
                        else:
                            print("Failed to initialize servo on pin 26")
                    elif command == 'servo_center':
                        self.center_servo_pin_26()
                    elif command == 'servo_status':
                        current_angle = self.get_servo_angle_pin_26()
                        print(f"Servo on pin 26 current angle: {current_angle}°")
                    elif command.startswith('servo '):
                        try:
                            angle = float(command.split()[1])
                            if 0 <= angle <= 180:
                                self.set_servo_angle_pin_26(angle)
                            else:
                                print("Angle must be between 0 and 180 degrees")
                        except (IndexError, ValueError):
                            print("Usage: servo <angle> (0-180 degrees)")
                    elif command == 'button_init':
                        if self.initialize_button_pin_23():
                            print("Button on pin 23 initialized successfully")
                        else:
                            print("Failed to initialize button on pin 23")
                    elif command == 'button_state':
                        state = self.get_button_state_pin_23()
                        debounced_state = self.get_button_state_debounced_pin_23()
                        if state == -1:
                            print("Error reading button state - button may not be initialized")
                        else:
                            state_text = "HIGH (Pressed)" if state == 1 else "LOW (Released)"
                            debounced_text = "HIGH (Pressed)" if debounced_state == 1 else "LOW (Released)"
                            print(f"Button state: {state_text}")
                            print(f"Debounced state: {debounced_text}")
                    elif command == 'button_status':
                        status = self.get_button_status_pin_23()
                        if 'error' in status:
                            print(f"Button status error: {status['error']}")
                        else:
                            print(f"Button Status: {status['status']}")
                            print(f"Raw state: {status['raw_state']} ({'Pressed' if status['raw_state'] == 1 else 'Released'})")
                            print(f"Debounced state: {status['debounced_state']} ({'Pressed' if status['debounced_state'] == 1 else 'Released'})")
                            print(f"Initialized: {status['initialized']}")
                            print(f"Last state change: {time.ctime(status['last_time'])}")
                    elif command == 'button_wait':
                        print("Press the button on pin 23...")
                        if self.wait_for_button_press_pin_23(timeout=10):
                            print("Button press detected!")
                        else:
                            print("No button press detected within timeout")
                    elif command == 'test':
                        self.run_test_sequence()
                    elif command == 'viz' or command == 'visualize':
                        print("Starting object visualization...")
                        self.start_object_visualization()
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
    
    def get_intelligent_distance_reading(self, requested_angle, detection_tolerance=15.0):
        """
        Intelligently choose between furthest point and closest point detection
        based on whether there are detected objects nearby
        
        Args:
            requested_angle (float): Target angle in degrees (0° = forward, positive = clockwise)
            detection_tolerance (float): Tolerance in degrees to check for nearby objects
        
        Returns:
            tuple: (distance, method_used) where method_used is 'furthest' or 'closest'
        """
        try:
            # Get all camera detections
            camera_detections = self.get_camera_detections()
            
            # Check if there are any detected objects within tolerance of the requested angle
            objects_nearby = False
            
            for detection in camera_detections:
                if detection['bearing'] is not None:
                    # Calculate angle difference
                    angle_diff = abs(detection['bearing'] - requested_angle)
                    
                    # Handle angle wrapping (e.g., difference between 359° and 1°)
                    if angle_diff > 180:
                        angle_diff = 360 - angle_diff
                    
                    # Check if object is within tolerance
                    if angle_diff <= detection_tolerance:
                        objects_nearby = True
                        break
            
            # Choose method based on whether objects are detected nearby
            if objects_nearby:
                # Objects detected nearby - use furthest point to see beyond them
                distance = self.get_furthest_point(requested_angle)
                method_used = 'furthest'
            else:
                # No objects nearby - use closest point for accurate obstacle detection
                distance = self.get_distance(requested_angle)
                method_used = 'closest'
            
            return distance, method_used
            
        except Exception as e:
            print(f"Error in intelligent distance reading at angle {requested_angle}°: {e}")
            # Fallback to closest point method
            return self.get_closest_point(requested_angle), 'closest'
    
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
            'red': (255, 0, 0)
        }
        
        try:
            pygame.init()
            screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            pygame.display.set_caption("TaxiDriver Object Visualization")
            clock = pygame.time.Clock()
            font = pygame.font.SysFont(None, 24)
            
            center_x, center_y = SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2
            
            print("Visualization started. Press 'q' or ESC to exit.")
            
            # Pre-create some font surfaces to avoid recreation
            forward_label = font.render("FORWARD", True, (0, 255, 0))
            instruction_label = font.render("Press 'q' or ESC to exit", True, (200, 200, 200))
            
            running = True
            frame_count = 0
            
            while running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key in [pygame.K_ESCAPE, pygame.K_q]):
                        running = False
                
                # Clear screen
                screen.fill(BACKGROUND_COLOR)
                
                # Get current gyroscope angle for offset calculations
                current_gyro_angle = (self.get_current_angle() - self.get_forward_offset_angle()) % 360
                print(current_gyro_angle)
                if current_gyro_angle > 180:
                    current_gyro_angle -= 360
                
                # Draw grid circles for distance reference (less frequently)
                if frame_count % 10 == 0:  # Update grid every 10 frames
                    for r in range(1000, MAX_DISTANCE_MM + 1, 1000):
                        radius_px = int(r / SCALE_FACTOR)
                        pygame.draw.circle(screen, GRID_COLOR, (center_x, center_y), radius_px, 1)
                        # Distance labels
                        label = font.render(f"{r//1000}m", True, GRID_COLOR)
                        screen.blit(label, (center_x + radius_px - 15, center_y - 10))
                else:
                    # Just draw circles without recreating labels
                    for r in range(1000, MAX_DISTANCE_MM + 1, 1000):
                        radius_px = int(r / SCALE_FACTOR)
                        pygame.draw.circle(screen, GRID_COLOR, (center_x, center_y), radius_px, 1)
                
                # Draw angle reference lines (every 45 degrees instead of 30 for performance)
                for angle_deg in range(0, 360, 45):
                    angle_rad = math.radians(angle_deg)
                    end_x = center_x + int((SCREEN_WIDTH // 2 - 50) * math.sin(angle_rad))
                    end_y = center_y - int((SCREEN_WIDTH // 2 - 50) * math.cos(angle_rad))
                    pygame.draw.line(screen, GRID_COLOR, (center_x, center_y), (end_x, end_y), 1)
                
                # Draw forward direction indicator
                pygame.draw.line(screen, (0, 255, 0), (center_x, center_y), (center_x, center_y - 100), 3)
                screen.blit(forward_label, (center_x - 30, center_y - 130))
                
                # Draw LiDAR points (optimized - sample every 3rd point for performance)
                lidar_points = self.get_lidar_points()
                for i, (x, y) in enumerate(lidar_points[::3]):  # Sample every 3rd point
                    distance = math.sqrt(x**2 + y**2)
                    if distance <= MAX_DISTANCE_MM:
                        screen_x = center_x + int(x / SCALE_FACTOR)
                        screen_y = center_y - int(y / SCALE_FACTOR)
                        pygame.draw.circle(screen, LIDAR_POINT_COLOR, (screen_x, screen_y), 1)
                
                # Get intelligent distance readings at key angles (offset by gyroscope)
                key_angles = [0, 90, -90]  # Forward, right, left
                distance_indicators = []
                
                for base_angle in key_angles:
                    # Offset by current gyroscope angle
                    actual_angle = base_angle + current_gyro_angle
                    
                    # Use intelligent distance detection
                    
                    
                    distance_reading, method_used = self.get_intelligent_distance_reading(actual_angle)
                    
                    if base_angle == 0:
                        distance_reading, method_used = self.get_furthest_point(actual_angle)

                    if distance_reading is not None and distance_reading <= MAX_DISTANCE_MM:
                        # Convert to screen coordinates
                        angle_rad = math.radians(actual_angle)
                        x_mm = distance_reading * math.sin(angle_rad)
                        y_mm = distance_reading * math.cos(angle_rad)
                        
                        screen_x = center_x + int(x_mm / SCALE_FACTOR)
                        screen_y = center_y - int(y_mm / SCALE_FACTOR)
                        
                        # Choose color based on direction
                        if base_angle == 0:
                            indicator_color = (0, 255, 0)    # Green for forward
                            direction_text = "F"
                        elif base_angle == 90:
                            indicator_color = (255, 255, 0)  # Yellow for right
                            direction_text = "R"
                        else:  # -90
                            indicator_color = (0, 255, 255)  # Cyan for left
                            direction_text = "L"
                        
                        # Draw distance indicator with different styles based on method
                        if method_used == 'furthest':
                            # Solid circle for furthest detection (looking beyond objects)
                            pygame.draw.circle(screen, indicator_color, (screen_x, screen_y), 6)
                            pygame.draw.circle(screen, (255, 255, 255), (screen_x, screen_y), 6, 2)
                        else:
                            # Hollow circle for closest detection (obstacle detection)
                            pygame.draw.circle(screen, (255, 255, 255), (screen_x, screen_y), 6, 2)
                            pygame.draw.circle(screen, indicator_color, (screen_x, screen_y), 4)
                        
                        # Draw line from robot to detection point
                        pygame.draw.line(screen, indicator_color, (center_x, center_y), (screen_x, screen_y), 2)
                        
                        # Add direction label with method indicator
                        method_symbol = "F" if method_used == 'furthest' else "C"
                        dir_label = font.render(f"{direction_text}{method_symbol}", True, indicator_color)
                        screen.blit(dir_label, (screen_x - 10, screen_y - 20))
                        
                        distance_indicators.append({
                            'direction': direction_text,
                            'angle': actual_angle,
                            'distance': distance_reading,
                            'method': method_used,
                            'color': indicator_color
                        })
                
                # Get camera detections and correlate with LiDAR distance (optimized)
                camera_detections = self.get_camera_detections()
                detected_objects = []
                
                # Batch process detections to avoid multiple LiDAR queries
                detection_angles = [det['bearing'] for det in camera_detections if det['bearing'] is not None]
                
                for detection in camera_detections:
                    if detection['bearing'] is not None:
                        angle = detection['bearing']
                        
                        # Get LiDAR distance at this angle (this is the bottleneck)
                        lidar_distance = self.get_closest_point(angle)
                        
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
                            
                            detected_objects.append({
                                'color': detection['color'],
                                'angle': angle,
                                'distance': lidar_distance,
                                'screen_pos': (screen_x, screen_y)
                            })
                
                # Draw robot (center circle)
                pygame.draw.circle(screen, ROBOT_COLOR, (center_x, center_y), 10)
                pygame.draw.circle(screen, (0, 0, 0), (center_x, center_y), 10, 2)
                
                # Status information (update text less frequently)
                status_y = 10
                if frame_count % 5 == 0:  # Update status every 5 frames
                    status_texts = [
                        f"Objects detected: {len(detected_objects)}",
                        f"LiDAR points: {self.get_lidar_points_count()}",
                        f"Camera detections: {self.get_camera_detection_count()}",
                        f"Gyro angle: {current_gyro_angle:.1f}°",
                        f"Frame: {frame_count}",
                    ]
                
                # Always draw the cached status texts
                if 'status_texts' in locals():
                    for text in status_texts:
                        label = font.render(text, True, (255, 255, 255))
                        screen.blit(label, (10, status_y))
                        status_y += 25
                
                # Distance indicators info (intelligent detection)
                if distance_indicators:
                    status_y += 10
                    distance_label = font.render("Intelligent Distance Readings:", True, (255, 255, 255))
                    screen.blit(distance_label, (10, status_y))
                    status_y += 25
                    
                    for indicator in distance_indicators:
                        method_text = "Far" if indicator['method'] == 'furthest' else "Close"
                        indicator_text = f"  {indicator['direction']}: {indicator['distance']:.0f}mm ({method_text}) @ {indicator['angle']:.1f}°"
                        indicator_label = font.render(indicator_text, True, indicator['color'])
                        screen.blit(indicator_label, (10, status_y))
                        status_y += 20
                
                # Object summary (limit to prevent text overflow)
                if detected_objects and frame_count % 3 == 0:  # Update every 3 frames
                    summary_y = status_y + 10
                    summary_label = font.render("Detected Objects:", True, (255, 255, 255))
                    screen.blit(summary_label, (10, summary_y))
                    summary_y += 25
                    
                    for obj in detected_objects[:3]:  # Show only first 3 objects
                        obj_text = f"  {obj['color']}: {obj['distance']:.0f}mm @ {obj['angle']:.1f}°"
                        obj_label = font.render(obj_text, True, OBJECT_COLORS.get(obj['color'], (255, 255, 255)))
                        screen.blit(obj_label, (10, summary_y))
                        summary_y += 20
                
                # Instructions
                screen.blit(instruction_label, (10, SCREEN_HEIGHT - 30))
                
                pygame.display.flip()
                clock.tick(30)  # 30 FPS
                frame_count += 1
                
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
        self.cleanup_servo_pin_26()
        self.cleanup_button_pin_23()
        self.stop_all_controllers()

def main():
    """Main function"""
    taxi = None
    try:
        print("=== Taxi Driver 2.0 ===")
        taxi = TaxiDriver()
        
        print("Starting all controllers...")
        taxi.start_all_controllers()
        
        print("Waiting for controllers to initialize...")
        time.sleep(5)  # Wait for controllers to initialize
        
        print("Starting object visualization...")
        print("- Green dots: LiDAR points")
        print("- Colored circles: Detected objects (green/red)")
        print("- White circle: Robot position")
        print("- Green line: Forward direction")
        print("- Press 'q' or ESC to exit visualization")
        print()
        
        # Start visualization automatically
        taxi.start_object_visualization()
        
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if taxi:
            taxi.cleanup()

if __name__ == "__main__":
    main()