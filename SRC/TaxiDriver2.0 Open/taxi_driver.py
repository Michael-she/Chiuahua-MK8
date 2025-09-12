#!/usr/bin/env python3
"""
Taxi Driver Main Controller
Controls motor using multiprocessing for communication with motor controller slave
"""

import multiprocessing
import time
import threading
import math
from motor_controller_slave import motor_controller_process
from gyroscope_slave import gyroscope_controller_process
from LiDAR_slave import lidar_data_process

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
        
        # Controller processes
        self.motor_process = None
        self.gyro_process = None
        self.lidar_process = None
        
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
    
    def start_all_controllers(self):
        """Start motor, gyroscope, and LiDAR controllers"""
        self.start_motor_controller()
        self.start_gyroscope_controller()
        self.start_lidar_controller()
    
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
    
    def stop_all_controllers(self):
        """Stop motor, gyroscope, and LiDAR controllers"""
        self.stop_motor_controller()
        self.stop_gyroscope_controller()
        self.stop_lidar_controller()
    
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
    
    def get_full_status(self):
        """
        Get complete system status
        
        Returns:
            dict: Complete system status
        """
        motor_status = self.get_motor_status()
        gyro_status = self.get_gyroscope_status()
        lidar_status = self.get_lidar_status()
        
        return {
            'motor': motor_status,
            'gyroscope': gyro_status,
            'lidar': lidar_status,
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
        print("Status Commands:")
        print("  status          - Show motor and encoder status")
        print("  gyro            - Show gyroscope and servo status")
        print("  full            - Show complete system status")
        print("  encoder         - Show encoder reading")
        print("  test            - Run test sequence")
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
                        print("\nStatus Commands:")
                        print("  status        - Show motor status")
                        print("  gyro          - Show gyroscope status")
                        print("  full          - Show complete system status")
                        print("  encoder       - Show encoder reading")
                        print("  rotations     - Show total rotations")
                        print("  absolute      - Show absolute angle")
                        print("\nOther:")
                        print("  test          - Run test sequence")
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
                    elif command == 'full':
                        status = self.get_full_status()
                        motor = status.get('motor', {})
                        gyro = status.get('gyroscope', {})
                        lidar = status.get('lidar', {})
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
        
        print("Starting all controllers...")
        taxi.start_all_controllers()
        
        # Give controllers time to initialize
        time.sleep(3)
        
        # Start interactive control
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