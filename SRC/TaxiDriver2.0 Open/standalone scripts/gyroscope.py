# SPDX-FileCopyrightText: 2020 Bryan Siepert, written for Adafruit Industries
#
# SPDX-License-Identifier: Unlicense
import time
import board
import busio
import sys
import math
import threading
import pigpio
from adafruit_bno08x import (
	BNO_REPORT_ACCELEROMETER,
	BNO_REPORT_GYROSCOPE,
	BNO_REPORT_MAGNETOMETER,
	BNO_REPORT_ROTATION_VECTOR,
)
from adafruit_bno08x.i2c import BNO08X_I2C

# Lookup table for angle correction (from Arduino script)
TRUE_ANGLE_LOOKUP = [
    0, 1, 2, 2, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 9, 9, 10, 10, 11, 11, 12, 13, 13, 14, 15, 15, 16, 16, 17, 18, 18, 19, 20, 20, 21, 21, 22, 23, 23, 24, 25, 25, 26, 26, 27, 28, 28, 29, 30, 30, 31, 31, 32, 33, 33, 34, 35, 35, 36, 36, 37, 38, 38, 39, 40, 40, 41, 41, 42, 43, 43, 44, 45, 45, 46, 46, 47, 48, 48, 49, 50, 50, 51, 52, 53, 53, 54, 55, 56, 57, 58, 58, 59, 60, 61, 62, 63, 63, 64, 65, 66, 67, 68, 68, 69, 70, 71, 72, 73, 73, 74, 75, 76, 77, 78, 78, 79, 80, 81, 82, 83, 83, 84, 85, 86, 87, 88, 88, 89, 90, 91, 92, 94, 95, 96, 98, 99, 100, 101, 102, 104, 105, 106, 107, 109, 110, 111, 112, 114, 115, 116, 117, 119, 120, 121, 122, 124, 125, 126, 127, 129, 130, 131, 133, 134, 135, 137, 138, 140, 142, 145, 147, 150, 153, 157, 161, 165, 170, 175, 180, 180
]

# Servo configuration
SERVO_PIN = 12      # GPIO pin for servo (BCM numbering)
SERVO_MIN_PULSE = 1000   # Minimum pulse width in microseconds (1ms)
SERVO_MAX_PULSE = 2000   # Maximum pulse width in microseconds (2ms)
SERVO_CENTER = 94   # Center position (from Arduino script)
SERVO_MIN = 64      # Minimum servo position (center - 30)
SERVO_MAX = 124     # Maximum servo position (center + 30)

class ServoController:
    def __init__(self, pin=SERVO_PIN):
        self.pin = pin
        
        # Initialize pigpio
        self.pi = pigpio.pi()
        if not self.pi.connected:
            raise RuntimeError("Failed to connect to pigpio daemon. Make sure pigpio daemon is running.")
        
        # Thread-safe variables
        self.target_angle = 0
        self.current_angle = 0
        self.servo_position = SERVO_CENTER
        self._lock = threading.Lock()
        self._running = True
        
        # Start servo control thread
        self._servo_thread = threading.Thread(target=self._servo_control_loop, daemon=True)
        self._servo_thread.start()
        
    def _servo_control_loop(self):
        """Independent servo control loop running at consistent timing"""
        while self._running:
            with self._lock:
                current_angle = self.current_angle
                target_angle = self.target_angle
            
            # Calculate steering adjustment
            angle_diff = current_angle - target_angle
            
            # Apply maximum steering limits
            max_steering = 30
            if angle_diff > max_steering:
                angle_diff = max_steering
            elif angle_diff < -max_steering:
                angle_diff = -max_steering
                
            # Calculate new servo position
            new_servo_position = SERVO_CENTER + angle_diff
            new_servo_position = max(SERVO_MIN, min(SERVO_MAX, new_servo_position))
            
            # Update servo position
            with self._lock:
                self.servo_position = new_servo_position
            
            # Convert to servo pulse width and set using pigpio
            pulse_width = self.position_to_pulse_width(new_servo_position)
            self.pi.set_servo_pulsewidth(self.pin, pulse_width)
            
            # Consistent 50Hz servo update rate (20ms period)
            time.sleep(0.02)
        
    def position_to_pulse_width(self, position):
        """Convert servo position (64-124) to pulse width in microseconds"""
        # Map position range to pulse width range
        # Position 64-124 maps to 1000-2000 microseconds
        normalized_position = (position - SERVO_MIN) / (SERVO_MAX - SERVO_MIN)
        pulse_width = SERVO_MIN_PULSE + normalized_position * (SERVO_MAX_PULSE - SERVO_MIN_PULSE)
        return int(pulse_width)
        
    def update_current_angle(self, angle):
        """Thread-safe method to update current angle from gyroscope"""
        with self._lock:
            self.current_angle = angle
        
    def set_target_angle(self, target):
        """Thread-safe method to set target angle"""
        with self._lock:
            self.target_angle = target
    
    def get_servo_position(self):
        """Thread-safe method to get current servo position"""
        with self._lock:
            return self.servo_position
    
    def get_target_angle(self):
        """Thread-safe method to get target angle"""
        with self._lock:
            return self.target_angle
        
    def cleanup(self):
        """Clean up pigpio resources and stop servo thread"""
        self._running = False
        if self._servo_thread.is_alive():
            self._servo_thread.join(timeout=1.0)
        
        # Center servo before shutdown
        center_pulse = self.position_to_pulse_width(SERVO_CENTER)
        self.pi.set_servo_pulsewidth(self.pin, center_pulse)
        time.sleep(0.5)
        
        # Turn off servo signal
        self.pi.set_servo_pulsewidth(self.pin, 0)
        
        # Disconnect from pigpio daemon
        self.pi.stop()

class GyroscopeProcessor:
    def __init__(self):
        self.true_angle = 0
        self.true_angle_offset = 0
        self.display_angle = 0
        self.angle_old = 9999
        self.stuck_count = 0
        self.last_update_time = 0
        
    def quaternion_to_angle(self, quat_i, quat_j, quat_k, quat_real):
        """
        Convert quaternion to angle using the Arduino script's algorithm
        with improved stability and negative angle fix
        """
        # Ensure quaternion is normalized (safety check)
        quat_norm = (quat_i**2 + quat_j**2 + quat_k**2 + quat_real**2) ** 0.5
        if quat_norm > 0:
            quat_i /= quat_norm
            quat_j /= quat_norm  
            quat_k /= quat_norm
            quat_real /= quat_norm
        
        # Calculate lookup angle from K component
        lookup_angle = round(quat_k * 180)
        
        # Handle four different quadrants based on K value ranges
        if lookup_angle >= 0 and quat_k <= 0.71:
            # Normal positive range
            self.true_angle = TRUE_ANGLE_LOOKUP[abs(lookup_angle % 180)]
            
        elif lookup_angle < 0 and quat_k >= -0.71:
            # Normal negative range - fix modulo operation for negative values
            self.true_angle = -TRUE_ANGLE_LOOKUP[abs(lookup_angle) % 180]
            
        elif quat_k > 0.71:
            # Reverse positive range (using real component)
            lookup_angle = round((1 - quat_real) * 180)
            measured_angle = TRUE_ANGLE_LOOKUP[abs(180 - lookup_angle % 180)]
            
            if measured_angle != 180:
                self.true_angle = 180 - TRUE_ANGLE_LOOKUP[abs(180 - lookup_angle % 180)]
            else:
                self.true_angle = 180
                
        elif quat_k < -0.71:
            # Reverse negative range (using real component)
            lookup_angle = round((1 - quat_real) * 180)
            measured_angle = abs(180 - lookup_angle % 180)
            
            if measured_angle != 180:
                self.true_angle = -(180 - TRUE_ANGLE_LOOKUP[abs(180 - lookup_angle % 180)])
            else:
                self.true_angle = -180
        
        # Additional stability check: detect and fix sign inconsistencies
        # This helps with the occasional negative angle glitch
        
        # Enhanced wrap-around logic with better jump detection
        if self.angle_old != 9999:  # Skip first reading
            angle_diff = self.true_angle - self.angle_old
            
            # Detect large jumps that indicate wrap-around (more conservative thresholds)
            if angle_diff > 300:  # Jump from negative to positive (e.g., -170° to +170°)
                self.true_angle_offset -= 360
                print(f"Wrap-around detected: {self.angle_old:.1f}° → {self.true_angle:.1f}° (negative offset)")
                
            elif angle_diff < -300:  # Jump from positive to negative (e.g., +170° to -170°)
                self.true_angle_offset += 360
                print(f"Wrap-around detected: {self.angle_old:.1f}° → {self.true_angle:.1f}° (positive offset)")
                
            # Additional check for boundary crossings (more conservative)
            elif self.angle_old >= 160 and self.true_angle <= -160:
                # Crossed from +180 to -180 region
                self.true_angle_offset += 360
                print(f"Boundary crossing: +{self.angle_old:.1f}° → {self.true_angle:.1f}° (positive offset)")
                
            elif self.angle_old <= -160 and self.true_angle >= 160:
                # Crossed from -180 to +180 region  
                self.true_angle_offset -= 360
                print(f"Boundary crossing: {self.angle_old:.1f}° → {self.true_angle:.1f}° (negative offset)")
        
        # Enhanced stuck value detection with brief anomaly filtering
        if self.angle_old == self.true_angle:
            self.stuck_count += 1
            if self.stuck_count > 5:  # Increased threshold for better reliability
                # Return previous display angle without updating
                return self.display_angle
        else:
            # Additional filter for brief anomalous readings
            if self.angle_old != 9999:  # Not first reading
            
                # Check if this looks like a brief sensor artifact
                # by seeing if it's very different from recent stable readings
                expected_range_low = self.angle_old - 20
                expected_range_high = self.angle_old + 20
                
                if not (expected_range_low <= self.true_angle <= expected_range_high):
                    # This looks like a brief anomaly - use the previous angle instead
                    print(f"Filtered anomaly: {self.true_angle:.1f}° -> {self.angle_old:.1f}°")
                    self.true_angle = self.angle_old
                    return self.display_angle
            
            self.stuck_count = 0  # Reset stuck counter when value changes
        
        self.angle_old = self.true_angle
        self.display_angle = self.true_angle + self.true_angle_offset
        
        return self.display_angle

def init_bno08x():
    """(Re)initialize the I2C bus and BNO08X sensor, with retries."""
    for attempt in range(5):
        try:
            i2c = busio.I2C(board.SCL, board.SDA, frequency=400000)
            bno = BNO08X_I2C(i2c)
            bno.enable_feature(BNO_REPORT_ROTATION_VECTOR)
            print("BNO08X sensor initialized.")
            return i2c, bno
        except Exception as e:
            print(f"BNO08X init failed (attempt {attempt+1}): {e}")
            time.sleep(0.5)
    raise RuntimeError("Failed to initialize BNO08X after multiple attempts.")

def main():
    """Main function with proper error handling and graceful degradation."""
    servo_controller = None
    try:
        # Initialize the gyroscope processor and servo controller
        gyro_processor = GyroscopeProcessor()
        servo_controller = ServoController()

        print("Initializing BNO08X sensor...")
        i2c, bno = init_bno08x()

        print("Sensor and servo initialized successfully. Starting data collection at 20Hz...")
        print("Press Ctrl+C to stop\n")
        print("Waiting 1 second for sensor stabilization...")
        time.sleep(1.0)

        # Get initial angle reading
        try:
            quat_i, quat_j, quat_k, quat_real = bno.quaternion # pylint:disable=no-member
            initial_angle = gyro_processor.quaternion_to_angle(quat_i, quat_j, quat_k, quat_real)
            servo_controller.set_target_angle(initial_angle)
            print(f"Target angle set to current angle: {initial_angle:.1f}°")
        except Exception as e:
            print(f"Could not read initial angle: {e}. Setting target to 0°")
            servo_controller.set_target_angle(0)

        loop_time = 1.0 / 20.0  # 50ms for 20Hz

        while True:
            start_time = time.time()
            try:
                quat_i, quat_j, quat_k, quat_real = bno.quaternion
                angle = gyro_processor.quaternion_to_angle(quat_i, quat_j, quat_k, quat_real)
                servo_controller.update_current_angle(angle)
                servo_position = servo_controller.get_servo_position()
                target_angle = servo_controller.get_target_angle()
                print(f"Angle: {angle:6.1f}°, Target: {target_angle:3.0f}°, "
                      f"Servo: {servo_position:3.0f}, Diff: {angle - target_angle:6.1f}°")
                elapsed_time = time.time() - start_time
                remaining_time = loop_time - elapsed_time
                if remaining_time > 0:
                    time.sleep(remaining_time)
                else:
                    print(f"Warning: Loop took {elapsed_time*1000:.1f}ms (target: {loop_time*1000:.1f}ms)")
            except KeyError as e:
                print(f"Warning: Received unknown packet type {e}. Continuing...")
                time.sleep(0.001)
                continue
            except OSError as e:
                print(f"I2C communication error: {e}. Attempting to reinitialize sensor...")
                try:
                    # Try to reinitialize the sensor
                    i2c, bno = init_bno08x()
                    print("Sensor reinitialized after I2C error.")
                except Exception as reinit_e:
                    print(f"Reinitialization failed: {reinit_e}. Will retry in 2 seconds...")
                    time.sleep(2)
                continue
            except Exception as e:
                print(f"Unexpected error: {e}. Continuing...")
                time.sleep(0.01)
                continue

    except KeyboardInterrupt:
        print("\nStopping sensor reading...")
        if servo_controller:
            print("Cleaning up servo...")
            servo_controller.cleanup()
    except Exception as e:
        print(f"Fatal error during initialization: {e}")
        if servo_controller:
            servo_controller.cleanup()
        sys.exit(1)

if __name__ == "__main__":
    main()
