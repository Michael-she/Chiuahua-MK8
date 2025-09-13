#!/usr/bin/env python3
"""
Simple Motor Controller
Controls a motor using PWM on pin 21 while keeping pin 20 grounded.
"""

import RPi.GPIO as GPIO
import time

class SimpleMotorController:
    def __init__(self, pin1=20, pin2=21, pwm_frequency=100):
        """
        Initialize Simple Motor Controller with direction control
        
        Args:
            pin1 (int): Motor pin 1 (default: 20)
            pin2 (int): Motor pin 2 (default: 21)
            pwm_frequency (int): PWM frequency in Hz (default: 1000)
        """
        self.pin1 = pin1
        self.pin2 = pin2
        self.pwm_frequency = pwm_frequency
        self.pwm1 = None
        self.pwm2 = None
        self.current_speed = 0
        self.current_direction = 1  # 1 for forward, -1 for reverse
        
        self._setup_gpio()
        print("Simple Motor Controller with Direction Control initialized")
        print(f"Motor pins: {self.pin1}, {self.pin2}")
        print(f"PWM frequency: {self.pwm_frequency} Hz")
    
    def _setup_gpio(self):
        """Setup GPIO pins for bidirectional control"""
        try:
            # Set GPIO mode
            GPIO.setmode(GPIO.BCM)
            GPIO.setwarnings(False)
            
            # Clean up any existing setup
            try:
                GPIO.cleanup()
                time.sleep(0.1)
            except:
                pass
            
            # Setup both motor pins as outputs
            GPIO.setup(self.pin1, GPIO.OUT)
            GPIO.setup(self.pin2, GPIO.OUT)
            
            # Setup PWM on both pins
            self.pwm1 = GPIO.PWM(self.pin1, self.pwm_frequency)
            self.pwm2 = GPIO.PWM(self.pin2, self.pwm_frequency)
            
            # Start both PWM with 0% duty cycle
            self.pwm1.start(0)
            self.pwm2.start(0)
            
            print("GPIO setup completed")
            
        except Exception as e:
            print(f"Error during GPIO setup: {e}")
            self.cleanup()
            raise
    
    def set_speed(self, speed_percent, direction=None):
        """
        Set motor speed and direction
        
        Args:
            speed_percent (float): Speed as percentage (0-100)
            direction (int, optional): 1 for forward, -1 for reverse
        """
        try:
            # Clamp speed to valid range
            speed_percent = max(0, min(100, speed_percent))
            
            # Update direction if specified
            if direction is not None:
                self.current_direction = 1 if direction > 0 else -1
            
            self.current_speed = speed_percent
            
            if self.pwm1 is not None and self.pwm2 is not None:
                if speed_percent == 0:
                    # Stop motor - both pins to 0
                    self.pwm1.ChangeDutyCycle(0)
                    self.pwm2.ChangeDutyCycle(0)
                    print("Motor stopped")
                elif self.current_direction == 1:
                    # Forward - PWM on pin1, pin2 low
                    self.pwm1.ChangeDutyCycle(speed_percent)
                    self.pwm2.ChangeDutyCycle(0)
                    print(f"Motor FORWARD at {speed_percent}%")
                else:
                    # Reverse - PWM on pin2, pin1 low
                    self.pwm1.ChangeDutyCycle(0)
                    self.pwm2.ChangeDutyCycle(speed_percent)
                    print(f"Motor REVERSE at {speed_percent}%")
            else:
                print("PWM not initialized")
                
        except Exception as e:
            print(f"Error setting motor speed: {e}")
    
    def set_direction(self, direction):
        """
        Set motor direction
        
        Args:
            direction (int): 1 for forward, -1 for reverse
        """
        self.current_direction = 1 if direction > 0 else -1
        # Apply current speed with new direction
        self.set_speed(self.current_speed)
    
    def forward(self, speed_percent):
        """Set motor to move forward at specified speed"""
        self.set_speed(speed_percent, direction=1)
    
    def reverse(self, speed_percent):
        """Set motor to move reverse at specified speed"""
        self.set_speed(speed_percent, direction=-1)
    
    def stop(self):
        """Stop the motor"""
        self.set_speed(0)
        print("Motor stopped")
    
    def get_speed(self):
        """Get current motor speed"""
        return self.current_speed
    
    def get_direction(self):
        """Get current motor direction"""
        return "Forward" if self.current_direction == 1 else "Reverse"
    
    def get_status(self):
        """Get current motor status"""
        if self.current_speed == 0:
            return "Stopped"
        else:
            direction_text = "Forward" if self.current_direction == 1 else "Reverse"
            return f"{direction_text} at {self.current_speed}%"
    
    def test_sequence(self):
        """Run a simple test sequence with direction changes"""
        print("\nRunning test sequence with direction changes...")
        
        # Test forward speeds
        print("Testing FORWARD direction:")
        for speed in [25, 50, 75, 100]:
            print(f"Forward {speed}%")
            self.forward(speed)
            time.sleep(2)
        
        # Stop briefly
        self.stop()
        time.sleep(1)
        
        # Test reverse speeds
        print("Testing REVERSE direction:")
        for speed in [25, 50, 75, 100]:
            print(f"Reverse {speed}%")
            self.reverse(speed)
            time.sleep(2)
        
        # Stop
        self.stop()
        print("Test sequence completed")
    
    def interactive_control(self):
        """Interactive motor control with direction"""
        print("\n--- Simple Motor Control with Direction ---")
        print("Commands:")
        print("  <number>        - Set speed (0-100) in current direction")
        print("  f <number>      - Forward at speed (0-100)")
        print("  r <number>      - Reverse at speed (0-100)")
        print("  forward         - Set direction to forward")
        print("  reverse         - Set direction to reverse")
        print("  stop            - Stop motor")
        print("  test            - Run test sequence")
        print("  status          - Show current status")
        print("  quit            - Exit")
        print()
        
        try:
            while True:
                try:
                    command = input("Motor> ").strip().lower()
                    
                    if command in ['quit', 'exit', 'q']:
                        break
                    elif command == 'stop':
                        self.stop()
                    elif command == 'forward':
                        self.set_direction(1)
                        print("Direction set to FORWARD")
                    elif command == 'reverse':
                        self.set_direction(-1)
                        print("Direction set to REVERSE")
                    elif command == 'test':
                        self.test_sequence()
                    elif command == 'status':
                        print(f"Status: {self.get_status()}")
                    elif command.startswith('f '):
                        try:
                            speed = float(command.split()[1])
                            self.forward(speed)
                        except (IndexError, ValueError):
                            print("Usage: f <speed>")
                    elif command.startswith('r '):
                        try:
                            speed = float(command.split()[1])
                            self.reverse(speed)
                        except (IndexError, ValueError):
                            print("Usage: r <speed>")
                    elif command == '':
                        continue
                    else:
                        try:
                            # Try to parse as a number
                            speed = float(command)
                            self.set_speed(speed)
                        except ValueError:
                            print("Unknown command. Type 'quit' to exit.")
                            
                except KeyboardInterrupt:
                    print("\nExiting...")
                    break
                except Exception as e:
                    print(f"Error: {e}")
                    
        finally:
            self.stop()
    
    def cleanup(self):
        """Clean up GPIO resources"""
        try:
            if self.pwm1 is not None:
                self.pwm1.stop()
            if self.pwm2 is not None:
                self.pwm2.stop()
            GPIO.cleanup()
            print("GPIO cleanup completed")
        except Exception as e:
            print(f"Error during cleanup: {e}")
    
    def __del__(self):
        """Destructor - ensure cleanup"""
        self.cleanup()


def main():
    """Main function"""
    print("=== Simple Motor Controller with Direction Control ===")
    print("Controls motor on pins 21 and 20 with bidirectional control")
    print()
    
    try:
        # Initialize motor controller
        motor = SimpleMotorController(
            pin1=20,
            pin2=21,
            pwm_frequency=100
        )
        
        # Start interactive control
        motor.interactive_control()
        
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'motor' in locals():
            motor.cleanup()


if __name__ == "__main__":
    main()

import RPi.GPIO as GPIO
import time
import threading
from datetime import datetime

class MotorController:
    def __init__(self, motor_pin1=21, motor_pin2=20, encoder_pin_a=19, encoder_pin_b=21, pwm_frequency=1000):
        """
        Initialize Motor Controller
        
        Args:
            motor_pin1 (int): PWM pin for motor control (default: 21)
            motor_pin2 (int): Direction/ground pin for motor (default: 20)
            encoder_pin_a (int): Encoder channel A (default: 19)
            encoder_pin_b (int): Encoder channel B (default: 21)
            pwm_frequency (int): PWM frequency in Hz (default: 1000)
        """
        self.motor_pin1 = motor_pin1
        self.motor_pin2 = motor_pin2
        self.encoder_pin_a = encoder_pin_a
        self.encoder_pin_b = encoder_pin_b
        self.pwm_frequency = pwm_frequency
        
        # Encoder variables
        self.encoder_count = 0
        self.last_encoder_a = 0
        self.last_encoder_b = 0
        self.encoder_direction = 0  # 1 for forward, -1 for backward
        self.encoder_lock = threading.Lock()
        self.use_interrupts = True  # Will be set based on interrupt setup success
        
        # Motor control variables
        self.pwm = None
        self.current_speed = 0
        self.motor_running = False
        
        # Statistics
        self.start_time = None
        self.total_pulses = 0
        
        self._setup_gpio()
        print("Motor Controller initialized successfully")
        print(f"Motor pins: {self.motor_pin1} (PWM), {self.motor_pin2} (GND)")
        print(f"Encoder pins: {self.encoder_pin_a} (A), {self.encoder_pin_b} (B)")
        print(f"PWM frequency: {self.pwm_frequency} Hz")
    
    def _setup_gpio(self):
        """Setup GPIO pins and interrupts"""
        try:
            # Set GPIO mode
            GPIO.setmode(GPIO.BCM)
            GPIO.setwarnings(False)
            
            # Clean up any existing setup first
            try:
                GPIO.cleanup()
                time.sleep(0.1)  # Give time for cleanup
            except:
                pass
            
            # Setup motor pins
            GPIO.setup(self.motor_pin1, GPIO.OUT)  # PWM pin
            GPIO.setup(self.motor_pin2, GPIO.OUT)  # Ground/direction pin
            
            # Keep motor_pin2 low (ground)
            GPIO.output(self.motor_pin2, GPIO.LOW)
            
            # Setup PWM on motor_pin1
            self.pwm = GPIO.PWM(self.motor_pin1, self.pwm_frequency)
            self.pwm.start(0)  # Start with 0% duty cycle (motor off)
            
            # Setup encoder pins with pull-up resistors
            GPIO.setup(self.encoder_pin_a, GPIO.IN, pull_up_down=GPIO.PUD_UP)
            GPIO.setup(self.encoder_pin_b, GPIO.IN, pull_up_down=GPIO.PUD_UP)
            
            # Read initial encoder states
            self.last_encoder_a = GPIO.input(self.encoder_pin_a)
            self.last_encoder_b = GPIO.input(self.encoder_pin_b)
            
            # Setup interrupts for encoder with error handling
            try:
                GPIO.add_event_detect(self.encoder_pin_a, GPIO.BOTH, callback=self._encoder_callback, bouncetime=2)
                time.sleep(0.1)
                GPIO.add_event_detect(self.encoder_pin_b, GPIO.BOTH, callback=self._encoder_callback, bouncetime=2)
                print("Encoder interrupts setup successfully")
            except Exception as e:
                print(f"Warning: Could not setup interrupts ({e}). Will use polling method.")
                # Remove any partial interrupt setup
                try:
                    GPIO.remove_event_detect(self.encoder_pin_a)
                    GPIO.remove_event_detect(self.encoder_pin_b)
                except:
                    pass
                # Add polling method as fallback
                self.use_interrupts = False
            else:
                self.use_interrupts = True
            
            print("GPIO setup completed successfully")
            
        except Exception as e:
            print(f"Error during GPIO setup: {e}")
            self.cleanup()
            raise
    
    def _encoder_callback(self, channel):
        """
        Interrupt callback for encoder pins
        Uses quadrature encoding to determine direction and count pulses
        """
        try:
            with self.encoder_lock:
                # Read current states
                current_a = GPIO.input(self.encoder_pin_a)
                current_b = GPIO.input(self.encoder_pin_b)
                
                # Quadrature encoding logic
                if channel == self.encoder_pin_a:
                    if current_a != self.last_encoder_a:
                        if current_a == current_b:
                            # Clockwise rotation
                            self.encoder_count += 1
                            self.encoder_direction = 1
                        else:
                            # Counter-clockwise rotation
                            self.encoder_count -= 1
                            self.encoder_direction = -1
                        self.total_pulses += 1
                        self.last_encoder_a = current_a
                
                elif channel == self.encoder_pin_b:
                    if current_b != self.last_encoder_b:
                        if current_a == current_b:
                            # Counter-clockwise rotation
                            self.encoder_count -= 1
                            self.encoder_direction = -1
                        else:
                            # Clockwise rotation
                            self.encoder_count += 1
                            self.encoder_direction = 1
                        self.total_pulses += 1
                        self.last_encoder_b = current_b
                        
        except Exception as e:
            print(f"Error in encoder callback: {e}")
    
    def _poll_encoder(self):
        """
        Polling method for encoder reading (fallback when interrupts fail)
        """
        try:
            with self.encoder_lock:
                # Read current states
                current_a = GPIO.input(self.encoder_pin_a)
                current_b = GPIO.input(self.encoder_pin_b)
                
                # Simple change detection
                if current_a != self.last_encoder_a:
                    if current_a == current_b:
                        self.encoder_count += 1  # Clockwise
                        self.encoder_direction = 1
                    else:
                        self.encoder_count -= 1  # Counter-clockwise
                        self.encoder_direction = -1
                    self.total_pulses += 1
                    self.last_encoder_a = current_a
                
                if current_b != self.last_encoder_b:
                    if current_a == current_b:
                        self.encoder_count -= 1  # Counter-clockwise
                        self.encoder_direction = -1
                    else:
                        self.encoder_count += 1  # Clockwise
                        self.encoder_direction = 1
                    self.total_pulses += 1
                    self.last_encoder_b = current_b
                    
        except Exception as e:
            print(f"Error polling encoder: {e}")
    
    def update_encoder(self):
        """Update encoder count (call this regularly if using polling)"""
        if not self.use_interrupts:
            self._poll_encoder()
    
    def set_motor_speed(self, speed_percent):
        """
        Set motor speed using PWM
        
        Args:
            speed_percent (float): Speed as percentage (0-100)
        """
        try:
            # Clamp speed to valid range
            speed_percent = max(0, min(100, speed_percent))
            
            if self.pwm is not None:
                self.pwm.ChangeDutyCycle(speed_percent)
                self.current_speed = speed_percent
                
                if speed_percent > 0:
                    if not self.motor_running:
                        self.motor_running = True
                        self.start_time = datetime.now()
                        print(f"Motor started at {speed_percent}% speed")
                    else:
                        print(f"Motor speed changed to {speed_percent}%")
                else:
                    if self.motor_running:
                        self.motor_running = False
                        print("Motor stopped")
                        
        except Exception as e:
            print(f"Error setting motor speed: {e}")
    
    def get_encoder_count(self):
        """Get current encoder count (thread-safe)"""
        with self.encoder_lock:
            return self.encoder_count
    
    def reset_encoder_count(self):
        """Reset encoder count to zero (thread-safe)"""
        with self.encoder_lock:
            self.encoder_count = 0
            self.total_pulses = 0
            print("Encoder count reset to zero")
    
    def get_motor_stats(self):
        """Get motor and encoder statistics"""
        with self.encoder_lock:
            current_count = self.encoder_count
            current_total = self.total_pulses
            current_direction = self.encoder_direction
        
        stats = {
            'motor_speed_percent': self.current_speed,
            'motor_running': self.motor_running,
            'encoder_count': current_count,
            'total_pulses': current_total,
            'encoder_direction': current_direction,
            'direction_text': 'Clockwise' if current_direction == 1 else 'Counter-clockwise' if current_direction == -1 else 'Stopped'
        }
        
        if self.start_time and self.motor_running:
            runtime = (datetime.now() - self.start_time).total_seconds()
            stats['runtime_seconds'] = runtime
            if runtime > 0:
                stats['pulses_per_second'] = current_total / runtime
        
        return stats
    
    def print_stats(self):
        """Print current motor and encoder statistics"""
        # Update encoder if using polling
        if not self.use_interrupts:
            self.update_encoder()
            
        stats = self.get_motor_stats()
        print(f"\n--- Motor Controller Stats ---")
        print(f"Encoder Method: {'Interrupts' if self.use_interrupts else 'Polling'}")
        print(f"Motor Speed: {stats['motor_speed_percent']}%")
        print(f"Motor Running: {stats['motor_running']}")
        print(f"Encoder Count: {stats['encoder_count']}")
        print(f"Total Pulses: {stats['total_pulses']}")
        print(f"Direction: {stats['direction_text']}")
        
        if 'runtime_seconds' in stats:
            print(f"Runtime: {stats['runtime_seconds']:.1f} seconds")
        if 'pulses_per_second' in stats:
            print(f"Pulse Rate: {stats['pulses_per_second']:.1f} pulses/sec")
        print("-" * 30)
    
    def run_motor_test(self, duration=10, speed=50):
        """
        Run a motor test for specified duration
        
        Args:
            duration (int): Test duration in seconds
            speed (float): Motor speed percentage (0-100)
        """
        print(f"\nStarting motor test: {speed}% speed for {duration} seconds")
        print("Press Ctrl+C to stop early")
        
        try:
            self.reset_encoder_count()
            self.set_motor_speed(speed)
            
            for i in range(duration):
                time.sleep(1)
                if not self.use_interrupts:
                    self.update_encoder()
                self.print_stats()
                
        except KeyboardInterrupt:
            print("\nTest interrupted by user")
        finally:
            self.set_motor_speed(0)
            print("\nTest completed")
            self.print_stats()
    
    def interactive_control(self):
        """Interactive motor control mode"""
        print("\n--- Interactive Motor Control ---")
        print("Commands:")
        print("  speed <0-100>  - Set motor speed percentage")
        print("  stop           - Stop motor")
        print("  reset          - Reset encoder count")
        print("  stats          - Show statistics")
        print("  test <speed>   - Run 10-second test at specified speed")
        print("  quit           - Exit program")
        print()
        
        try:
            while True:
                try:
                    command = input("Motor> ").strip().lower()
                    
                    if command == 'quit' or command == 'exit':
                        break
                    elif command == 'stop':
                        self.set_motor_speed(0)
                    elif command == 'reset':
                        self.reset_encoder_count()
                    elif command == 'stats':
                        self.print_stats()
                    elif command.startswith('speed '):
                        try:
                            speed = float(command.split()[1])
                            self.set_motor_speed(speed)
                        except (IndexError, ValueError):
                            print("Usage: speed <0-100>")
                    elif command.startswith('test '):
                        try:
                            speed = float(command.split()[1])
                            self.run_motor_test(duration=10, speed=speed)
                        except (IndexError, ValueError):
                            print("Usage: test <speed>")
                    elif command == '':
                        continue
                    else:
                        print("Unknown command. Type 'quit' to exit.")
                        
                except KeyboardInterrupt:
                    print("\nExiting...")
                    break
                except Exception as e:
                    print(f"Error: {e}")
                    
        finally:
            self.set_motor_speed(0)
    
    def cleanup(self):
        """Clean up GPIO resources"""
        try:
            if self.pwm is not None:
                self.pwm.stop()
            GPIO.cleanup()
            print("GPIO cleanup completed")
        except Exception as e:
            print(f"Error during cleanup: {e}")
    
    def __del__(self):
        """Destructor - ensure cleanup"""
        self.cleanup()


def main():
    """Main function"""
    print("=== Motor Controller with PWM and Encoder ===")
    print("This program controls a motor using PWM and counts encoder pulses.")
    print()
    
    try:
        # Initialize motor controller
        motor = MotorController(
            motor_pin1=20,    # PWM pin
            motor_pin2=21,    # Ground pin
            encoder_pin_a=19, # Encoder A
            encoder_pin_b=21, # Encoder B
            pwm_frequency=1000
        )
        
        # Show initial stats
        motor.print_stats()
        
        # Start interactive control
        motor.interactive_control()
        
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'motor' in locals():
            motor.cleanup()


if __name__ == "__main__":
    main()
