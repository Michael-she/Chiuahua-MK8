#!/usr/bin/env python3
"""
Simple Motor Controller with Encoder Support
Controls a motor using PWM on pins 20/21 with encoder feedback on pins 19/13.
"""

import RPi.GPIO as GPIO
import time
import threading

class SimpleMotorController:
    def __init__(self, pin1=21, pin2=20, encoder_pin_a=19, encoder_pin_b=13, pwm_frequency= 2000):
        """
        Initialize Simple Motor Controller with direction control and encoder support
        
        Args:
            pin1 (int): Motor pin 1 (default: 20)
            pin2 (int): Motor pin 2 (default: 21)
            encoder_pin_a (int): Encoder channel A (default: 19)
            encoder_pin_b (int): Encoder channel B (default: 13)
            pwm_frequency (int): PWM frequency in Hz (default: 20000)
        """
        self.pin1 = pin1
        self.pin2 = pin2
        self.encoder_pin_a = encoder_pin_a
        self.encoder_pin_b = encoder_pin_b
        self.pwm_frequency = pwm_frequency
        self.pwm1 = None
        self.pwm2 = None
        self.current_speed = 0
        self.current_direction = 1  # 1 for forward, -1 for reverse
        
        # Encoder variables
        self.encoder_count = 0
        self.encoder_lock = threading.Lock()
        self.last_encoder_a = 0
        self.last_encoder_b = 0
        self.polling_active = False
        self.polling_thread = None
        
        self._setup_gpio()
        print("Simple Motor Controller with Direction Control and Encoders initialized")
        print(f"Motor pins: {self.pin1}, {self.pin2}")
        print(f"Encoder pins: {self.encoder_pin_a}, {self.encoder_pin_b}")
        print(f"PWM frequency: {self.pwm_frequency} Hz")
    
    def _setup_gpio(self):
        """Setup GPIO pins for bidirectional control and encoders"""
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
            
            # Setup encoder pins as inputs with pull-up resistors
            GPIO.setup(self.encoder_pin_a, GPIO.IN, pull_up_down=GPIO.PUD_UP)
            GPIO.setup(self.encoder_pin_b, GPIO.IN, pull_up_down=GPIO.PUD_UP)
            
            # Setup PWM on both pins
            self.pwm1 = GPIO.PWM(self.pin1, self.pwm_frequency)
            self.pwm2 = GPIO.PWM(self.pin2, self.pwm_frequency)
            
            # Start both PWM with 0% duty cycle
            self.pwm1.start(0)
            self.pwm2.start(0)
            
            # Setup encoder interrupts
            self._setup_encoder_interrupts()
            
            print("GPIO setup completed")
            
        except Exception as e:
            print(f"Error during GPIO setup: {e}")
            self.cleanup()
            raise
    
    def _setup_encoder_interrupts(self):
        """Setup encoder interrupt handling with fallback to polling"""
        try:
            # Try to set up interrupts
            GPIO.add_event_detect(self.encoder_pin_a, GPIO.BOTH, callback=self._encoder_callback, bouncetime=1)
            print("Encoder interrupts enabled")
        except Exception as e:
            print(f"Warning: Could not setup encoder interrupts: {e}")
            print("Falling back to polling method")
            # Start polling thread as fallback
            self.polling_thread = threading.Thread(target=self._encoder_polling, daemon=True)
            self.polling_active = True
            self.polling_thread.start()
    
    def _encoder_callback(self, channel):
        """Encoder interrupt callback for quadrature decoding"""
        try:
            with self.encoder_lock:
                # Read current states
                state_a = GPIO.input(self.encoder_pin_a)
                state_b = GPIO.input(self.encoder_pin_b)
                
                # Simple quadrature decoding
                if state_a != self.last_encoder_a:
                    if state_a == state_b:
                        self.encoder_count += 1
                    else:
                        self.encoder_count -= 1
                
                self.last_encoder_a = state_a
                self.last_encoder_b = state_b
        except Exception as e:
            print(f"Encoder callback error: {e}")
    
    def _encoder_polling(self):
        """Fallback polling method for encoder reading"""
        try:
            last_a = GPIO.input(self.encoder_pin_a)
            last_b = GPIO.input(self.encoder_pin_b)
            
            while getattr(self, 'polling_active', True):
                try:
                    current_a = GPIO.input(self.encoder_pin_a)
                    current_b = GPIO.input(self.encoder_pin_b)
                    
                    if current_a != last_a:
                        with self.encoder_lock:
                            if current_a == current_b:
                                self.encoder_count += 1
                            else:
                                self.encoder_count -= 1
                    
                    last_a = current_a
                    last_b = current_b
                    time.sleep(0.001)  # 1ms polling interval
                except Exception as e:
                    print(f"Encoder polling error: {e}")
                    break
        except Exception as e:
            print(f"Error starting encoder polling: {e}")
    
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
    
    # Encoder methods
    def get_encoder_count(self):
        """Get current encoder count"""
        with self.encoder_lock:
            return self.encoder_count
    
    def reset_encoder_count(self):
        """Reset encoder count to zero"""
        with self.encoder_lock:
            self.encoder_count = 0
        print("Encoder count reset to 0")
    
    def get_encoder_status(self):
        """Get encoder status including current readings"""
        try:
            state_a = GPIO.input(self.encoder_pin_a)
            state_b = GPIO.input(self.encoder_pin_b)
            count = self.get_encoder_count()
            return {
                'count': count,
                'pin_a_state': state_a,
                'pin_b_state': state_b,
                'pin_a': self.encoder_pin_a,
                'pin_b': self.encoder_pin_b
            }
        except Exception as e:
            print(f"Error reading encoder status: {e}")
            return None
    
    def test_sequence_with_encoders(self):
        """Run a test sequence with encoder monitoring"""
        print("\nRunning test sequence with encoder monitoring...")
        
        # Reset encoder count
        self.reset_encoder_count()
        
        # Test forward speeds
        print("Testing FORWARD direction:")
        for speed in [25, 50, 75]:
            initial_count = self.get_encoder_count()
            print(f"Forward {speed}% - Starting encoder count: {initial_count}")
            self.forward(speed)
            time.sleep(3)
            final_count = self.get_encoder_count()
            change = final_count - initial_count
            print(f"  Encoder change: {change} counts")
        
        # Stop briefly
        self.stop()
        time.sleep(1)
        
        # Test reverse speeds
        print("Testing REVERSE direction:")
        for speed in [25, 50, 75]:
            initial_count = self.get_encoder_count()
            print(f"Reverse {speed}% - Starting encoder count: {initial_count}")
            self.reverse(speed)
            time.sleep(3)
            final_count = self.get_encoder_count()
            change = final_count - initial_count
            print(f"  Encoder change: {change} counts")
        
        # Stop
        self.stop()
        print(f"Test sequence completed. Final encoder count: {self.get_encoder_count()}")
    
    def interactive_control(self):
        """Interactive motor control with direction and encoder feedback"""
        print("\n--- Simple Motor Control with Direction and Encoders ---")
        print("Commands:")
        print("  <number>        - Set speed (0-100) in current direction")
        print("  f <number>      - Forward at speed (0-100)")
        print("  r <number>      - Reverse at speed (0-100)")
        print("  forward         - Set direction to forward")
        print("  reverse         - Set direction to reverse")
        print("  stop            - Stop motor")
        print("  test            - Run test sequence with encoder monitoring")
        print("  status          - Show current motor and encoder status")
        print("  encoder         - Show encoder status")
        print("  reset           - Reset encoder count")
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
                        self.test_sequence_with_encoders()
                    elif command == 'status':
                        status = self.get_status()
                        encoder_count = self.get_encoder_count()
                        print(f"Motor: {status}")
                        print(f"Encoder: {encoder_count} counts")
                    elif command == 'encoder':
                        encoder_status = self.get_encoder_status()
                        if encoder_status:
                            print(f"Encoder Count: {encoder_status['count']}")
                            print(f"Pin {encoder_status['pin_a']} (A): {encoder_status['pin_a_state']}")
                            print(f"Pin {encoder_status['pin_b']} (B): {encoder_status['pin_b_state']}")
                    elif command == 'reset':
                        self.reset_encoder_count()
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
                    elif command.replace('.', '').isdigit():
                        speed = float(command)
                        self.set_speed(speed)
                    else:
                        print("Unknown command. Type 'quit' to exit.")
                        
                except KeyboardInterrupt:
                    print("\nInterrupted by user")
                    break
                except Exception as e:
                    print(f"Error processing command: {e}")
                    
        except KeyboardInterrupt:
            print("\nExiting...")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up GPIO and stop threads"""
        try:
            print("Cleaning up...")
            
            # Stop polling thread if running
            if hasattr(self, 'polling_active'):
                self.polling_active = False
            if hasattr(self, 'polling_thread') and self.polling_thread:
                self.polling_thread.join(timeout=1)
            
            # Stop PWM
            if self.pwm1:
                self.pwm1.stop()
            if self.pwm2:
                self.pwm2.stop()
                
            # Clean up GPIO
            GPIO.cleanup()
            print("Cleanup completed")
            
        except Exception as e:
            print(f"Error during cleanup: {e}")

def main():
    """Main function to run the motor controller"""
    controller = None
    try:
        # Initialize motor controller with encoder pins 19 and 13
        controller = SimpleMotorController(pin1=21, pin2=20, encoder_pin_a=19, encoder_pin_b=13)
        
        # Start interactive control
        controller.interactive_control()
        
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if controller:
            controller.cleanup()

if __name__ == "__main__":
    main()
