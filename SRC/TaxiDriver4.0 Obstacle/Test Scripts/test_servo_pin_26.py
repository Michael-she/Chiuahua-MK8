#!/usr/bin/env python3
"""
Test script for the servo connected to pin 26
Demonstrates how to use the servo control methods in TaxiDriver
"""

import time
import sys
from taxi_driver import TaxiDriver

def test_servo_pin_26():
    """Test the servo control on pin 26"""
    print("=== Servo Pin 26 Test ===")
    
    # Create TaxiDriver instance
    taxi = TaxiDriver()
    
    try:
        # Initialize servo
        print("\n1. Initializing servo on pin 26...")
        if not taxi.initialize_servo_pin_26():
            print("Failed to initialize servo. Make sure pigpio daemon is running.")
            sys.exit(1)
        
        # Test different angles
        test_angles = [0, 45, 90, 135, 180, 90]  # End at center
        
        print("\n2. Testing different servo angles...")
        for angle in test_angles:
            print(f"Setting servo to {angle}°...")
            taxi.set_servo_angle_pin_26(angle)
            
            # Show current angle
            current = taxi.get_servo_angle_pin_26()
            print(f"Current servo angle: {current}°")
            
            time.sleep(2)  # Wait 2 seconds between movements
        
        # Test center command
        print("\n3. Testing center command...")
        taxi.center_servo_pin_26()
        current = taxi.get_servo_angle_pin_26()
        print(f"Servo centered at: {current}°")
        
        print("\n4. Test completed successfully!")
        
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"Error during test: {e}")
    finally:
        # Clean up
        print("\nCleaning up...")
        taxi.cleanup_servo_pin_26()
        print("Servo test finished")

def interactive_servo_test():
    """Interactive servo control for testing"""
    print("=== Interactive Servo Control (Pin 26) ===")
    print("Commands:")
    print("  <angle>     - Set servo to angle (0-180)")
    print("  center      - Center servo to 90°")
    print("  status      - Show current angle")
    print("  sweep       - Perform a sweep test")
    print("  quit        - Exit")
    print()
    
    # Create TaxiDriver instance
    taxi = TaxiDriver()
    
    try:
        # Initialize servo
        if not taxi.initialize_servo_pin_26():
            print("Failed to initialize servo. Make sure pigpio daemon is running.")
            return
        
        print("Servo initialized. Ready for commands!")
        
        while True:
            try:
                command = input("Servo> ").strip().lower()
                
                if command in ['quit', 'exit', 'q']:
                    break
                elif command == 'center':
                    taxi.center_servo_pin_26()
                    current = taxi.get_servo_angle_pin_26()
                    print(f"Servo centered at: {current}°")
                elif command == 'status':
                    current = taxi.get_servo_angle_pin_26()
                    print(f"Current servo angle: {current}°")
                elif command == 'sweep':
                    print("Performing servo sweep...")
                    for angle in range(0, 181, 30):
                        print(f"  Setting to {angle}°...")
                        taxi.set_servo_angle_pin_26(angle)
                        time.sleep(1)
                    for angle in range(150, -1, -30):
                        print(f"  Setting to {angle}°...")
                        taxi.set_servo_angle_pin_26(angle)
                        time.sleep(1)
                    taxi.center_servo_pin_26()
                    print("Sweep completed, servo centered")
                else:
                    try:
                        angle = float(command)
                        if 0 <= angle <= 180:
                            taxi.set_servo_angle_pin_26(angle)
                            current = taxi.get_servo_angle_pin_26()
                            print(f"Servo set to: {current}°")
                        else:
                            print("Angle must be between 0 and 180 degrees")
                    except ValueError:
                        print("Invalid command. Enter an angle (0-180), 'center', 'status', 'sweep', or 'quit'")
                        
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"Error: {e}")
                
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Clean up
        print("Cleaning up...")
        taxi.cleanup_servo_pin_26()
        print("Interactive servo test finished")

if __name__ == "__main__":
    print("Servo Pin 26 Test Options:")
    print("1. Automated test")
    print("2. Interactive control")
    
    try:
        choice = input("Choose option (1 or 2): ").strip()
        
        if choice == "1":
            test_servo_pin_26()
        elif choice == "2":
            interactive_servo_test()
        else:
            print("Invalid choice. Running automated test...")
            test_servo_pin_26()
            
    except KeyboardInterrupt:
        print("\nExiting...")
