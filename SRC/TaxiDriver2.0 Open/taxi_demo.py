#!/usr/bin/env python3
"""
Simple demonstration of the TaxiDriver class
Shows how to control both motor and steering programmatically
"""

import time
from taxi_driver import TaxiDriver

def main():
    """Demonstration of programmatic motor and steering control"""
    taxi = None
    try:
        print("=== Taxi Driver Demo with Steering ===")
        
        # Initialize taxi driver
        taxi = TaxiDriver()
        taxi.start_all_controllers()
        
        # Give controllers time to initialize
        print("Initializing controllers...")
        time.sleep(4)
        
        # Get initial readings
        initial_encoder = taxi.get_encoder_reading()
        initial_angle = taxi.get_current_angle()
        print(f"Initial encoder reading: {initial_encoder}")
        print(f"Initial angle: {initial_angle:.1f}°")
        
        # Demo 1: Basic movement with current steering
        print("\n1. Moving forward at 30% speed for 2 seconds (current steering)...")
        taxi.move_forward(30)
        time.sleep(2)
        
        encoder_reading = taxi.get_encoder_reading()
        current_angle = taxi.get_current_angle()
        print(f"Encoder after forward: {encoder_reading}, Current angle: {current_angle:.1f}°")
        
        # Stop briefly
        print("\n2. Stopping for 1 second...")
        taxi.stop_motor()
        time.sleep(1)
        
        # Demo 2: Test steering control
        print("\n3. Testing steering control...")
        current_angle = taxi.get_current_angle()
        target_left = current_angle + 20  # Turn left 20 degrees
        target_right = current_angle - 20  # Turn right 20 degrees
        
        print(f"   Setting target angle to {target_left:.1f}° (left turn)...")
        taxi.set_target_angle(target_left)
        time.sleep(3)  # Wait for servo to adjust
        
        gyro_status = taxi.get_gyroscope_status()
        print(f"   Gyro status: {gyro_status['status']}")
        
        print(f"   Setting target angle to {target_right:.1f}° (right turn)...")
        taxi.set_target_angle(target_right)
        time.sleep(3)  # Wait for servo to adjust
        
        gyro_status = taxi.get_gyroscope_status()
        print(f"   Gyro status: {gyro_status['status']}")
        
        # Demo 3: Movement with steering
        print("\n4. Testing movement with steering...")
        print("   Moving forward while steering left...")
        taxi.set_target_angle(target_left)
        taxi.move_forward(25)
        time.sleep(2)
        
        print("   Moving forward while steering right...")
        taxi.set_target_angle(target_right)
        time.sleep(2)
        
        print("   Centering steering and stopping...")
        taxi.set_target_angle(initial_angle)
        taxi.stop_motor()
        time.sleep(2)
        
        # Demo 4: Show final status
        print("\n5. Final system status...")
        full_status = taxi.get_full_status()
        motor_status = full_status['motor']
        gyro_status = full_status['gyroscope']
        
        final_encoder = taxi.get_encoder_reading()
        total_change = final_encoder - initial_encoder
        
        print(f"\n=== Demo Complete ===")
        print(f"Motor status: {motor_status['status']}")
        print(f"Final encoder reading: {final_encoder}")
        print(f"Total encoder change: {total_change}")
        print(f"Gyroscope status: {gyro_status['status']}")
        print(f"Final servo position: {gyro_status['servo_position']}")
        
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    except Exception as e:
        print(f"Demo error: {e}")
    finally:
        if taxi:
            taxi.cleanup()
        print("Demo finished")

if __name__ == "__main__":
    main()
