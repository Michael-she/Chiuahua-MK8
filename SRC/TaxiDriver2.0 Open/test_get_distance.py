#!/usr/bin/env python3
"""
Test script to demonstrate the get_distance method
"""

import time
from taxi_driver import TaxiDriver

def main():
    """Test the get_distance method"""
    print("=== get_distance() Method Test ===")
    
    taxi = None
    try:
        # Initialize taxi driver
        taxi = TaxiDriver()
        
        # Start only the LiDAR controller for this test
        print("Starting LiDAR controller...")
        taxi.start_lidar_controller()
        
        # Give it time to initialize
        print("Waiting for LiDAR to initialize...")
        time.sleep(5)
        
        # Test different angles
        test_angles = [0, 30, -30, 45, -45, 60, -60, 90, -90]
        
        print("\n=== Testing get_distance() at various angles ===")
        print("Note: 0° = forward, positive = clockwise, negative = counter-clockwise")
        print("Each measurement includes points within ±2.5° of the target angle\n")
        
        for iteration in range(3):
            print(f"--- Measurement Round {iteration + 1} ---")
            
            for angle in test_angles:
                distance = taxi.get_distance(angle)
                if distance is not None:
                    print(f"  {angle:3d}°: {distance:6.1f} mm ({distance/1000:.2f} m)")
                else:
                    print(f"  {angle:3d}°: No points found")
            
            print()
            time.sleep(3)
        
        # Test some specific use cases
        print("=== Practical Use Cases ===")
        
        # Forward clearance
        forward_distance = taxi.get_distance(0)
        if forward_distance is not None:
            print(f"Forward clearance: {forward_distance:.1f} mm ({forward_distance/1000:.2f} m)")
        else:
            print("No forward clearance data available")
        
        # Left and right sides
        left_distance = taxi.get_distance(-90)
        right_distance = taxi.get_distance(90)
        
        if left_distance is not None:
            print(f"Left side clearance: {left_distance:.1f} mm ({left_distance/1000:.2f} m)")
        else:
            print("No left side data available")
            
        if right_distance is not None:
            print(f"Right side clearance: {right_distance:.1f} mm ({right_distance/1000:.2f} m)")
        else:
            print("No right side data available")
        
        # Corner checks (useful for turning)
        front_left = taxi.get_distance(-45)
        front_right = taxi.get_distance(45)
        
        if front_left is not None:
            print(f"Front-left diagonal: {front_left:.1f} mm ({front_left/1000:.2f} m)")
        else:
            print("No front-left diagonal data available")
            
        if front_right is not None:
            print(f"Front-right diagonal: {front_right:.1f} mm ({front_right/1000:.2f} m)")
        else:
            print("No front-right diagonal data available")
        
        print("\n=== Test completed successfully ===")
        
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"Test error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if taxi:
            print("Stopping LiDAR controller...")
            taxi.stop_lidar_controller()
            print("Test cleanup complete")

if __name__ == "__main__":
    main()
