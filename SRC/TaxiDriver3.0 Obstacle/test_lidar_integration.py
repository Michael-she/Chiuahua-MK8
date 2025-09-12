#!/usr/bin/env python3
"""
Test script to demonstrate LiDAR multiprocessing integration
"""

import time
import math
from taxi_driver import TaxiDriver

def main():
    """Test the LiDAR integration with the taxi driver"""
    print("=== LiDAR Integration Test ===")
    
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
        
        # Test LiDAR data access
        print("\n=== LiDAR Data Test ===")
        for i in range(10):
            print(f"\n--- Test iteration {i+1} ---")
            
            # Get LiDAR status
            status = taxi.get_lidar_status()
            print(f"LiDAR Status: {status}")
            
            # Get point count
            count = taxi.get_lidar_points_count()
            print(f"Points available: {count}")
            
            # Get some sample points
            points = taxi.get_lidar_points()
            if points:
                print(f"Sample points (showing first 5 of {len(points)}):")
                for j, (x, y) in enumerate(points[:5]):
                    distance = math.sqrt(x**2 + y**2)
                    angle_deg = math.degrees(math.atan2(x, y))
                    print(f"  Point {j+1}: ({x:.1f}, {y:.1f}) mm, distance: {distance:.1f} mm, angle: {angle_deg:.1f}°")
            else:
                print("No points available")
            
            # Check for obstacles
            closest = taxi.get_closest_obstacle_distance()
            if closest is not None:
                print(f"Closest obstacle: {closest:.1f} mm ({closest/1000:.2f} m)")
            else:
                print("No obstacles detected in front")
            
            time.sleep(2)
        
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
