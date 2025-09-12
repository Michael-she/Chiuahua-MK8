#!/usr/bin/env python3
"""
Test script for the new LiDAR methods: get_furthest_point and get_closest_point
"""

import sys
import time
from taxi_driver import TaxiDriver

def test_lidar_methods():
    """Test the new LiDAR methods"""
    print("=== Testing LiDAR Methods ===")
    
    # Initialize the taxi driver
    taxi = TaxiDriver()
    
    # Start only the LiDAR controller for testing
    print("Starting LiDAR controller...")
    taxi.start_lidar_controller()
    
    # Wait for LiDAR to start collecting data
    print("Waiting for LiDAR data collection...")
    time.sleep(5)
    
    try:
        # Test different angles
        test_angles = [0, 30, -30, 45, -45, 90, -90]
        
        for angle in test_angles:
            print(f"\n--- Testing angle {angle}° ---")
            
            # Test get_furthest_point
            furthest = taxi.get_furthest_point(angle)
            if furthest is not None:
                print(f"95th percentile furthest point at {angle}°: {furthest:.1f} mm ({furthest/1000:.2f} m)")
            else:
                print(f"No furthest point found at {angle}°")
            
            # Test get_closest_point
            closest = taxi.get_closest_point(angle)
            if closest is not None:
                print(f"Closest point at {angle}°: {closest:.1f} mm ({closest/1000:.2f} m)")
            else:
                print(f"No closest point found at {angle}°")
            
            # Compare with existing distance method
            distance = taxi.get_distance(angle)
            if distance is not None and distance != 9999:
                print(f"Median distance at {angle}° (±2.5°): {distance:.1f} mm ({distance/1000:.2f} m)")
            else:
                print(f"No median distance found at {angle}°")
            
            # Show LiDAR point count for context
            points_count = taxi.get_lidar_points_count()
            print(f"Total LiDAR points: {points_count}")
    
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    
    finally:
        print("\nStopping LiDAR controller...")
        taxi.stop_lidar_controller()
        print("Test completed")

if __name__ == '__main__':
    test_lidar_methods()
