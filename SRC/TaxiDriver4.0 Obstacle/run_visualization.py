#!/usr/bin/env python3
"""
Standalone Object Visualization for TaxiDriver
Displays relative positions of detected objects using camera angles and LiDAR distances
"""

import sys
import time
from taxi_driver import TaxiDriver

def main():
    """Main function to run the visualization"""
    print("=== TaxiDriver Object Visualization ===")
    print("This will show detected objects in real-time using camera and LiDAR data")
    print("Make sure objects are visible to the camera and within LiDAR range")
    print()
    
    taxi = None
    try:
        # Initialize taxi driver
        taxi = TaxiDriver()
        
        # Start all controllers
        print("Starting all controllers...")
        taxi.start_all_controllers()
        
        # Wait for controllers to initialize
        print("Waiting for controllers to initialize...")
        time.sleep(5)
        
        # Check if controllers are running
        motor_status = taxi.get_motor_status()
        gyro_status = taxi.get_gyroscope_status()
        lidar_status = taxi.get_lidar_status()
        camera_status = taxi.get_camera_status()
        
        print(f"Motor: {motor_status.get('status', 'Unknown')}")
        print(f"Gyroscope: {gyro_status.get('status', 'Unknown')}")
        print(f"LiDAR: {lidar_status.get('status', 'Unknown')}")
        print(f"Camera: {camera_status.get('status', 'Unknown')}")
        print()
        
        # Start visualization
        print("Starting visualization...")
        print("- Green dots: LiDAR points")
        print("- Colored circles: Detected objects (green/red)")
        print("- White circle: Robot position")
        print("- Green line: Forward direction")
        print("- Press 'q' or ESC to exit")
        print()
        
        input("Press Enter to start visualization...")
        
        taxi.start_object_visualization()
        
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if taxi:
            print("Cleaning up...")
            taxi.cleanup()
        print("Visualization stopped")

if __name__ == '__main__':
    main()
