#!/usr/bin/env python3
"""
Test script for camera controller integration with TaxiDriver
"""

import sys
import time
from taxi_driver import TaxiDriver

def test_camera_integration():
    """Test the camera controller integration"""
    print("=== Testing Camera Controller Integration ===")
    
    # Initialize the taxi driver
    taxi = TaxiDriver()
    
    # Start only the camera controller for testing
    print("Starting camera controller...")
    taxi.start_camera_controller()
    
    # Wait for camera to start collecting data
    print("Waiting for camera data collection...")
    time.sleep(5)
    
    try:
        for i in range(20):  # Test for 20 iterations
            print(f"\n--- Test iteration {i+1} ---")
            
            # Test camera status
            status = taxi.get_camera_status()
            print(f"Camera Status: {status}")
            
            # Test detection count
            count = taxi.get_camera_detection_count()
            print(f"Detection count: {count}")
            
            # Test getting all detections
            detections = taxi.get_camera_detections()
            if detections:
                print(f"Found {len(detections)} objects:")
                for j, det in enumerate(detections):
                    bearing_str = f"{det['bearing']:.1f}°" if det['bearing'] is not None else "Unknown"
                    print(f"  {j+1}. {det['color'].title()} at {bearing_str}")
                    print(f"     Position: ({det['center_x']}, {det['center_y']})")
                    print(f"     Area: {det['area']:.0f} pixels")
                
                # Test color filtering
                for color in ['green', 'red', 'pink']:
                    color_objects = taxi.get_objects_by_color(color)
                    if color_objects:
                        print(f"  {color.title()} objects: {len(color_objects)}")
                
                # Test angle-based object search
                test_angles = [0, 30, -30]
                for angle in test_angles:
                    closest_obj = taxi.get_closest_object_by_angle(angle, tolerance=15)
                    if closest_obj:
                        bearing_str = f"{closest_obj['bearing']:.1f}°" if closest_obj['bearing'] is not None else "Unknown"
                        print(f"  Closest to {angle}°: {closest_obj['color']} at {bearing_str}")
            else:
                print("No objects detected")
            
            # Show full system status
            if i % 5 == 0:  # Every 5th iteration
                full_status = taxi.get_full_status()
                camera_status = full_status.get('camera', {})
                print(f"Full camera status: {camera_status}")
            
            time.sleep(2)  # Wait 2 seconds between tests
    
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    
    finally:
        print("Stopping camera controller...")
        taxi.stop_camera_controller()
        print("Test completed")

def test_interactive_camera_commands():
    """Test the interactive camera commands"""
    print("\n=== Testing Interactive Camera Commands ===")
    print("Available camera commands:")
    print("  camera          - Show camera status")
    print("  detections      - Show all object detections")
    print("  color <color>   - Show objects of specific color")
    print("  angle_obj <angle> - Show closest object to angle")
    print("  full            - Show complete system status")
    print("  quit            - Exit test")
    
    taxi = TaxiDriver()
    taxi.start_camera_controller()
    time.sleep(3)
    
    try:
        while True:
            command = input("Test> ").strip().lower()
            
            if command in ['quit', 'exit']:
                break
            elif command == 'camera':
                status = taxi.get_camera_status()
                print(f"Camera Status: {status}")
            elif command == 'detections':
                detections = taxi.get_camera_detections()
                if detections:
                    print(f"Found {len(detections)} objects:")
                    for i, det in enumerate(detections):
                        bearing_str = f"{det['bearing']:.1f}°" if det['bearing'] is not None else "Unknown"
                        print(f"  {i+1}. {det['color'].title()} at {bearing_str} (area: {det['area']:.0f})")
                else:
                    print("No objects detected")
            elif command.startswith('color '):
                try:
                    color = command.split()[1]
                    objects = taxi.get_objects_by_color(color)
                    if objects:
                        print(f"Found {len(objects)} {color} objects:")
                        for obj in objects:
                            bearing_str = f"{obj['bearing']:.1f}°" if obj['bearing'] is not None else "Unknown"
                            print(f"  {color.title()} at {bearing_str}")
                    else:
                        print(f"No {color} objects detected")
                except IndexError:
                    print("Usage: color <color_name> (green/red/pink)")
            elif command.startswith('angle_obj '):
                try:
                    angle = float(command.split()[1])
                    obj = taxi.get_closest_object_by_angle(angle)
                    if obj:
                        bearing_str = f"{obj['bearing']:.1f}°" if obj['bearing'] is not None else "Unknown"
                        print(f"Closest object to {angle}°: {obj['color'].title()} at {bearing_str}")
                    else:
                        print(f"No objects found near {angle}°")
                except (IndexError, ValueError):
                    print("Usage: angle_obj <angle> (e.g., angle_obj 0)")
            elif command == 'full':
                status = taxi.get_full_status()
                print(f"Full status: {status}")
            else:
                print("Unknown command. Type 'quit' to exit.")
    
    except KeyboardInterrupt:
        print("\nTest interrupted")
    finally:
        taxi.stop_camera_controller()

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'interactive':
        test_interactive_camera_commands()
    else:
        test_camera_integration()
