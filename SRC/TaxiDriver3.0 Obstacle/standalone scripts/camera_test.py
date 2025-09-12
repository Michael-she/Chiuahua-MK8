#!/usr/bin/env python3
import cv2

def test_camera_indices():
    """Test multiple camera indices to find working cameras"""
    # Test common indices and the video devices we found
    test_indices = [0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 23, 31]
    
    working_cameras = []
    
    for index in test_indices:
        print(f"Testing camera index {index}...")
        cap = cv2.VideoCapture(index)
        
        if cap.isOpened():
            # Try to read a frame to make sure it's really working
            ret, frame = cap.read()
            if ret and frame is not None:
                height, width = frame.shape[:2]
                print(f"  ✓ Camera {index} works! Resolution: {width}x{height}")
                working_cameras.append(index)
            else:
                print(f"  ✗ Camera {index} opened but couldn't read frame")
        else:
            print(f"  ✗ Camera {index} could not be opened")
        
        cap.release()
    
    return working_cameras

if __name__ == "__main__":
    print("Scanning for available cameras...")
    cameras = test_camera_indices()
    
    if cameras:
        print(f"\nFound working cameras at indices: {cameras}")
        print(f"Try using camera_index = {cameras[0]} in your script")
    else:
        print("\nNo working cameras found!")
        print("This could mean:")
        print("1. No camera is connected")
        print("2. Camera driver issues")
        print("3. Permission issues")
        print("4. Camera is being used by another process")
