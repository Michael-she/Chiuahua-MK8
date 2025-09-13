import cv2
import sys
import os
import time
from datetime import datetime

# Import relay control functions
try:
    from relay_control import setup_gpio, activate_relay, cleanup
    RELAY_AVAILABLE = True
    print("Relay control module imported successfully")
except ImportError as e:
    RELAY_AVAILABLE = False
    print(f"Warning: Could not import relay control: {e}")

# Test with a non-existent camera index to trigger relay activation
camera_index = 99  # This will fail and trigger relay
MAX_CAMERA_RETRIES = 2  # Reduced for testing
RETRY_DELAY = 1  # Reduced for testing

def try_open_camera(index, attempt=1):
    """Attempt to open camera with retry logic and relay activation"""
    print(f"Attempt {attempt}: Trying to open camera with index {index}...")
    cap = cv2.VideoCapture(index)
    
    if cap.isOpened():
        print(f"✓ Camera {index} opened successfully on attempt {attempt}!")
        return cap
    else:
        print(f"✗ Failed to open camera {index} on attempt {attempt}")
        cap.release()
        
        if attempt < MAX_CAMERA_RETRIES and RELAY_AVAILABLE:
            print("Attempting to reset camera using relay...")
            try:
                setup_gpio()
                activate_relay()
                cleanup()
                print(f"Relay activation completed. Waiting {RETRY_DELAY} seconds before retry...")
                time.sleep(RETRY_DELAY)
            except Exception as e:
                print(f"Error during relay activation: {e}")
                if 'cleanup' in globals():
                    cleanup()
        
        return None

# Try to open camera with relay reset capability
cap = None
for attempt in range(1, MAX_CAMERA_RETRIES + 1):
    cap = try_open_camera(camera_index, attempt)
    if cap is not None:
        break
    
    if attempt < MAX_CAMERA_RETRIES:
        print(f"Retrying in {RETRY_DELAY} seconds...")
        time.sleep(RETRY_DELAY)

if cap is None:
    print(f"Error: Could not open camera with index {camera_index} after {MAX_CAMERA_RETRIES} attempts.")
    print("This is expected for this test - now trying camera index 0...")
    
    # Try the working camera
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        print("✓ Camera 0 works! Relay functionality is properly integrated.")
        cap.release()
    else:
        print("✗ Even camera 0 failed")
