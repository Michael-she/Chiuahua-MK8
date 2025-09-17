#!/usr/bin/env python3
"""
Test script for all new TaxiDriver features:
- Servo control on pin 26
- Button control on pin 23  
- Camera testing methods with conflict resolution

Usage: python3 test_all_new_features.py
"""

import time
import sys
from taxi_driver import TaxiDriver

def test_servo_functionality(taxi):
    """Test servo control on pin 26"""
    print("\n=== Testing Servo Control (Pin 26) ===")
    
    # Initialize servo
    if not taxi.initialize_servo_pin_26():
        print("❌ Failed to initialize servo")
        return False
    
    print("✅ Servo initialized successfully")
    
    # Test center position
    print("Testing center position (90°)...")
    taxi.center_servo_pin_26()
    time.sleep(1)
    
    # Test different angles
    test_angles = [0, 45, 90, 135, 180]
    for angle in test_angles:
        print(f"Setting servo to {angle}°...")
        taxi.set_servo_angle_pin_26(angle)
        time.sleep(0.5)
        
        # Verify angle
        current_angle = taxi.get_servo_angle_pin_26()
        print(f"Current angle: {current_angle}°")
    
    print("✅ Servo test completed")
    return True

def test_button_functionality(taxi):
    """Test button control on pin 23"""
    print("\n=== Testing Button Control (Pin 23) ===")
    
    # Initialize button
    if not taxi.initialize_button_pin_23():
        print("❌ Failed to initialize button")
        return False
    
    print("✅ Button initialized successfully")
    
    # Test button state reading
    state = taxi.get_button_state_pin_23()
    debounced_state = taxi.get_button_state_debounced_pin_23()
    
    if state == -1:
        print("❌ Error reading button state")
        return False
    
    state_text = "HIGH (Pressed)" if state == 1 else "LOW (Released)"
    debounced_text = "HIGH (Pressed)" if debounced_state == 1 else "LOW (Released)"
    
    print(f"Current button state: {state_text}")
    print(f"Debounced state: {debounced_text}")
    
    # Get full status
    status = taxi.get_button_status_pin_23()
    if 'error' not in status:
        print(f"Button status: {status['status']}")
        print(f"Initialized: {status['initialized']}")
    
    print("✅ Button test completed")
    return True

def test_camera_functionality(taxi):
    """Test camera testing methods with conflict resolution"""
    print("\n=== Testing Camera Methods (with conflict resolution) ===")
    
    # Start camera controller first
    print("Starting camera controller...")
    taxi.start_camera_controller()
    time.sleep(2)  # Let it initialize
    
    # Test single pixel
    print("Testing single pixel RGB at center (320, 240)...")
    r, g, b, success = taxi.test_camera_pixel_rgb(320, 240)
    
    if success:
        print(f"✅ Center pixel RGB: ({r}, {g}, {b})")
        brightness = (r + g + b) / 3
        print(f"Brightness: {brightness:.1f}")
    else:
        print("❌ Failed to capture center pixel")
        return False
    
    # Test multiple pixels
    print("Testing multiple pixels...")
    results = taxi.test_camera_multiple_pixels()
    
    if results and len(results) > 0:
        print(f"✅ Multiple pixel test completed ({len(results)} pixels tested)")
        successful_tests = sum(1 for r in results if r['success'])
        print(f"Successful captures: {successful_tests}/{len(results)}")
    else:
        print("❌ Failed to complete multiple pixel tests")
        return False
    
    # Test color detection
    print("Testing color detection...")
    color_results = taxi.test_camera_color_detection()
    
    if color_results:
        print("✅ Color detection test completed")
        detected_colors = [color for color, data in color_results.items() if data['detected']]
        if detected_colors:
            print(f"Colors detected: {', '.join(detected_colors)}")
        else:
            print("No target colors detected in current view")
    else:
        print("❌ Failed to complete color detection test")
        return False
    
    # Verify camera controller is still running
    time.sleep(1)
    camera_status = taxi.get_camera_status()
    if camera_status.get('is_active', False):
        print("✅ Camera controller restarted successfully after tests")
    else:
        print("⚠️  Camera controller may not have restarted properly")
    
    print("✅ Camera test completed")
    return True

def main():
    """Main test function"""
    print("=== TaxiDriver New Features Test ===")
    print("Testing servo control, button control, and camera methods...")
    
    taxi = None
    try:
        # Initialize TaxiDriver
        print("Initializing TaxiDriver...")
        taxi = TaxiDriver()
        print("✅ TaxiDriver initialized")
        
        # Run tests
        tests_passed = 0
        total_tests = 3
        
        if test_servo_functionality(taxi):
            tests_passed += 1
        
        if test_button_functionality(taxi):
            tests_passed += 1
            
        if test_camera_functionality(taxi):
            tests_passed += 1
        
        # Results summary
        print(f"\n=== Test Results ===")
        print(f"Tests passed: {tests_passed}/{total_tests}")
        
        if tests_passed == total_tests:
            print("🎉 All tests passed!")
            return_code = 0
        else:
            print("❌ Some tests failed")
            return_code = 1
        
        print("\n=== Interactive Commands Available ===")
        print("You can now use these commands in interactive mode:")
        print("  servo <angle>       - Set servo angle (0-180°)")
        print("  servo_center        - Center servo at 90°")
        print("  button_state        - Check button state")
        print("  test_pixel <x> <y>  - Test RGB at pixel coordinates")
        print("  test_pixels         - Test RGB at multiple points")
        print("  test_colors         - Test color detection")
        
        return return_code
        
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return 1
    finally:
        if taxi:
            print("Cleaning up...")
            taxi.cleanup()

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
