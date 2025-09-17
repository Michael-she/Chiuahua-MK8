#!/usr/bin/env python3
"""
Test script for button connected to GPIO pin 23
Button applies 5V when pressed (HIGH = pressed, LOW = released)
Demonstrates basic button reading functionality
"""

import time
import sys
import os
from taxi_driver import TaxiDriver

def test_button_basic():
    """Test basic button functionality"""
    print("=== Button Pin 23 Basic Test ===")
    
    taxi = None
    try:
        # Initialize TaxiDriver
        taxi = TaxiDriver()
        
        # Initialize button
        print("Initializing button on pin 23...")
        if not taxi.initialize_button_pin_23():
            print("Failed to initialize button. Make sure pigpio daemon is running.")
            return False
        
        print("Button initialized successfully!")
        print()
        
        # Test basic button reading
        print("=== Basic Button State Test ===")
        for i in range(10):
            state = taxi.get_button_state_pin_23()
            debounced_state = taxi.get_button_state_debounced_pin_23()
            is_pressed = taxi.is_button_pressed_pin_23()
            is_pressed_debounced = taxi.is_button_pressed_debounced_pin_23()
            
            if state == -1:
                print(f"Test {i+1}: Error reading button state")
            else:
                print(f"Test {i+1}: Raw={state} ({'Pressed' if state == 0 else 'Released'}), "
                      f"Debounced={debounced_state} ({'Pressed' if debounced_state == 0 else 'Released'}), "
                      f"IsPressed={is_pressed}, IsPressedDebounced={is_pressed_debounced}")
            
            time.sleep(0.5)
        
        print()
        print("=== Button Status Test ===")
        status = taxi.get_button_status_pin_23()
        if 'error' in status:
            print(f"Error getting status: {status['error']}")
        else:
            print(f"Button Status: {status['status']}")
            print(f"Raw state: {status['raw_state']}")
            print(f"Debounced state: {status['debounced_state']}")
            print(f"Is pressed: {status['is_pressed']}")
            print(f"Is pressed (debounced): {status['is_pressed_debounced']}")
            print(f"Initialized: {status['initialized']}")
            print(f"Last state: {status['last_state']}")
            print(f"Last time: {time.ctime(status['last_time'])}")
        
        return True
        
    except Exception as e:
        print(f"Error in button test: {e}")
        return False
    finally:
        if taxi:
            taxi.cleanup_button_pin_23()
            print("Button cleanup completed")

def test_button_wait():
    """Test button wait functionality"""
    print("\n=== Button Wait Test ===")
    
    taxi = None
    try:
        # Initialize TaxiDriver
        taxi = TaxiDriver()
        
        # Initialize button
        print("Initializing button on pin 23...")
        if not taxi.initialize_button_pin_23():
            print("Failed to initialize button")
            return False
        
        print("Button initialized successfully!")
        print()
        
        # Test wait for button press
        print("Testing wait for button press (10 second timeout)...")
        print("Press the button on pin 23!")
        
        if taxi.wait_for_button_press_pin_23(timeout=10):
            print("✓ Button press detected!")
            
            # Wait for release
            print("Now release the button...")
            if taxi.wait_for_button_release_pin_23(timeout=5):
                print("✓ Button release detected!")
            else:
                print("✗ Button release timeout")
        else:
            print("✗ No button press detected within timeout")
        
        return True
        
    except Exception as e:
        print(f"Error in button wait test: {e}")
        return False
    finally:
        if taxi:
            taxi.cleanup_button_pin_23()
            print("Button cleanup completed")

def test_button_interactive():
    """Interactive button monitoring"""
    print("\n=== Interactive Button Monitor ===")
    print("Press Ctrl+C to exit")
    
    taxi = None
    try:
        # Initialize TaxiDriver
        taxi = TaxiDriver()
        
        # Initialize button
        print("Initializing button on pin 23...")
        if not taxi.initialize_button_pin_23():
            print("Failed to initialize button")
            return False
        
        print("Button initialized successfully!")
        print("Monitoring button state (press Ctrl+C to exit)...")
        print()
        
        last_state = None
        
        while True:
            current_state = taxi.is_button_pressed_debounced_pin_23()
            
            if current_state != last_state:
                timestamp = time.strftime("%H:%M:%S")
                if current_state:
                    print(f"[{timestamp}] Button PRESSED")
                else:
                    print(f"[{timestamp}] Button RELEASED")
                last_state = current_state
            
            time.sleep(0.05)  # 50ms polling
        
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user")
        return True
    except Exception as e:
        print(f"Error in interactive monitor: {e}")
        return False
    finally:
        if taxi:
            taxi.cleanup_button_pin_23()
            print("Button cleanup completed")

def main():
    """Main test function"""
    print("Button Pin 23 Test Script")
    print("========================")
    print()
    
    if len(sys.argv) > 1:
        test_type = sys.argv[1].lower()
    else:
        print("Available tests:")
        print("  basic      - Basic button state reading")
        print("  wait       - Button press/release waiting")
        print("  monitor    - Interactive button monitoring")
        print()
        test_type = input("Select test type (basic/wait/monitor): ").strip().lower()
    
    if test_type == 'basic':
        success = test_button_basic()
    elif test_type == 'wait':
        success = test_button_wait()
    elif test_type == 'monitor':
        success = test_button_interactive()
    else:
        print(f"Unknown test type: {test_type}")
        print("Available options: basic, wait, monitor")
        return
    
    if success:
        print("\n✓ Test completed successfully")
    else:
        print("\n✗ Test failed")

if __name__ == "__main__":
    main()
