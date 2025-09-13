#!/usr/bin/env python3
"""
Camera RGB Testing Script
Test camera pixel RGB values without starting the full TaxiDriver system
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from taxi_driver import TaxiDriver

def main():
    """Main function to test camera RGB functionality"""
    print("=== Camera RGB Testing ===")
    print("Testing camera without starting other controllers...")
    
    # Create TaxiDriver instance (but don't start controllers)
    taxi = TaxiDriver()
    
    try:
        while True:
            print("\n=== Camera Test Menu ===")
            print("1. Test center pixel RGB (320, 240)")
            print("2. Test custom pixel coordinates")
            print("3. Test multiple pixel locations")
            print("4. Test color detection capabilities")
            print("5. Quick RGB at current center")
            print("q. Quit")
            
            choice = input("\nSelect test (1-5 or q): ").strip().lower()
            
            if choice == 'q':
                break
            elif choice == '1':
                print("\nTesting center pixel...")
                r, g, b, success = taxi.test_camera_center_pixel()
                if success:
                    brightness = (r + g + b) / 3
                    print(f"✓ Center pixel RGB: ({r}, {g}, {b})")
                    print(f"  Brightness: {brightness:.1f}")
                    
                    # Color analysis
                    if brightness < 50:
                        print("  Analysis: Very dark")
                    elif brightness < 100:
                        print("  Analysis: Dark")
                    elif brightness < 150:
                        print("  Analysis: Medium brightness")
                    elif brightness < 200:
                        print("  Analysis: Bright")
                    else:
                        print("  Analysis: Very bright")
                else:
                    print("✗ Failed to capture center pixel")
                    
            elif choice == '2':
                try:
                    x = int(input("Enter X coordinate (0-639): "))
                    y = int(input("Enter Y coordinate (0-479): "))
                    
                    print(f"\nTesting pixel at ({x}, {y})...")
                    r, g, b, success = taxi.test_camera_pixel_rgb(x, y)
                    if success:
                        brightness = (r + g + b) / 3
                        print(f"✓ Pixel RGB: ({r}, {g}, {b})")
                        print(f"  Brightness: {brightness:.1f}")
                        
                        # Dominant color
                        max_val = max(r, g, b)
                        if max_val == r and r > g + 20 and r > b + 20:
                            print("  Dominant color: Red")
                        elif max_val == g and g > r + 20 and g > b + 20:
                            print("  Dominant color: Green")
                        elif max_val == b and b > r + 20 and b > g + 20:
                            print("  Dominant color: Blue")
                        else:
                            print("  Dominant color: Mixed/Gray")
                    else:
                        print("✗ Failed to capture pixel")
                except ValueError:
                    print("Invalid coordinates. Please enter integers.")
                    
            elif choice == '3':
                print("\nTesting multiple pixel locations...")
                results = taxi.test_camera_multiple_pixels()
                
                if results:
                    print("\n=== Multi-Pixel Test Results ===")
                    avg_brightness = sum(r['brightness'] for r in results) / len(results)
                    print(f"Average brightness: {avg_brightness:.1f}")
                    
                    # Find brightest and darkest
                    brightest = max(results, key=lambda x: x['brightness'])
                    darkest = min(results, key=lambda x: x['brightness'])
                    
                    print(f"Brightest pixel: {brightest['coordinates']} - {brightest['brightness']:.1f}")
                    print(f"Darkest pixel: {darkest['coordinates']} - {darkest['brightness']:.1f}")
                else:
                    print("✗ Failed to complete multi-pixel test")
                    
            elif choice == '4':
                print("\nTesting color detection capabilities...")
                results = taxi.test_camera_color_detection()
                
                if results:
                    print("\n=== Color Detection Results ===")
                    for color, data in results.items():
                        status = "✓ DETECTED" if data['detected'] else "✗ Not detected"
                        print(f"{color.title()}: {data['pixel_count']} pixels ({data['percentage']:.2f}%) - {status}")
                    
                    # Overall analysis
                    detected_colors = [color for color, data in results.items() if data['detected']]
                    if detected_colors:
                        print(f"\nColors in view: {', '.join(detected_colors)}")
                    else:
                        print("\nNo target colors detected in current camera view")
                else:
                    print("✗ Failed to complete color detection test")
                    
            elif choice == '5':
                # Quick test without saving file
                print("\nQuick center pixel test...")
                try:
                    import cv2
                    cap = cv2.VideoCapture(0)
                    if cap.isOpened():
                        ret, frame = cap.read()
                        if ret:
                            frame = cv2.flip(frame, -1)
                            bgr_pixel = frame[240, 320]  # Center of 640x480
                            b, g, r = bgr_pixel
                            print(f"Quick RGB: ({r}, {g}, {b})")
                        cap.release()
                    else:
                        print("Cannot open camera")
                except Exception as e:
                    print(f"Quick test failed: {e}")
            else:
                print("Invalid choice. Please select 1-5 or q.")
                
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"Error during testing: {e}")
    finally:
        # Clean up any resources
        taxi.cleanup()
        print("\nCamera testing completed")

if __name__ == "__main__":
    main()
