#!/usr/bin/env python3
"""
Interactive mode for TaxiDriver - allows manual control via commands
Use this if you want to control the robot interactively instead of visualization
"""

import sys
import time
from taxi_driver import TaxiDriver

def main():
    """Main function for interactive control"""
    taxi = None
    try:
        print("=== Taxi Driver 2.0 - Interactive Mode ===")
        taxi = TaxiDriver()
        
        print("Starting all controllers...")
        taxi.start_all_controllers()
        
        # Give controllers time to initialize
        print("Waiting for controllers to initialize...")
        time.sleep(3)
        
        print("Controllers initialized. Starting interactive control...")
        print("Type 'help' to see available commands")
        print("Type 'viz' to start object visualization")
        print()
        
        # Start interactive control
        taxi.interactive_control()
        
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if taxi:
            taxi.cleanup()

if __name__ == "__main__":
    main()
