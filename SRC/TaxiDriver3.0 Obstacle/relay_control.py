#!/usr/bin/env python3
"""
Raspberry Pi GPIO Relay Control
Controls a relay connected to a GPIO pin by switching it on for 0.5 seconds, then off.
"""

import RPi.GPIO as GPIO
import time

# Configuration
RELAY_PIN = 17  # GPIO pin number (BCM numbering) - change this to your actual pin
PULSE_DURATION = 0.2  # Duration in seconds to keep relay on

def setup_gpio():
    """Initialize GPIO settings"""
    # Set the GPIO mode to BCM (Broadcom pin numbering)
    GPIO.setmode(GPIO.BCM)
    
    # Disable GPIO warnings
    GPIO.setwarnings(False)
    
    # Set up the relay pin as output
    GPIO.setup(RELAY_PIN, GPIO.OUT)
    
    # Ensure relay starts in OFF state
    GPIO.output(RELAY_PIN, GPIO.LOW)
    print(f"GPIO pin {RELAY_PIN} initialized as output")

def activate_relay():
    """Activate relay for specified duration"""
    try:
        print(f"Turning relay ON (GPIO pin {RELAY_PIN})")
        GPIO.output(RELAY_PIN, GPIO.HIGH)
        
        print(f"Waiting {PULSE_DURATION} seconds...")
        time.sleep(PULSE_DURATION)
        
        print("Turning relay OFF")
        GPIO.output(RELAY_PIN, GPIO.LOW)
        
        print("Relay control completed successfully")
        
    except Exception as e:
        print(f"Error controlling relay: {e}")
        # Ensure relay is turned off in case of error
        GPIO.output(RELAY_PIN, GPIO.LOW)

def cleanup():
    """Clean up GPIO resources"""
    GPIO.cleanup()
    print("GPIO cleanup completed")

def main():
    """Main function"""
    try:
        setup_gpio()
        activate_relay()
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        cleanup()

if __name__ == "__main__":
    main()
