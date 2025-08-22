import RPi.GPIO as GPIO  # Imports the standard Raspberry Pi GPIO library
from time import sleep   # Imports sleep (aka wait or pause) into the program

# --- Configuration ---
# Set the GPIO pin for the servo signal
SERVO_PIN = 11
# Set the servo's "straight ahead" or zero position in degrees.
# An input of 0 will move the servo to this angle.
SERVO_ZERO_POINT = 80

# --- Setup ---
GPIO.setmode(GPIO.BOARD)   # Sets the pin numbering system to the physical layout
GPIO.setup(SERVO_PIN, GPIO.OUT) # Sets up the servo pin as an output
p = GPIO.PWM(SERVO_PIN, 50)     # Sets up the pin as a PWM pin with 50Hz frequency
p.start(0)                      # Starts running PWM, but with no signal (0% duty cycle)

def set_steering_angle(relative_angle):
    """
    Moves the servo to a position relative to the SERVO_ZERO_POINT.

    Args:
        relative_angle (int): The desired angle, from -90 (left) to +90 (right).
    """
    # Calculate the final absolute angle for the servo
    target_angle = SERVO_ZERO_POINT + relative_angle

    # --- Safety Check ---
    # Ensure the target angle is within the servo's physical limits (0-180 degrees)
    if not 0 <= target_angle <= 180:
        print(f"Error: Calculated angle ({target_angle}) is outside the servo's 0-180 degree range.")
        # Clamp the angle to the nearest limit to prevent damage
        if target_angle < 0:
            target_angle = 0
        elif target_angle > 180:
            target_angle = 180
        print(f"Moving to the closest valid angle: {target_angle}")

    # Map the absolute angle (0-180) to the required duty cycle (2.5-12.5)
    duty_cycle = 2.5 + (target_angle / 18.0)

    # Change the duty cycle to move the servo
    print(f"Input: {relative_angle}, Zero Point: {SERVO_ZERO_POINT}, Moving to absolute angle: {target_angle}")
    p.ChangeDutyCycle(duty_cycle)

    # Allow time for the servo to reach the position
    sleep(1)

    # Stop sending a signal to the servo to prevent jitter and reduce power consumption
    p.ChangeDutyCycle(0)

try:
    print("Servo steering control initialized.")
    print(f"Zero point is set to {SERVO_ZERO_POINT} degrees.")
    print("Enter a value between -90 (full left) and +90 (full right).")

    # Move to the zero position on startup
    set_steering_angle(0)

    while True:
        # Ask the user for an angle
        try:
            user_input = input("Enter steering angle (-90 to 90, or 'q' to quit): ")
            if user_input.lower() == 'q':
                break

            angle_input = int(user_input)

            if not -90 <= angle_input <= 90:
                print("Invalid input. Please enter an angle between -90 and 90.")
                continue

            set_steering_angle(angle_input)

        except ValueError:
            print("Invalid input. Please enter a number or 'q' to quit.")

except KeyboardInterrupt:
    # This will run if you press CTRL+C
    print("\nExiting program.")

finally:
    # Clean up everything at the end
    print("Stopping PWM and cleaning up GPIO pins.")
    p.stop()                 # Stop the PWM signal
    GPIO.cleanup()           # Reset the GPIO pins to their default state
