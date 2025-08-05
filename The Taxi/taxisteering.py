
import board
import busio
import math
import time
import threading
import RPi.GPIO as GPIO
from adafruit_bno08x.i2c import BNO08X_I2C
from adafruit_bno08x import BNO_REPORT_ROTATION_VECTOR

# --- 1. CONFIGURATION ---

# Servo Configuration
SERVO_PIN = 11          # GPIO pin for the servo (using BOARD numbering)
SERVO_ZERO_POINT = 90   # The servo's angle (0-180) for "straight ahead"
SERVO_RANGE = 90        # Max deviation from zero point (+/- 90 degrees)

# P-Controller Configuration
KP = 1.0  # Proportional Gain. Tune this value!

# --- 2. SHARED STATE & THREADING SETUP ---
data_lock = threading.Lock()
current_yaw = 0.0
target_yaw = 0.0
exit_event = threading.Event()

# --- 3. GYROSCOPE AND MATH FUNCTIONS ---

def calculate_yaw(quaternion):
    """Calculate yaw angle from a quaternion."""
    q_i, q_j, q_k, q_real = quaternion
    siny_cosp = 2 * (q_real * q_k + q_i * q_j)
    cosy_cosp = 1 - 2 * (q_j * q_j + q_k * q_k)
    yaw_rad = math.atan2(siny_cosp, cosy_cosp)
    return math.degrees(yaw_rad)

def gyro_thread_func():
    """
    Thread function to continuously read the BNO08x sensor and update the global yaw.
    """
    global current_yaw

    print("[Gyro Thread] Initializing BNO08x sensor...")
    try:
        i2c = busio.I2C(board.SCL, board.SDA)
        bno = BNO08X_I2C(i2c)
        bno.enable_feature(BNO_REPORT_ROTATION_VECTOR)
    except Exception as e:
        print(f"[Gyro Thread] ERROR: Failed to initialize BNO08x: {e}")
        exit_event.set() # Signal other threads to exit
        return

    # --- SENSOR SETTLING AND CALIBRATION ---
    print("[Gyro Thread] Allowing sensor to settle...")
    for _ in range(20):
        try:
            _ = bno.quaternion
            time.sleep(0.02)
        except (KeyError, RuntimeError):
            time.sleep(0.1)

    print("[Gyro Thread] Calibrating zero offset...")
    initial_yaw_readings = []
    while len(initial_yaw_readings) < 50 and not exit_event.is_set():
        try:
            quat = bno.quaternion
            initial_yaw = calculate_yaw(quat)
            initial_yaw_readings.append(initial_yaw)
            time.sleep(0.02)
        except (KeyError, RuntimeError):
            time.sleep(0.1)

    if not initial_yaw_readings:
        print("[Gyro Thread] ERROR: Could not get calibration readings. Exiting.")
        exit_event.set()
        return

    yaw_offset = sum(initial_yaw_readings) / len(initial_yaw_readings)
    print(f"[Gyro Thread] Calibration complete. Zero offset: {yaw_offset:.2f} degrees.")
    
    # --- CONTINUOUS ANGLE TRACKING LOOP ---
    continuous_yaw_val = 0.0
    last_raw_yaw = None
    
    while not exit_event.is_set():
        try:
            quat = bno.quaternion
            raw_yaw = calculate_yaw(quat)

            if last_raw_yaw is None:
                last_raw_yaw = raw_yaw
                continuous_yaw_val = raw_yaw - yaw_offset
            else:
                delta_yaw = raw_yaw - last_raw_yaw
                if delta_yaw > 180: delta_yaw -= 360
                elif delta_yaw < -180: delta_yaw += 360
                
                continuous_yaw_val += delta_yaw
                last_raw_yaw = raw_yaw

            with data_lock:
                current_yaw = continuous_yaw_val
            
            time.sleep(0.01)

        except (KeyError, RuntimeError):
            time.sleep(0.1)
            continue

    print("[Gyro Thread] Exiting.")

# --- 4. SERVO CONTROL FUNCTIONS ---

def set_servo_duty_cycle(pwm_controller, duty_cycle):
    """Sets the servo's PWM duty cycle directly."""
    pwm_controller.ChangeDutyCycle(duty_cycle)

# MODIFIED: The function now accepts the pwm object as an argument
def servo_thread_func(servo_pwm):
    """
    Thread function to control the servo based on the error between
    the target and current yaw.
    """
    print("[Servo Thread] Starting control loop.")
    
    while not exit_event.is_set():
        with data_lock:
            local_current_yaw = current_yaw
            local_target_yaw = target_yaw

        error = local_target_yaw - local_current_yaw
        correction_angle = KP * error
        correction_angle = max(-SERVO_RANGE, min(SERVO_RANGE, correction_angle))
        final_servo_angle = SERVO_ZERO_POINT - correction_angle
        final_servo_angle = max(0, min(180, final_servo_angle))
        duty_cycle = 2.5 + (final_servo_angle / 18.0)
        set_servo_duty_cycle(servo_pwm, duty_cycle)
        time.sleep(0.05)

    print("[Servo Thread] Exiting.")


# --- 5. MAIN EXECUTION BLOCK ---
if __name__ == "__main__":
    servo_pwm = None
    try:
        # --- CENTRALIZED HARDWARE SETUP ---
        # Set the GPIO mode ONCE, before anything else.
        # We will use BCM mode to be compatible with the adafruit-blinka library.
        # This requires changing the pin number from physical (11) to BCM (17).
        # Physical pin 11 is GPIO 17.
        print("[Main] Setting GPIO mode to BCM")
        GPIO.setmode(GPIO.BCM)
        SERVO_PIN_BCM = 17 # <--- IMPORTANT: Use BCM pin number
        
        # Setup the servo pin and create the PWM object ONCE.
        print(f"[Main] Setting up GPIO pin {SERVO_PIN_BCM} for servo PWM")
        GPIO.setup(SERVO_PIN_BCM, GPIO.OUT)
        servo_pwm = GPIO.PWM(SERVO_PIN_BCM, 50)
        servo_pwm.start(0)

        # Move servo to initial zero position
        initial_duty_cycle = 2.5 + (SERVO_ZERO_POINT / 18.0)
        set_servo_duty_cycle(servo_pwm, initial_duty_cycle)
        time.sleep(1)
        set_servo_duty_cycle(servo_pwm, 0) # Relax servo

        # --- THREAD CREATION AND START ---
        print("[Main] Creating and starting threads...")
        gyro_reader = threading.Thread(target=gyro_thread_func)
        # Pass the servo_pwm object to the thread
        servo_controller = threading.Thread(target=servo_thread_func, args=(servo_pwm,))
        
        gyro_reader.start()
        servo_controller.start()

        # Wait for gyro calibration to finish
        while gyro_reader.is_alive() and not exit_event.is_set() and current_yaw == 0.0:
            time.sleep(0.5)
        
        print("\n--- System Initialized ---")
        print("Enter a target yaw angle in degrees.")
        print("The servo will correct to match the target.")
        print("Enter 'q' to quit.\n")

        while not exit_event.is_set():
            with data_lock:
                c_yaw, t_yaw = current_yaw, target_yaw
            
            # Use carriage return to print on the same line for a cleaner interface
            print(f"Current Yaw: {c_yaw:7.2f}°  |  Target Yaw: {t_yaw:7.2f}°   ", end='\r')
            
            user_input = input("New Target Angle > ")
            
            if user_input.lower() == 'q':
                print("\nShutdown signal received.")
                exit_event.set()
                break
            
            try:
                new_target = float(user_input)
                with data_lock:
                    target_yaw = new_target
                print(f"New target set to: {new_target:.2f}°")
            except ValueError:
                print("\nInvalid input. Please enter a number or 'q'.")

    except KeyboardInterrupt:
        print("\nCtrl+C detected. Shutting down...")
        exit_event.set()

    finally:
        # --- CENTRALIZED CLEANUP ---
        if gyro_reader.is_alive():
            gyro_reader.join()
        if servo_controller.is_alive():
            servo_controller.join()
        
        print("[Main] Stopping PWM and cleaning up GPIO.")
        if servo_pwm:
            servo_pwm.stop()
        GPIO.cleanup()
        print("All threads have been joined. Program terminated.")
