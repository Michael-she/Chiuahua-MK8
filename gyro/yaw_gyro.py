import board
import busio
import math
import time
from adafruit_bno08x.i2c import BNO08X_I2C
from adafruit_bno08x import BNO_REPORT_ROTATION_VECTOR

# --- Configuration ---
# Number of initial readings to average for the offset
# A higher number will be more stable but take longer to start.
STABILIZATION_READINGS = 20

# Initialize I2C communication
i2c = busio.I2C(board.SCL, board.SDA)
bno = BNO08X_I2C(i2c)

# Enable the rotation vector report, which provides quaternion data
bno.enable_feature(BNO_REPORT_ROTATION_VECTOR)

def calculate_yaw(quaternion):
    """
    Calculate yaw angle from a quaternion.
    """
    q_i, q_j, q_k, q_real = quaternion

    # Yaw (z-axis rotation)
    siny_cosp = 2 * (q_real * q_k + q_i * q_j)
    cosy_cosp = 1 - 2 * (q_j * q_j + q_k * q_k)
    yaw_rad = math.atan2(siny_cosp, cosy_cosp)

    return math.degrees(yaw_rad)

def get_stable_yaw():
    """Read the sensor multiple times to get a stable initial value."""
    readings = []
    print(f"Acquiring {STABILIZATION_READINGS} readings for stable offset...")
    while len(readings) < STABILIZATION_READINGS:
        try:
            quat = bno.quaternion
            yaw = calculate_yaw(quat)
            readings.append(yaw)
            time.sleep(0.05) # Small delay between readings
        except (RuntimeError, KeyError, OSError) as e:
            # Ignore errors during stabilization and try again
            print(f"Warning: Caught error during stabilization, retrying. {e}")
            time.sleep(0.1)
            continue
    return sum(readings) / len(readings)


# --- Main Program ---
yaw_offset = 0.0

print("Calibrating sensor... Do not move the sensor.")
# Establish a stable yaw offset at the start
yaw_offset = get_stable_yaw()
print(f"Calibration complete. Offset set to: {yaw_offset:.2f} degrees.")
print("\nReading calibrated yaw from BNO08x. Press Ctrl-C to exit.")


while True:
    try:
        # The quaternion data is available from the .quaternion property
        quat = bno.quaternion
        current_yaw = calculate_yaw(quat)

        # Apply the offset
        calibrated_yaw = current_yaw - yaw_offset

        # Normalize the angle to be within -180 to +180 degrees
        if calibrated_yaw > 180:
            calibrated_yaw -= 360
        elif calibrated_yaw < -180:
            calibrated_yaw += 360

        print(f"Calibrated Yaw: {calibrated_yaw:0.2f} degrees")
        time.sleep(0.1)

    except KeyError:
        # This error can be safely ignored
        continue
    except RuntimeError as e:
        # This error may indicate a communication issue
        print(f"Caught a RuntimeError: {e}")
        time.sleep(0.1)
        continue
    except OSError as e:
        # This is an I/O error, likely due to a wiring or power issue
        print(f"Caught OSError: {e}. Check wiring and power.")
        time.sleep(1) # Pause for a second to let the bus settle
        continue