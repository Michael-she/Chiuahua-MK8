import board
import busio
import math
import time
from adafruit_bno08x.i2c import BNO08X_I2C
from adafruit_bno08x import BNO_REPORT_ROTATION_VECTOR

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

print("Reading yaw from BNO08x. Press Ctrl-C to exit.")

while True:
    try:
        # The quaternion data is available from the .quaternion property
        quat = bno.quaternion

        # Calculate yaw from the quaternion
        yaw = calculate_yaw(quat)

        print("Yaw: %0.2f degrees" % yaw)

    except KeyError:
        # This error occurs when the library receives a report ID (like 123)
        # that it doesn't recognize. We can safely ignore this and
        # continue to the next loop iteration.
        # print("Caught a KeyError, likely from an unknown report type. Continuing...")
        continue
    except RuntimeError as e:
        # This error can happen if the sensor isn't ready or there's a
        # communication issue.
        print("Caught a RuntimeError:", e)
        # A brief pause can help the sensor recover
        time.sleep(0.1)
        continue