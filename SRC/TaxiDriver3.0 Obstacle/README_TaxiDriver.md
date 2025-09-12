# Taxi Driver API Documentation

## Overview
The `TaxiDriver` class provides a high-level interface to control a motor using multiprocessing for communication with the motor controller hardware. It uses `multiprocessing.Value` objects for thread-safe communication between the main process and the motor controller process.

## Key Features
- **Motor Speed Control**: Set speed from 0-100%
- **Direction Control**: Forward, reverse, or stop
- **Encoder Feedback**: Real-time encoder count reading
- **Multiprocessing**: Separate process handles GPIO operations
- **Thread-Safe**: Uses multiprocessing.Value for safe data sharing

## Quick Start

```python
from taxi_driver import TaxiDriver
import time

# Initialize taxi driver
taxi = TaxiDriver()
taxi.start_motor_controller()
time.sleep(2)  # Allow initialization

# Basic motor control
taxi.move_forward(50)        # Move forward at 50% speed
time.sleep(2)
taxi.move_reverse(30)        # Move reverse at 30% speed
time.sleep(2)
taxi.stop_motor()            # Stop motor

# Get encoder reading
encoder_count = taxi.get_encoder_reading()
print(f"Encoder count: {encoder_count}")

# Cleanup when done
taxi.cleanup()
```

## API Reference

### Initialization

#### `TaxiDriver()`
Creates a new TaxiDriver instance with multiprocessing communication setup.

```python
taxi = TaxiDriver()
```

#### `start_motor_controller()`
Starts the motor controller process. Must be called before motor operations.

```python
taxi.start_motor_controller()
```

#### `stop_motor_controller()`
Stops the motor controller process. Called automatically by cleanup().

```python
taxi.stop_motor_controller()
```

### Motor Control

#### `set_motor_speed(speed_percent)`
Set motor speed as a percentage.

**Parameters:**
- `speed_percent` (float): Speed from 0-100

```python
taxi.set_motor_speed(75)  # Set to 75% speed
```

#### `set_motor_direction(direction)`
Set motor direction.

**Parameters:**
- `direction` (int): 1=forward, -1=reverse, 0=stop

```python
taxi.set_motor_direction(1)   # Forward
taxi.set_motor_direction(-1)  # Reverse
taxi.set_motor_direction(0)   # Stop
```

#### `move_forward(speed_percent)`
Move forward at specified speed.

**Parameters:**
- `speed_percent` (float): Speed from 0-100

```python
taxi.move_forward(60)  # Move forward at 60% speed
```

#### `move_reverse(speed_percent)`
Move reverse at specified speed.

**Parameters:**
- `speed_percent` (float): Speed from 0-100

```python
taxi.move_reverse(40)  # Move reverse at 40% speed
```

#### `stop_motor()`
Stop the motor immediately.

```python
taxi.stop_motor()
```

### Sensor Reading

#### `get_encoder_reading()`
Get current encoder count.

**Returns:**
- `int`: Current encoder count

```python
count = taxi.get_encoder_reading()
print(f"Encoder: {count} counts")
```

### Status Information

#### `get_motor_status()`
Get comprehensive motor status.

**Returns:**
- `dict`: Status dictionary with keys:
  - `speed`: Current speed percentage
  - `direction`: Direction value (-1, 0, 1)
  - `direction_text`: Human-readable direction
  - `encoder_count`: Current encoder count
  - `status`: Combined status string

```python
status = taxi.get_motor_status()
print(f"Motor: {status['status']}")
print(f"Encoder: {status['encoder_count']}")
```

### Utility Functions

#### `run_test_sequence()`
Run a predefined test sequence demonstrating motor control.

```python
taxi.run_test_sequence()
```

#### `interactive_control()`
Start interactive command-line control interface.

```python
taxi.interactive_control()
```

#### `cleanup()`
Clean up resources and stop motor controller process. Should always be called when done.

```python
taxi.cleanup()
```

## Example Usage Patterns

### Basic Control Loop
```python
taxi = TaxiDriver()
taxi.start_motor_controller()
time.sleep(2)

try:
    # Control loop
    for speed in [25, 50, 75, 100]:
        taxi.move_forward(speed)
        time.sleep(1)
        encoder = taxi.get_encoder_reading()
        print(f"Speed: {speed}%, Encoder: {encoder}")
    
    taxi.stop_motor()
    
finally:
    taxi.cleanup()
```

### Encoder-Based Movement
```python
taxi = TaxiDriver()
taxi.start_motor_controller()
time.sleep(2)

try:
    # Move until encoder reaches certain count
    target_encoder = taxi.get_encoder_reading() + 1000
    taxi.move_forward(50)
    
    while taxi.get_encoder_reading() < target_encoder:
        time.sleep(0.1)
    
    taxi.stop_motor()
    print(f"Reached target encoder: {taxi.get_encoder_reading()}")
    
finally:
    taxi.cleanup()
```

## Hardware Configuration

The motor controller is configured for:
- **Motor Control Pins**: GPIO 20, 21 (PWM control)
- **Encoder Pins**: GPIO 19, 13 (quadrature encoder)
- **PWM Frequency**: 2000 Hz

## Error Handling

Always use try/finally blocks to ensure cleanup:

```python
taxi = TaxiDriver()
try:
    taxi.start_motor_controller()
    # Your control code here
finally:
    taxi.cleanup()
```

## Multiprocessing Details

The `TaxiDriver` uses `multiprocessing.Value` objects for communication:
- `motor_speed`: Double precision float (0-100)
- `motor_direction`: Integer (-1, 0, 1)
- `encoder_value`: Integer (encoder count)
- `shutdown_flag`: Integer (process control)

This design ensures thread-safe communication and isolates GPIO operations in a separate process.
