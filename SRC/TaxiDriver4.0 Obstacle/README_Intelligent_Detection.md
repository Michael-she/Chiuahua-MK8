# Intelligent Distance Detection Feature

## Overview
Added intelligent distance detection that automatically chooses between furthest point and closest point detection methods based on nearby object detection.

## New Function: `get_intelligent_distance_reading()`

### Purpose
Intelligently selects the optimal distance reading method for forward (0°), right (90°), and left (-90°) directions based on camera object detection.

### Logic
- **When objects detected within 5°**: Uses `get_furthest_point()` to see beyond detected objects
- **When no objects nearby**: Uses `get_closest_point()` for accurate obstacle detection

### Parameters
- `requested_angle`: Target angle in degrees (0° = forward, positive = clockwise)
- `detection_tolerance`: Angle tolerance for checking nearby objects (default: 5.0°)

### Returns
- `tuple`: (distance, method_used) where method_used is 'furthest' or 'closest'

## Visualization Updates

### Visual Indicators
- **Solid circles**: Furthest point detection (looking beyond objects)
- **Hollow circles**: Closest point detection (obstacle detection)
- **Direction labels**: F/R/L with method indicator (F=Furthest, C=Closest)

### Status Display
- Shows method used for each direction
- Displays distance and detection method
- Real-time updates based on object presence

## Interactive Commands

### New Command: `smart <angle>`
Test the intelligent distance reading at any angle:
```
smart 0     # Forward direction
smart 90    # Right direction  
smart -90   # Left direction
```

### Enhanced Gyroscope Status
Added relative angle and forward offset information to gyro status display.

## Implementation Benefits

1. **Adaptive Detection**: Automatically switches between methods based on context
2. **Object Awareness**: Uses camera detection to inform LiDAR processing
3. **Better Navigation**: Sees beyond objects when needed, detects obstacles when clear
4. **Visual Feedback**: Clear indication of which method is being used
5. **Real-time Operation**: Works seamlessly in the visualization loop

## Usage in Main Program

The intelligent detection is automatically used in:
- Real-time visualization (F/R/L indicators)
- Interactive control commands
- Navigation decision making

## Technical Details

### Object Detection Tolerance
- 5° tolerance for checking nearby objects
- Handles angle wrapping (359° to 1° differences)
- Considers all camera-detected objects (green, red, pink)

### Method Selection
- **Furthest**: When objects block the view, look beyond them
- **Closest**: When path is clear, detect immediate obstacles

### Performance
- Minimal overhead added to existing visualization
- Cached detection results where possible
- Real-time operation at 30 FPS maintained
