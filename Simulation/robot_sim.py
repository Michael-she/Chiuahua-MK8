import pygame
import math
import random
import numpy as np

# --- Configuration Constants ---
# Screen
WIDTH, HEIGHT = 800, 800
FPS = 30

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)

# Robot properties
ROBOT_SIZE = 15
MAX_SPEED = 3.0
MAX_STEERING = math.radians(40) # Max steering angle in radians

# Lidar properties
LIDAR_RANGE = 250
LIDAR_RAYS = 90
LIDAR_INACCURACY = 0.02 # 2% noise

# Planner properties
SAFETY_DISTANCE = 30 # How far to stay from walls
WAYPOINT_THRESHOLD = 20 # How close to get to target to consider it "reached"

# --- Wall Definition ---
# Define the course as a list of line segments [(x1, y1), (x2, y2)]
# A square within a square
WALLS = [
    # Outer square
    ((50, 50), (750, 50)),
    ((750, 50), (750, 750)),
    ((750, 750), (50, 750)),
    ((50, 750), (50, 50)),
    # Inner square
    ((250, 250), (550, 250)),
    ((550, 250), (550, 550)),
    ((550, 550), (250, 550)),
    ((250, 550), (250, 250)),
]

class PIDController:
    """A simple PID controller."""
    def __init__(self, Kp, Ki, Kd):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.prev_error = 0
        self.integral = 0

    def update(self, error, dt):
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        self.prev_error = error
        return output

class Robot:
    """Represents the robot and its sensors."""
    def __init__(self, x, y, theta):
        self.x = x
        self.y = y
        self.theta = theta  # Angle in radians
        self.speed = 0
        self.steering_angle = 0
        self.wheelbase = 30 # Distance between front and back wheels

    def update_pose(self, dt):
        """Update robot's position and orientation based on current speed and steering."""
        # Using a simple bicycle model for kinematics
        self.x += self.speed * math.cos(self.theta) * dt
        self.y += self.speed * math.sin(self.theta) * dt
        self.theta += (self.speed / self.wheelbase) * math.tan(self.steering_angle) * dt
        self.theta = self.theta % (2 * math.pi)

    def get_lidar_scan(self, walls):
        """Simulate a LiDAR scan."""
        scan = []
        for i in range(LIDAR_RAYS):
            angle = self.theta - math.radians(LIDAR_RAYS / 2) + math.radians(i)
            ray_end_x = self.x + LIDAR_RANGE * math.cos(angle)
            ray_end_y = self.y + LIDAR_RANGE * math.sin(angle)

            min_dist = LIDAR_RANGE
            for wall in walls:
                p1, p2 = wall
                dist = self._get_ray_wall_intersection(p1, p2, (self.x, self.y), (ray_end_x, ray_end_y))
                if dist and dist < min_dist:
                    min_dist = dist
            
            # Add inaccuracy
            min_dist *= (1 + random.uniform(-LIDAR_INACCURACY, LIDAR_INACCURACY))
            scan.append(min_dist)
        return scan

    def _get_ray_wall_intersection(self, p1, p2, p3, p4):
        """Helper to find intersection between a ray and a wall segment."""
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4
        
        den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if den == 0:
            return None # Parallel
            
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
        u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / den
        
        if 0 < t < 1 and u > 0:
            # Intersection point exists
            ix = x1 + t * (x2 - x1)
            iy = y1 + t * (y2 - y1)
            return math.sqrt((ix - x3)**2 + (iy - y3)**2)
        return None

def draw_robot(screen, robot):
    """Draws the robot on the screen."""
    pygame.draw.circle(screen, RED, (int(robot.x), int(robot.y)), ROBOT_SIZE)
    # Draw a line for direction
    end_x = robot.x + ROBOT_SIZE * math.cos(robot.theta)
    end_y = robot.y + ROBOT_SIZE * math.sin(robot.theta)
    pygame.draw.line(screen, WHITE, (int(robot.x), int(robot.y)), (int(end_x), int(end_y)), 2)
    
def draw_walls(screen, walls):
    """Draws the course walls."""
    for wall in walls:
        pygame.draw.line(screen, WHITE, wall[0], wall[1], 3)

def draw_lidar(screen, robot, scan_data):
    """Draws the LiDAR scan points."""
    for i, dist in enumerate(scan_data):
        if dist < LIDAR_RANGE * 0.99: # Don't draw max-range rays
            angle = robot.theta - math.radians(LIDAR_RAYS / 2) + math.radians(i)
            end_x = robot.x + dist * math.cos(angle)
            end_y = robot.y + dist * math.sin(angle)
            pygame.draw.circle(screen, YELLOW, (int(end_x), int(end_y)), 2)

# --- SENSE: Localization Algorithm ---
def localize_from_scan(scan_data):
    """
    Calculates the robot's orientation and distance to the nearest wall.
    This is a simplified version of line-extraction. It finds the closest points
    and assumes they form a line representing the nearest wall.
    """
    # Convert scan to points in robot's frame
    points = []
    for i, dist in enumerate(scan_data):
        if dist < LIDAR_RANGE * 0.9: # Use only valid points
            angle = -math.radians(LIDAR_RAYS / 2) + math.radians(i)
            x = dist * math.cos(angle)
            y = dist * math.sin(angle)
            points.append((x, y))

    if len(points) < 10:
        return None, None # Not enough data

    # Find the two closest points to the robot (origin in this frame)
    points.sort(key=lambda p: p[0]**2 + p[1]**2)
    closest_points = points[:10] # Use a small cluster of closest points for stability

    # Use numpy's polyfit to do a linear regression on these points
    x_coords = np.array([p[0] for p in closest_points])
    y_coords = np.array([p[1] for p in closest_points])
    
    # Fit a line (y = mx + c), which is x = (1/m)y - c/m in robot frame
    # We swap x and y to handle vertical walls better
    try:
        slope, intercept = np.polyfit(y_coords, x_coords, 1)
    except np.linalg.LinAlgError:
        return None, None # Could not fit a line
        
    # Wall angle relative to the robot
    measured_wall_angle = math.atan(slope)
    
    # Distance from robot (0,0) to the line ax+by+c=0 is |c|/sqrt(a^2+b^2)
    # Line is x - slope*y - intercept = 0
    # a=1, b=-slope, c=-intercept
    dist_to_wall = abs(intercept) / math.sqrt(1 + slope**2)
    
    # The world walls are at 0, 90, 180, 270 degrees. Find the closest one.
    known_angles = [0, math.pi/2, math.pi, 3*math.pi/2]
    # Adjust for robot's current rough theta to guess which wall we are seeing
    # (This part is a simplification. A real system would use a better matching heuristic)
    # For now, we assume the wall is the one closest to horizontal or vertical
    
    return measured_wall_angle, dist_to_wall


def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Lidar Robot Simulation")
    clock = pygame.time.Clock()

    robot = Robot(WIDTH / 2, HEIGHT - 150, -math.pi / 2)
    target_pos = None

    # Steering PID
    # Kp: Proportional - how much to react to the current error
    # Ki: Integral - corrects for steady-state error (lingering offset)
    # Kd: Derivative - dampens overshoot and oscillations
    steering_pid = PIDController(Kp=1.2, Ki=0.05, Kd=0.5)

    running = True
    while running:
        dt = clock.tick(FPS) / 1000.0  # Time delta in seconds

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.MOUSEBUTTONDOWN:
                target_pos = pygame.mouse.get_pos()
                print(f"--- New Target Set: {target_pos} ---")

        # --- SENSE ---
        scan_data = robot.get_lidar_scan(WALLS)
        
        # --- LOCALIZATION DEMONSTRATION ---
        # This is where the robot tries to figure out its orientation
        # based on what it sees. This is a simplified stand-in for a full localization system.
        # In a real system, you'd match multiple lines and use a filter (like Kalman).
        wall_angle_relative, dist_to_wall = localize_from_scan(scan_data)
        estimated_theta = None
        if wall_angle_relative is not None:
             # Assume wall is horizontal (0 rad) or vertical (pi/2 rad)
            if abs(wall_angle_relative) < math.radians(45):
                # We are likely looking at a horizontal wall (top/bottom)
                # If wall angle is 0.1 rad, robot's theta must be -0.1 rad
                estimated_theta = (0 - wall_angle_relative) % (2*math.pi)
            else:
                # We are likely looking at a vertical wall (left/right)
                # If wall angle is 1.6 rad (pi/2 + 0.03), robot's theta must be pi/2 - 1.6 = -0.03
                estimated_theta = (math.pi/2 - wall_angle_relative) % (2*math.pi)
                # This logic is simplified and has ambiguities, but demonstrates the principle.


        # --- PLAN ---
        target_steering = 0
        motor_on = False

        if target_pos:
            dx = target_pos[0] - robot.x
            dy = target_pos[1] - robot.y
            dist_to_target = math.sqrt(dx**2 + dy**2)
            
            if dist_to_target > WAYPOINT_THRESHOLD:
                motor_on = True

                # 1. Go-to-Goal behavior
                angle_to_target = math.atan2(dy, dx)
                angle_error = (angle_to_target - robot.theta + math.pi) % (2 * math.pi) - math.pi
                
                # 2. Obstacle Avoidance behavior (repulsive force)
                avoidance_steer = 0
                if dist_to_wall and dist_to_wall < SAFETY_DISTANCE:
                    # If we are too close, steer away.
                    # The sign of wall_angle_relative tells us if the wall is to the "left" or "right"
                    # of the robot's forward direction. We steer away from it.
                    avoidance_steer = -math.copysign(1.0, wall_angle_relative) * 1.5 # Strong repulsion
                
                # 3. Combine behaviors
                # Give more weight to avoidance when very close
                avoid_weight = 1.0 if (dist_to_wall and dist_to_wall < SAFETY_DISTANCE) else 0.2
                goal_weight = 1.0 - avoid_weight
                
                combined_error = goal_weight * angle_error + avoid_weight * avoidance_steer
                target_steering = combined_error
            else:
                print("--- Target Reached! ---")
                target_pos = None # Stop when reached
        
        # --- ACT ---
        # Update PID and get the final command
        steering_command = steering_pid.update(target_steering, dt)
        
        # Clamp steering to physical limits
        robot.steering_angle = max(-MAX_STEERING, min(MAX_STEERING, steering_command))
        
        # Set speed
        robot.speed = MAX_SPEED if motor_on else 0
        # Simple safety stop: if about to hit something, stop!
        if dist_to_wall and dist_to_wall < ROBOT_SIZE + 5:
            robot.speed = 0

        # Update robot physics
        robot.update_pose(dt)
        
        # --- VISUALIZE ---
        screen.fill(BLACK)
        draw_walls(screen, WALLS)
        draw_lidar(screen, robot, scan_data)
        draw_robot(screen, robot)
        if target_pos:
            pygame.draw.circle(screen, GREEN, target_pos, 10)
            pygame.draw.line(screen, GREEN, (robot.x, robot.y), target_pos, 1)

        pygame.display.flip()

        # --- SERIAL VIEW (Console Output) ---
        print(f"Motor: {'ON ' if motor_on else 'OFF'} | "
              f"Steering Target: {math.degrees(target_steering):+5.1f}° | "
              f"Steering Command: {math.degrees(robot.steering_angle):+5.1f}° | "
              f"Est. Theta: {math.degrees(estimated_theta) if estimated_theta is not None else 'N/A':>6.1f}° | "
              f"Real Theta: {math.degrees(robot.theta):6.1f}°")


    pygame.quit()

if __name__ == "__main__":
    main()