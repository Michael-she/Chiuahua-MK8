import pygame
import numpy as np
import random
import math
import heapq # Priority queue for A*

# --- Configuration ---
# Screen
SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 700
# Colors
COLOR_BACKGROUND = (10, 10, 10)
COLOR_WALL = (200, 200, 200)
COLOR_ROBOT = (255, 0, 0)
COLOR_ESTIMATED_WALL = (100, 255, 100)
COLOR_INLIER_POINTS = (0, 100, 255)
COLOR_TEXT = (255, 255, 255)
COLOR_PERPENDICULAR = (0, 170, 255)
COLOR_NON_PERPENDICULAR = (200, 160, 255)
COLOR_OBSTACLE_DETECTED = (255, 255, 0)
COLOR_OBSTACLE_POTENTIAL = (255, 255, 0, 100)
COLOR_OBSTACLE_RED = (255, 100, 100)
COLOR_OBSTACLE_GREEN = (100, 200, 100)
COLOR_DETECTION_ARC = (255, 255, 0, 30)
COLOR_WAYPOINT = (128, 0, 128) # Purple
COLOR_PLANNED_PATH = (150, 150, 255, 150) # Light transparent blue
COLOR_SHARP_TURN = (255, 100, 0) # Orange
COLOR_DYNAMIC_WALL = (255, 165, 0) # Bright Orange for new walls
COLOR_SEGMENT_BORDER = (255, 0, 255) # Magenta for segment borders
COLOR_UNASSOCIATED_POINTS_DEBUG = (255, 128, 0) # Orange for debugging
COLOR_FINISH_GATE = (0, 255, 255) # Cyan for finish gate

ORANGE = (255, 0, 255)
BLACK =  (10, 10, 10)
# Robot properties
ROBOT_SIZE = 15
ROBOT_WHEELBASE = 10.0
ROBOT_MAX_STEERING_ANGLE = 35.0
ROBOT_STEERING_RATE = 10.0
ROBOT_STEERING_RETURN_RATE = 0.9
ROBOT_BASE_SPEED = 1.0
# LiDAR properties
LIDAR_RANGE = 5000
LIDAR_RAYS = 360
LIDAR_INACCURACY = 6.0
# RANSAC & Clustering Parameters
RANSAC_ITERATIONS = 20
# --- TUNING FIX: RANSAC threshold is tightened to better separate obstacles from walls ---
RANSAC_THRESHOLD_DISTANCE = 8.0
RANSAC_MIN_INLIERS = 10
MAX_POINT_DISTANCE = 60.0
PERPENDICULAR_ANGLE_THRESHOLD = 5.0
MIN_WALL_LENGTH = 100.0
# Obstacle Detection Parameters
OBSTACLE_PROXIMITY = 10.0
OBSTACLE_MIN_POINTS = 3
OBSTACLE_MAX_SIZE = 20.0
OBSTACLE_DETECTION_ARC_DEGREES = 120.0
UI_ARC_RADIUS = 150
# Obstacle Persistence Parameters
OBSTACLE_CONFIDENCE_INCREMENT = 1.0
OBSTACLE_CONFIDENCE_DECAY = 0.95
OBSTACLE_CONFIRMATION_THRESHOLD = 5.0
OBSTACLE_MATCHING_DISTANCE = 20.0
# Heatmap Configuration
HEATMAP_CELL_SIZE = 8
HEATMAP_DECAY_RATE = 0.98
HEATMAP_MAX_HITS_ADJUST_RATE = 0.995
# Pathfinding and Navigation Parameters
PATHFINDING_GRID_SIZE = 20
WAYPOINT_DISTANCE = 40.0
WAYPOINT_ARRIVAL_THRESHOLD = 15.0
MAX_PATH_TURN_ANGLE = 180.0
REPLAN_STUCK_THRESHOLD = 10
REVERSE_DURATION_MS = 1000
# Course & Obstacle Definition
COURSE_TOP_LEFT = 50
COURSE_WIDTH = 600
OBSTACLE_SIZE = (10, 10)
INNER_TL = (200 + COURSE_TOP_LEFT, 200 + COURSE_TOP_LEFT)
INNER_TR = (400 + COURSE_TOP_LEFT, 200 + COURSE_TOP_LEFT)
INNER_BL = (200 + COURSE_TOP_LEFT, 400 + COURSE_TOP_LEFT)
INNER_BR = (400 + COURSE_TOP_LEFT, 400 + COURSE_TOP_LEFT)

# Direction of Travel
DIRECTION_OF_TRAVEL = "clockwise"  # Can be "clockwise" or "anticlockwise"

# --- Finish Gate Configuration ---
FINISH_GATE_WIDTH = 60
FINISH_GATE_SEGMENTS = [
    # Segment 1 (Top): gate at bottom of top segment
    ((INNER_TL[0] + INNER_TR[0]) // 2 - FINISH_GATE_WIDTH // 2, INNER_TL[1] - 10, 
     (INNER_TL[0] + INNER_TR[0]) // 2 + FINISH_GATE_WIDTH // 2, INNER_TL[1] - 10),
    # Segment 2 (Right): gate at left of right segment  
    (INNER_TR[0] - 10, (INNER_TR[1] + INNER_BR[1]) // 2 - FINISH_GATE_WIDTH // 2,
     INNER_TR[0] - 10, (INNER_TR[1] + INNER_BR[1]) // 2 + FINISH_GATE_WIDTH // 2),
    # Segment 3 (Bottom): gate at top of bottom segment
    ((INNER_BL[0] + INNER_BR[0]) // 2 - FINISH_GATE_WIDTH // 2, INNER_BR[1] + 10,
     (INNER_BL[0] + INNER_BR[0]) // 2 + FINISH_GATE_WIDTH // 2, INNER_BR[1] + 10),
    # Segment 4 (Left): gate at right of left segment
    (INNER_TL[0] + 10, (INNER_TL[1] + INNER_BL[1]) // 2 - FINISH_GATE_WIDTH // 2,
     INNER_TL[0] + 10, (INNER_TL[1] + INNER_BL[1]) // 2 + FINISH_GATE_WIDTH // 2)
]

# --- Dynamic Wall Rule Configuration ---
# Controls which color obstacle creates walls in which direction
DYNAMIC_WALL_RULE = "swapped"  # Can be "standard" or "swapped"

# --- Ray Persistence Configuration ---
RAY_PERSISTENCE_DURATION = 1000  # milliseconds

ray_cache = {}


SHOW_GRID = True
COURSE_WALLS = [
    (0+COURSE_TOP_LEFT, 0+COURSE_TOP_LEFT, COURSE_WIDTH+COURSE_TOP_LEFT, 0+COURSE_TOP_LEFT),
    (COURSE_WIDTH+COURSE_TOP_LEFT, 0+COURSE_TOP_LEFT, COURSE_WIDTH+COURSE_TOP_LEFT, COURSE_WIDTH+COURSE_TOP_LEFT),
    (COURSE_WIDTH+COURSE_TOP_LEFT, COURSE_WIDTH+COURSE_TOP_LEFT, 0+COURSE_TOP_LEFT, COURSE_WIDTH+COURSE_TOP_LEFT),
    (0+COURSE_TOP_LEFT, COURSE_WIDTH+COURSE_TOP_LEFT, 0+COURSE_TOP_LEFT, 0+COURSE_TOP_LEFT),
    (INNER_TL[0], INNER_TL[1], INNER_TR[0], INNER_TR[1]),
    (INNER_TR[0], INNER_TR[1], INNER_BR[0], INNER_BR[1]),
    (INNER_BR[0], INNER_BR[1], INNER_BL[0], INNER_BL[1]),
    (INNER_BL[0], INNER_BL[1], INNER_TL[0], INNER_TL[1]),
]

# --- Helper Functions ---
def find_closest_point_on_segment(p, a, b):
    p_np, a_np, b_np = np.array(p), np.array(a), np.array(b)
    ab, ap = b_np - a_np, p_np - a_np
    ab_len_sq = np.dot(ab, ab)
    if ab_len_sq == 0: return a
    t = np.clip(np.dot(ap, ab) / ab_len_sq, 0, 1)
    return tuple(a_np + t * ab)

def check_obstacle_wall_collision(obstacle, walls):
    center, radius_sq = obstacle['center'], obstacle['radius']**2
    for wall in walls:
        
        if len(wall) != 4:
            wall_start = (wall[0][0], wall[0][1])
            wall_end = (wall[1][0], wall[1][1])
        
            closest_point = find_closest_point_on_segment(center, wall_start, wall_end)
            dist_sq = (center[0] - closest_point[0])**2 + (center[1] - closest_point[1])**2
            if dist_sq < radius_sq:
                return True
    return False

def generate_obstacles():
    obstacle_slots = {k: [(v[0], v[1]) for v in vs] for k, vs in {
        "top": [(200+COURSE_TOP_LEFT, 80+COURSE_TOP_LEFT), (300+COURSE_TOP_LEFT, 80+COURSE_TOP_LEFT), (400+COURSE_TOP_LEFT, 80+COURSE_TOP_LEFT), (200+COURSE_TOP_LEFT, 120+COURSE_TOP_LEFT), (300+COURSE_TOP_LEFT, 120+COURSE_TOP_LEFT), (400+COURSE_TOP_LEFT, 120+COURSE_TOP_LEFT)],
        "bottom": [(200+COURSE_TOP_LEFT, 480+COURSE_TOP_LEFT), (300+COURSE_TOP_LEFT, 480+COURSE_TOP_LEFT), (400+COURSE_TOP_LEFT, 480+COURSE_TOP_LEFT), (200+COURSE_TOP_LEFT, 520+COURSE_TOP_LEFT), (300+COURSE_TOP_LEFT, 520+COURSE_TOP_LEFT), (400+COURSE_TOP_LEFT, 520+COURSE_TOP_LEFT)],
        "left": [(80+COURSE_TOP_LEFT, 200+COURSE_TOP_LEFT), (80+COURSE_TOP_LEFT, 300+COURSE_TOP_LEFT), (80+COURSE_TOP_LEFT, 400+COURSE_TOP_LEFT), (120+COURSE_TOP_LEFT, 200+COURSE_TOP_LEFT), (120+COURSE_TOP_LEFT, 300+COURSE_TOP_LEFT), (120+COURSE_TOP_LEFT, 400+COURSE_TOP_LEFT)],
        "right": [(480+COURSE_TOP_LEFT, 200+COURSE_TOP_LEFT), (480+COURSE_TOP_LEFT, 300+COURSE_TOP_LEFT), (480+COURSE_TOP_LEFT, 400+COURSE_TOP_LEFT), (520+COURSE_TOP_LEFT, 200+COURSE_TOP_LEFT), (520+COURSE_TOP_LEFT, 300+COURSE_TOP_LEFT), (520+COURSE_TOP_LEFT, 400+COURSE_TOP_LEFT)],
    }.items()}
    generated_obstacles, obstacle_walls = [], []
    for _, slots in obstacle_slots.items():
        num_to_spawn = random.choice([1, 2 ,2])
        chosen_slots = []
        if num_to_spawn == 1: chosen_slots.append(random.choice(slots))
        elif num_to_spawn == 2: pair = random.choice([(0, 2), (3, 5), (0, 5), (2, 3)]); chosen_slots.extend([slots[pair[0]], slots[pair[1]]])
        for slot_center in chosen_slots:
            rect = pygame.Rect((0, 0), OBSTACLE_SIZE); rect.center = slot_center
            color = random.choice([COLOR_OBSTACLE_RED, COLOR_OBSTACLE_GREEN])
            generated_obstacles.append({'rect': rect, 'color': color})
            tl, tr, bl, br = rect.topleft, rect.topright, rect.bottomleft, rect.bottomright
            obstacle_walls.extend([(tl[0], tl[1], tr[0], tr[1]), (tr[0], tr[1], br[0], br[1]), (br[0], br[1], bl[0], bl[1]), (bl[0], bl[1], tl[0], tl[1])])
    return generated_obstacles, obstacle_walls
def distance_from_point_to_line(p, l1, l2):
    x0,y0=p; x1,y1=l1; x2,y2=l2
    num = abs((y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1)
    den = math.sqrt((y2-y1)**2 + (x2-x1)**2)
    return num/den if den != 0 else 0
def line_intersection(p1, p2, p3, p4):
    x1,y1=p1; x2,y2=p2; x3,y3=p3; x4,y4=p4
    den = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
    if den == 0: return None, float('inf')
    t = ((x1-x3)*(y3-y4) - (y1-y3)*(x3-x4))/den
    u = -((x1-x2)*(y1-y3) - (y1-y2)*(x1-x3))/den
    if 0<t<1 and 0<u<1:
        px, py = x1+t*(x2-x1), y1+t*(y2-y1)
        return (px, py), math.hypot(px-x1, py-y1)
    return None, float('inf')

def cast_ray_and_find_hit(start_pos: tuple, target_pos: tuple, walls: list, max_range: float = LIDAR_RANGE):
    """
    Casts a ray from a start position towards a target and finds the first wall it hits.

    Args:
        start_pos (tuple): The (x, y) coordinate where the ray begins.
        target_pos (tuple): An (x, y) coordinate that defines the direction of the ray.
        walls (list): A list of wall segments to check for intersection.
        max_range (float): The maximum distance the ray can travel.

    Returns:
        tuple: The (x, y) coordinate of the closest intersection point, or None if no wall is hit.
    """
    start_vec = pygame.math.Vector2(start_pos)
    direction_vec = pygame.math.Vector2(target_pos)

    # Calculate the direction vector and a far-off end point for the ray
    try:
        unit_vector = (direction_vec - start_vec).normalize()
    except ValueError:
        return None # Start and target are the same point

    ray_end_pos = start_vec + unit_vector * max_range

    closest_hit_point = None
    min_dist = float('inf')

    # Check for intersection with each wall
    for wall in walls:
        p1 = (wall[0], wall[1])
        p2 = (wall[2], wall[3])
        intersection_point, dist = line_intersection(start_pos, ray_end_pos, p1, p2)

        if intersection_point and dist < min_dist:
            min_dist = dist
            closest_hit_point = intersection_point

    return closest_hit_point

def get_obstacle_at_coord(x, y, obstacles_list):
    """
    Checks if a point (x, y) is inside any of the generated physical obstacles.

    This function checks against the ground-truth list of obstacles, not
    the robot's detected obstacles.

    Args:
        x (int): The x-coordinate to check.
        y (int): The y-coordinate to check.
        obstacles_list (list): The list of obstacle dictionaries, 
                             where each dict has a 'rect' and 'color' key.

    Returns:
        tuple: A tuple of (bool, color). 
               - If an obstacle is found, returns (True, the_obstacle_color).
               - If no obstacle is found, returns (False, None).
    """
    point_to_check = (x, y)
    # Iterate through all the obstacles in the provided list
    for obstacle in obstacles_list:
        # Use the powerful and efficient collidepoint() method of pygame.Rect
        if obstacle['rect'].collidepoint(point_to_check):
            # If the point is inside the rect, we found a hit.
            # Return True and the color of that obstacle.
            return (True, obstacle['color'])
    
    # If the loop finishes without finding any obstacle at the point,
    # return False and None for the color.
    return (False, None)

def check_finish_gate_collision(robot_pos, robot_prev_pos, segment):
    """
    Check if robot has passed through the finish gate for the current segment.
    
    Args:
        robot_pos: Current robot position (x, y)
        robot_prev_pos: Previous robot position (x, y)
        segment: Current segment (1=Top, 2=Right, 3=Bottom, 4=Left)
    
    Returns:
        bool: True if robot passed through finish gate
    """
    if segment < 1 or segment > 4:
        return False
    
    gate = FINISH_GATE_SEGMENTS[segment - 1]  # Convert to 0-based index
    
    # Check if robot crossed the gate line
    if segment == 1 or segment == 3:  # Horizontal gates
        gate_y = gate[1]
        # Check if robot crossed the horizontal line
        if ((robot_prev_pos[1] <= gate_y <= robot_pos[1]) or 
            (robot_pos[1] <= gate_y <= robot_prev_pos[1])):
            # Check if crossing was within gate bounds
            gate_x_min, gate_x_max = min(gate[0], gate[2]), max(gate[0], gate[2])
            robot_x = robot_pos[0]
            if gate_x_min <= robot_x <= gate_x_max:
                return True
    else:  # Vertical gates
        gate_x = gate[0]
        # Check if robot crossed the vertical line
        if ((robot_prev_pos[0] <= gate_x <= robot_pos[0]) or 
            (robot_pos[0] <= gate_x <= robot_prev_pos[0])):
            # Check if crossing was within gate bounds
            gate_y_min, gate_y_max = min(gate[1], gate[3]), max(gate[1], gate[3])
            robot_y = robot_pos[1]
            if gate_y_min <= robot_y <= gate_y_max:
                return True
    
    return False

def get_opposite_segment(segment):
    """
    Get the opposite segment for auto-target mode.
    
    Args:
        segment: Current segment (1=Top, 2=Right, 3=Bottom, 4=Left)
    
    Returns:
        int: Opposite segment number
    """
    # Opposite segments: 1<->3 (Top<->Bottom), 2<->4 (Right<->Left)
    if segment == 1:
        return 3  # Top -> Bottom
    elif segment == 2:
        return 4  # Right -> Left
    elif segment == 3:
        return 1  # Bottom -> Top
    elif segment == 4:
        return 2  # Left -> Right
    else:
        return 1  # Default to top

def get_target_position_for_segment(segment):
    """
    Get the target position for a specific segment.
    
    Args:
        segment: Target segment (1=Top, 2=Right, 3=Bottom, 4=Left)
    
    Returns:
        tuple: (target_pos, target_angle) for the segment
    """
    # Define target positions for each segment (center of outer walls)
    segment_targets = {
        1: ((COURSE_TOP_LEFT + COURSE_WIDTH // 2, COURSE_TOP_LEFT + 50), 0),     # Top segment, facing right
        2: ((COURSE_TOP_LEFT + COURSE_WIDTH - 50, COURSE_TOP_LEFT + COURSE_WIDTH // 2), 90),   # Right segment, facing down  
        3: ((COURSE_TOP_LEFT + COURSE_WIDTH // 2, COURSE_TOP_LEFT + COURSE_WIDTH - 50), 180),  # Bottom segment, facing left
        4: ((COURSE_TOP_LEFT + 50, COURSE_TOP_LEFT + COURSE_WIDTH // 2), 270)    # Left segment, facing up
    }
    
    return segment_targets.get(segment, segment_targets[1])

def get_next_target_position(current_segment, direction_of_travel):
    """
    Get the next target position based on current segment and direction of travel.
    
    Args:
        current_segment: Current segment (1=Top, 2=Right, 3=Bottom, 4=Left)
        direction_of_travel: "clockwise" or "anticlockwise"
    
    Returns:
        tuple: (target_pos, target_angle) for the next segment
    """
    if direction_of_travel == "clockwise":
        next_segment = (current_segment % 4) + 1
    else:  # anticlockwise
        next_segment = ((current_segment - 2) % 4) + 1
    
    # Define target positions for each segment (center of outer walls)
    segment_targets = {
        1: ((COURSE_TOP_LEFT + COURSE_WIDTH // 2, COURSE_TOP_LEFT + 50), 0),     # Top segment, facing right
        2: ((COURSE_TOP_LEFT + COURSE_WIDTH - 50, COURSE_TOP_LEFT + COURSE_WIDTH // 2), 90),   # Right segment, facing down  
        3: ((COURSE_TOP_LEFT + COURSE_WIDTH // 2, COURSE_TOP_LEFT + COURSE_WIDTH - 50), 180),  # Bottom segment, facing left
        4: ((COURSE_TOP_LEFT + 50, COURSE_TOP_LEFT + COURSE_WIDTH // 2), 270)    # Left segment, facing up
    }
    
    return segment_targets[next_segment]

def draw_finish_gate(screen, segment):
    """
    Draw the finish gate for the current segment.
    
    Args:
        screen: Pygame screen surface
        segment: Current segment (1=Top, 2=Right, 3=Bottom, 4=Left)
    """
    if segment < 1 or segment > 4:
        return
        
    gate = FINISH_GATE_SEGMENTS[segment - 1]  # Convert to 0-based index
    
    # Draw the gate as a thick cyan line
    pygame.draw.line(screen, COLOR_FINISH_GATE, (gate[0], gate[1]), (gate[2], gate[3]), 8)
    
    # Draw gate markers (small perpendicular lines at ends)
    if segment == 1 or segment == 3:  # Horizontal gates
        # Draw vertical markers
        pygame.draw.line(screen, COLOR_FINISH_GATE, (gate[0], gate[1] - 5), (gate[0], gate[1] + 5), 3)
        pygame.draw.line(screen, COLOR_FINISH_GATE, (gate[2], gate[3] - 5), (gate[2], gate[3] + 5), 3)
    else:  # Vertical gates
        # Draw horizontal markers
        pygame.draw.line(screen, COLOR_FINISH_GATE, (gate[0] - 5, gate[1]), (gate[0] + 5, gate[1]), 3)
        pygame.draw.line(screen, COLOR_FINISH_GATE, (gate[2] - 5, gate[3]), (gate[2] + 5, gate[3]), 3)

def draw_pathfinding_grid(screen, pathfinder):
    """
    Draws the pathfinder's collision grid on the screen for debugging.
    
    Args:
        screen (pygame.Surface): The main display surface.
        pathfinder (Pathfinder): The pathfinder object containing the grid.
    """

    
    # Create a semi-transparent surface to draw the grid on
    grid_surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
    
    # Define colors for the grid visualization
    COLOR_GRID_LINE = (50, 50, 50, 100)  # Faint gray for grid lines
    COLOR_COLLISION_CELL = (255, 0, 0, 80) # Semi-transparent red for obstacles
    
    # Loop through every cell in the pathfinder's grid
    for gx in range(pathfinder.width):
        for gy in range(pathfinder.height):
            # Calculate the screen position and size of the grid cell
            rect = (gx * pathfinder.grid_size, 
                    gy * pathfinder.grid_size, 
                    pathfinder.grid_size, 
                    pathfinder.grid_size)

            # If the cell is marked as a collision, draw a filled red square
            if pathfinder.collision_grid[gx, gy]:
                pygame.draw.rect(grid_surface, COLOR_COLLISION_CELL, rect)
            
            # Always draw the outline of the grid cell
            pygame.draw.rect(grid_surface, COLOR_GRID_LINE, rect, 1)

    # Blit the entire grid surface onto the main screen
    screen.blit(grid_surface, (0, 0))

class Robot:
    def __init__(self, x, y):
        self.x, self.y = x, y
        self.prev_x, self.prev_y = x, y  # Track previous position for gate collision detection
        self.angle = 0.0
        self.steering_angle = 0.0
        self.speed_level = 0
        self.lidar_points, self.estimated_walls, self.inlier_points_for_viz, self.unassociated_points = [], [], [], []
        self.path, self.current_waypoint_index = [], 0
        self.mode = 'MANUAL'
        self.replan_needed = False
        self.consecutive_replan_count = 0
        self.sharp_turn_locations = []
        self.reverse_timer_start = 0
        self.current_segment = 1
        self.last_replan_attempt = 0

    def set_path(self, path):
        if path:
            self.path, self.current_waypoint_index, self.mode = path, 0, 'AUTO'
            self.replan_needed = False
        else:
            self.mode, self.path, self.speed_level = 'MANUAL', [], 0
            self.replan_needed = False
        self.consecutive_replan_count = 0
        self.sharp_turn_locations.clear()

    def follow_path(self):
        if not self.path or self.current_waypoint_index >= len(self.path):
            self.mode, self.speed_level = 'MANUAL', 0
            return

        if self.current_waypoint_index + 1 < len(self.path):
            p_current, p_next, p_future = (self.x, self.y), self.path[self.current_waypoint_index], self.path[self.current_waypoint_index + 1]
            angle1 = math.atan2(p_next[1] - p_current[1], p_next[0] - p_current[0])
            angle2 = math.atan2(p_future[1] - p_next[1], p_future[0] - p_next[0])
            turn_angle = (math.degrees(angle2) - math.degrees(angle1) + 180) % 360 - 180
            if turn_angle == -180: turn_angle = 0

            if abs(turn_angle) > MAX_PATH_TURN_ANGLE:
                self.speed_level = 0
                if p_next not in self.sharp_turn_locations: self.sharp_turn_locations.append(p_next)
                self.consecutive_replan_count += 1
                if self.consecutive_replan_count >= REPLAN_STUCK_THRESHOLD:
                    self.mode = 'REVERSING'
                    self.reverse_timer_start = pygame.time.get_ticks()
                    self.consecutive_replan_count = 0
                    self.path, self.sharp_turn_locations = [], []
                else:
                    self.replan_needed = True
                return

        target_pos = self.path[self.current_waypoint_index]
        dist_to_target = math.hypot(target_pos[0] - self.x, target_pos[1] - self.y)

        if dist_to_target < WAYPOINT_ARRIVAL_THRESHOLD:
            self.current_waypoint_index += 1
            self.consecutive_replan_count = 0
            self.sharp_turn_locations.clear()
            if self.current_waypoint_index >= len(self.path):
                self.mode, self.speed_level = 'MANUAL', 0
                return

        angle_to_target_rad = math.atan2(target_pos[1] - self.y, target_pos[0] - self.x)
        angle_diff = (math.degrees(angle_to_target_rad) - self.angle + 180) % 360 - 180
        self.steering_angle = np.clip(angle_diff * 0.5, -ROBOT_MAX_STEERING_ANGLE, ROBOT_MAX_STEERING_ANGLE)
        self.speed_level = 1

    def update_segment(self):
        c1 = INNER_TR[1] + INNER_TR[0]
        side1 = self.y + self.x - c1
        c2 = INNER_TL[1] - INNER_TL[0]
        side2 = self.y - self.x - c2

        if side1 < 0 and side2 < 0: self.current_segment = 1 # Top
        elif side1 > 0 and side2 < 0: self.current_segment = 2 # Right
        elif side1 > 0 and side2 > 0: self.current_segment = 3 # Bottom
        elif side1 < 0 and side2 > 0: self.current_segment = 4 # Left
        else: self.current_segment = 1

    def update(self, keys_pressed):
        # Store previous position for gate collision detection
        self.prev_x, self.prev_y = self.x, self.y
        
        self.update_segment()
        if self.mode == 'MANUAL':
            steer_input = 0
            if keys_pressed[pygame.K_a]: steer_input = 1
            if keys_pressed[pygame.K_d]: steer_input = -1
            if any(keys_pressed): self.set_path([]); self.sharp_turn_locations.clear()
            if steer_input != 0: self.steering_angle = np.clip(self.steering_angle - steer_input * ROBOT_STEERING_RATE, -ROBOT_MAX_STEERING_ANGLE, ROBOT_MAX_STEERING_ANGLE)
            else: self.steering_angle *= ROBOT_STEERING_RETURN_RATE
        elif self.mode == 'AUTO': self.follow_path()
        elif self.mode == 'REVERSING':
            self.speed_level = -1; self.steering_angle = 0
            if pygame.time.get_ticks() - self.reverse_timer_start > REVERSE_DURATION_MS:
                self.speed_level, self.mode, self.replan_needed = 0, 'AUTO', True
        speed = self.speed_level * ROBOT_BASE_SPEED
        if speed != 0:
            turning_radius = ROBOT_WHEELBASE / math.tan(math.radians(self.steering_angle)) if self.steering_angle != 0 else float('inf')
            angular_velocity_deg = math.degrees(speed / turning_radius)
            self.angle += angular_velocity_deg
        self.angle %= 360
        angle_rad = math.radians(self.angle)
        self.x += speed * math.cos(angle_rad)
        self.y += speed * math.sin(angle_rad)

    def change_speed(self, direction): self.speed_level = max(-1, min(1, self.speed_level + direction))
    def simulate_lidar(self, walls):
        self.lidar_points = []
        robot_angle_rad = math.radians(self.angle)
        for i in range(LIDAR_RAYS):
            ray_angle_rad = robot_angle_rad + math.radians((360 / LIDAR_RAYS) * i)
            end_x, end_y = self.x + LIDAR_RANGE * math.cos(ray_angle_rad), self.y + LIDAR_RANGE * math.sin(ray_angle_rad)
            closest_dist, hit_point = LIDAR_RANGE, None
            for wall in walls:
                p, d = line_intersection((self.x, self.y), (end_x, end_y), (wall[0], wall[1]), (wall[2], wall[3]))
                if p and d < closest_dist: closest_dist, hit_point = d, p
            if hit_point:
                dist_with_error = closest_dist + random.uniform(-LIDAR_INACCURACY, LIDAR_INACCURACY)
                self.lidar_points.append((self.x + dist_with_error * math.cos(ray_angle_rad), self.y + dist_with_error * math.sin(ray_angle_rad)))
    def estimate_walls(self):
        self.estimated_walls, self.inlier_points_for_viz = [], []
        remaining_points = list(self.lidar_points)
        while len(remaining_points) > RANSAC_MIN_INLIERS:
            best_inliers = []
            for _ in range(RANSAC_ITERATIONS):
                if len(remaining_points) < 2: break
                p1, p2 = random.sample(remaining_points, 2)
                current_inliers = [p for p in remaining_points if distance_from_point_to_line(p, p1, p2) < RANSAC_THRESHOLD_DISTANCE]
                if len(current_inliers) > len(best_inliers): best_inliers = current_inliers
            if len(best_inliers) > RANSAC_MIN_INLIERS:
                contiguous_clusters = self.cluster_inliers_by_distance(best_inliers)
                points_to_remove = set()
                for cluster in contiguous_clusters:
                    if len(cluster) > RANSAC_MIN_INLIERS:
                        final_wall_segment = self.fit_line_with_pca(cluster)
                        if final_wall_segment:
                            length = math.hypot(final_wall_segment[1][0] - final_wall_segment[0][0], final_wall_segment[1][1] - final_wall_segment[0][1])
                            if length >= MIN_WALL_LENGTH:
                                self.estimated_walls.append(final_wall_segment)
                                self.inlier_points_for_viz.extend(cluster)
                                for point in cluster: points_to_remove.add(point)
                if not points_to_remove: break
                remaining_points = [p for p in remaining_points if p not in points_to_remove]
            else: break
        self.unassociated_points = remaining_points
    def cluster_inliers_by_distance(self, inliers):
        if not inliers: return []
        data = np.array(inliers)
        mean = np.mean(data, axis=0)
        direction_vector = np.linalg.eigh(np.cov(data.T))[1][:, -1]
        projected = sorted([(np.dot(p - mean, direction_vector), p) for p in inliers], key=lambda x: x[0])
        sorted_inliers = [p for _, p in projected]
        clusters, current_cluster = [], [sorted_inliers[0]]
        for i in range(1, len(sorted_inliers)):
            p1, p2 = sorted_inliers[i-1], sorted_inliers[i]
            if math.hypot(p1[0] - p2[0], p1[1] - p2[1]) < MAX_POINT_DISTANCE: current_cluster.append(p2)
            else: clusters.append(current_cluster); current_cluster = [p2]
        clusters.append(current_cluster)
        return clusters
    def fit_line_with_pca(self, points):
        if len(points) < 2: return None
        data, mean = np.array(points), np.mean(points, axis=0)
        _, eigenvectors = np.linalg.eigh(np.cov(data.T))
        direction_vector = eigenvectors[:, -1]
        projections = np.dot(data - mean, direction_vector)
        return ((mean + np.min(projections) * direction_vector).tolist(), (mean + np.max(projections) * direction_vector).tolist())
    def cluster_unassociated_points(self):
        points_in_arc = [p for p in self.unassociated_points if abs((math.degrees(math.atan2(p[1] - self.y, p[0] - self.x)) - self.angle + 180) % 360 - 180) <= OBSTACLE_DETECTION_ARC_DEGREES / 2.0]
        clusters_found, unvisited_points = [], set(points_in_arc)
        while unvisited_points:
            queue, current_cluster = [unvisited_points.pop()], []
            current_cluster.append(queue[0])
            head = 0
            while head < len(queue):
                current_point = queue[head]; head += 1
                neighbors = [p for p in unvisited_points if math.hypot(current_point[0] - p[0], current_point[1] - p[1]) < OBSTACLE_PROXIMITY]
                for neighbor in neighbors: unvisited_points.remove(neighbor); current_cluster.append(neighbor); queue.append(neighbor)
            if len(current_cluster) >= OBSTACLE_MIN_POINTS:
                center = np.mean(np.array(current_cluster), axis=0)
                radius = max(math.hypot(p[0] - center[0], p[1] - center[1]) for p in current_cluster) + 5
                if radius <= OBSTACLE_MAX_SIZE: clusters_found.append({'center': center.tolist(), 'radius': radius})
        return clusters_found

class Heatmap:
    def __init__(self, width, height, cell_size):
        self.cell_size, self.grid_width, self.grid_height = cell_size, width // cell_size, height // cell_size
        self.grid = np.zeros((self.grid_width, self.grid_height), dtype=float)
        self.surface = pygame.Surface((width, height), pygame.SRCALPHA)
        self.max_hits = 1.0
    def add_points(self, points):
        for x, y in points:
            grid_x, grid_y = int(x // self.cell_size), int(y // self.cell_size)
            if 0 <= grid_x < self.grid_width and 0 <= grid_y < self.grid_height: self.grid[grid_x, grid_y] += 1
    def decay(self, rate=0.99): self.grid *= rate; self.grid[self.grid < 0.1] = 0
    def update_and_draw(self, screen):
        current_max = np.max(self.grid)
        self.max_hits = current_max if current_max > self.max_hits else max(1.0, self.max_hits * HEATMAP_MAX_HITS_ADJUST_RATE)
        self.surface.fill((0, 0, 0, 0))
        non_zero_indices = np.argwhere(self.grid > 0)
        for grid_x, grid_y in non_zero_indices:
            intensity = min(self.grid[grid_x, grid_y] / self.max_hits, 1.0)
            r, g, b = int(255 * intensity**1.2), int(255 * intensity**2.0), int(max(0, 100 * (0.5 - intensity)))
            alpha = int(50 + 200 * intensity)
            rect = (grid_x * self.cell_size, grid_y * self.cell_size, self.cell_size, self.cell_size)
            pygame.draw.rect(self.surface, (r, g, b, alpha), rect)
        screen.blit(self.surface, (0, 0))

class ObstacleManager:
   
    def __init__(self):
        self.potential_obstacles, self.confirmed_obstacles = [], []
        self.debug_rays = [] # <<< ADD THIS LINE

    def reset(self):
        self.potential_obstacles.clear(); self.confirmed_obstacles.clear()
        self.debug_rays.clear() # <<< ADD THIS LINE

    # <<< CHANGE THE SIGNATURE OF THIS FUNCTION
    def update(self, current_frame_clusters, walls_for_validation, robot_position, all_physical_walls, obstacles_for_drawing):
        # (The first part of the function remains the same)
        invalidated_confirmed_indices = [i for i, obs in enumerate(self.confirmed_obstacles) if check_obstacle_wall_collision(obs, walls_for_validation)]
        for i in sorted(invalidated_confirmed_indices, reverse=True): del self.confirmed_obstacles[i]
        for pot_obs in self.potential_obstacles: pot_obs['seen_this_frame'], pot_obs['confidence'] = False, pot_obs['confidence']*OBSTACLE_CONFIDENCE_DECAY;
        for cluster in current_frame_clusters:
            if check_obstacle_wall_collision(cluster, walls_for_validation): continue
            best_match, min_dist = None, OBSTACLE_MATCHING_DISTANCE
            for pot_obs in self.potential_obstacles:
                dist = math.hypot(cluster['center'][0] - pot_obs['center'][0], cluster['center'][1] - pot_obs['center'][1])
                if dist < min_dist: min_dist, best_match = dist, pot_obs
            if best_match:
                best_match['confidence'] += OBSTACLE_CONFIDENCE_INCREMENT; w = 0.2;
                best_match['center'][0] = best_match['center'][0] * (1-w) + cluster['center'][0] * w
                best_match['center'][1] = best_match['center'][1] * (1-w) + cluster['center'][1] * w
                best_match['radius'] = best_match['radius'] * (1-w) + cluster['radius'] * w
                best_match['seen_this_frame'] = True
            else: self.potential_obstacles.append({'center': cluster['center'], 'radius': cluster['radius'], 'confidence': OBSTACLE_CONFIDENCE_INCREMENT, 'seen_this_frame': True})
        
        newly_confirmed_obstacles = []
        newly_confirmed_indices = [i for i, pot_obs in enumerate(self.potential_obstacles) 
                                 if pot_obs['confidence'] >= OBSTACLE_CONFIRMATION_THRESHOLD 
                                 and not any(math.hypot(c['center'][0] - pot_obs['center'][0], 
                                                      c['center'][1] - pot_obs['center'][1]) < (c['radius'] + pot_obs['radius']) 
                                           for c in self.confirmed_obstacles)]
        
        for i in sorted(newly_confirmed_indices, reverse=True):
            obs_to_confirm = self.potential_obstacles.pop(i)
            print(f"New obstacle confirmed: {obs_to_confirm['center']} with radius {obs_to_confirm['radius']}, confidence {obs_to_confirm['confidence']}")
            obs_to_confirm['segment'] = self.getblockSegment(obs_to_confirm['center'])

            # Detect obstacle color
            is_obstacle, obstacle_color = get_obstacle_at_coord(int(obs_to_confirm['center'][0]), int(obs_to_confirm['center'][1]), obstacles_for_drawing)
            obs_to_confirm['color'] = obstacle_color if is_obstacle else COLOR_OBSTACLE_RED  # Default to red if not detected

            # Use the new function to find where the ray would hit
            hit_point = cast_ray_and_find_hit(robot_position, obs_to_confirm['center'], all_physical_walls)
            if hit_point:
                # Store the ray to be drawn later
                self.debug_rays.append({
                    'start': robot_position,
                    'end': hit_point,
                    'time': pygame.time.get_ticks()
                })
                
            else:
                print("Raycast did not hit any physical wall.")

            self.confirmed_obstacles.append(obs_to_confirm)
            newly_confirmed_obstacles.append(obs_to_confirm)
        self.potential_obstacles = [p for p in self.potential_obstacles if p['confidence'] > 0.1]
       
        return newly_confirmed_obstacles

    def getblockSegment(self, point):
        # (This function is unchanged)
        c1 = INNER_TR[1] + INNER_TR[0]
        side1 = point[0] + point[1] - c1
        c2 = INNER_TL[1] - INNER_TL[0]
        side2 = point[1] - point[0] - c2

        if side1 < 0 and side2 < 0: return 1 # Top
        elif side1 > 0 and side2 < 0: return 2 # Right
        elif side1 > 0 and side2 > 0: return 3 # Bottom
        elif side1 < 0 and side2 > 0: return 4 # Left
        
    def draw(self, screen):
        # --- Draw the debug rays ---
        current_time = pygame.time.get_ticks()
        # Filter out old rays while drawing
        active_rays = []
        for ray in self.debug_rays:
            if current_time - ray['time'] < 2000: # Ray visible for 2 seconds
                pygame.draw.line(screen, ORANGE, ray['start'], ray['end'], 2)
                active_rays.append(ray)
        self.debug_rays = active_rays
        # --- End of debug ray drawing ---
        
        for obs in self.confirmed_obstacles: pygame.draw.circle(screen, COLOR_OBSTACLE_DETECTED, obs['center'], obs['radius'], 2)
        s = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        for obs in self.potential_obstacles: alpha = min(200, 20 + int(obs['confidence'] * 40)); pygame.draw.circle(s, (*COLOR_OBSTACLE_POTENTIAL[:3], alpha), obs['center'], obs['radius'])
        screen.blit(s, (0,0))

    def remove_obstacles(self, obstacles_to_remove):
        """Remove specified obstacles from confirmed obstacles list."""
        for obs_to_remove in obstacles_to_remove:
            # Remove from confirmed obstacles
            for i, confirmed_obs in enumerate(self.confirmed_obstacles):
                if (confirmed_obs['center'][0] == obs_to_remove['center'][0] and 
                    confirmed_obs['center'][1] == obs_to_remove['center'][1]):
                    print(f"Removing obstacle at {confirmed_obs['center']}")
                    del self.confirmed_obstacles[i]
                    break
            
            # Also remove from potential obstacles if present
            for i, potential_obs in enumerate(self.potential_obstacles):
                if (potential_obs['center'][0] == obs_to_remove['center'][0] and 
                    potential_obs['center'][1] == obs_to_remove['center'][1]):
                    del self.potential_obstacles[i]
                    break

def create_dynamic_wall_for_obstacle(obstacle_center, obstacle_color, segment, direction_of_travel):
    """
    Create a dynamic wall based on obstacle color and direction of travel.
    Wall length is limited to 200 pixels maximum.
    
    Args:
        obstacle_center: (x, y) position of the obstacle
        obstacle_color: Color of the obstacle (red or green)
        segment: Current segment (1=Top, 2=Right, 3=Bottom, 4=Left)
        direction_of_travel: "clockwise" or "anticlockwise"
    
    Returns:
        tuple: (start_x, start_y, end_x, end_y) representing the wall
    """
    cx, cy = obstacle_center
    MAX_WALL_LENGTH = 200
    
    # Determine if we should pass on the right or left based on color, direction, and rule
    if DYNAMIC_WALL_RULE == "standard":
        if direction_of_travel == "clockwise":
            # For clockwise: red obstacles on right, green on left
            pass_on_right = (obstacle_color == COLOR_OBSTACLE_RED)
        else:
            # For anticlockwise: green obstacles on right, red on left
            pass_on_right = (obstacle_color == COLOR_OBSTACLE_GREEN)
    else:  # swapped
        if direction_of_travel == "clockwise":
            # For clockwise: green obstacles on right, red on left (swapped)
            pass_on_right = (obstacle_color == COLOR_OBSTACLE_GREEN)
        else:
            # For anticlockwise: red obstacles on right, green on left (swapped)
            pass_on_right = (obstacle_color == COLOR_OBSTACLE_RED)
    
    # Create walls based on segment and passing side
    if segment == 1:  # Top segment
        if pass_on_right:
            # Create wall extending down from obstacle
            wall_end_y = min(cy + MAX_WALL_LENGTH, COURSE_TOP_LEFT + COURSE_WIDTH)
            wall_end = (cx, wall_end_y)
        else:
            # Create wall extending up from obstacle
            wall_end_y = max(cy - MAX_WALL_LENGTH, COURSE_TOP_LEFT)
            wall_end = (cx, wall_end_y)
    elif segment == 2:  # Right segment
        if pass_on_right:
            # Create wall extending left from obstacle
            wall_end_x = max(cx - MAX_WALL_LENGTH, COURSE_TOP_LEFT)
            wall_end = (wall_end_x, cy)
        else:
            # Create wall extending right from obstacle
            wall_end_x = min(cx + MAX_WALL_LENGTH, COURSE_TOP_LEFT + COURSE_WIDTH)
            wall_end = (wall_end_x, cy)
    elif segment == 3:  # Bottom segment
        if pass_on_right:
            # Create wall extending up from obstacle
            wall_end_y = max(cy - MAX_WALL_LENGTH, COURSE_TOP_LEFT)
            wall_end = (cx, wall_end_y)
        else:
            # Create wall extending down from obstacle
            wall_end_y = min(cy + MAX_WALL_LENGTH, COURSE_TOP_LEFT + COURSE_WIDTH)
            wall_end = (cx, wall_end_y)
    elif segment == 4:  # Left segment
        if pass_on_right:
            # Create wall extending right from obstacle
            wall_end_x = min(cx + MAX_WALL_LENGTH, COURSE_TOP_LEFT + COURSE_WIDTH)
            wall_end = (wall_end_x, cy)
        else:
            # Create wall extending left from obstacle
            wall_end_x = max(cx - MAX_WALL_LENGTH, COURSE_TOP_LEFT)
            wall_end = (wall_end_x, cy)
    else:
        # Default behavior for unknown segments
        if abs(cy - COURSE_TOP_LEFT) < abs(cy - (COURSE_TOP_LEFT + COURSE_WIDTH)):
            wall_end_y = max(cy - MAX_WALL_LENGTH, COURSE_TOP_LEFT)
        else:
            wall_end_y = min(cy + MAX_WALL_LENGTH, COURSE_TOP_LEFT + COURSE_WIDTH)
        wall_end = (cx, wall_end_y)
    
    return (cx, cy, wall_end[0], wall_end[1])

def create_dynamic_wall_for_obstacle_group(obstacle_group, direction_of_travel):
    """
    Create a dynamic wall for a group of obstacles.
    
    Args:
        obstacle_group: List of obstacles in the group
        direction_of_travel: "clockwise" or "anticlockwise"
    
    Returns:
        tuple: (start_x, start_y, end_x, end_y) representing the wall
    """
    if not obstacle_group:
        return None
    
    # Use the first obstacle's properties as representative
    representative_obstacle = obstacle_group[0]
    
    # Calculate the bounding box of all obstacles in the group
    min_x = min(obs['center'][0] for obs in obstacle_group)
    max_x = max(obs['center'][0] for obs in obstacle_group)
    min_y = min(obs['center'][1] for obs in obstacle_group)
    max_y = max(obs['center'][1] for obs in obstacle_group)
    
    # Use the center of the bounding box as the wall origin
    group_center = ((min_x + max_x) / 2, (min_y + max_y) / 2)
    
    # Create wall using the representative obstacle's color and segment
    return create_dynamic_wall_for_obstacle(
        group_center,
        representative_obstacle.get('color', COLOR_OBSTACLE_RED),
        representative_obstacle['segment'],
        direction_of_travel
    )

def is_direction_allowed(current_pos, next_pos, segment, direction_of_travel):
    """
    Check if a movement direction is allowed based on the direction of travel and current segment.
    
    Args:
        current_pos: Current (x, y) position
        next_pos: Next (x, y) position
        segment: Current segment (1=Top, 2=Right, 3=Bottom, 4=Left)
        direction_of_travel: "clockwise" or "anticlockwise"
    
    Returns:
        bool: True if movement is allowed
    """
    dx = next_pos[0] - current_pos[0]
    dy = next_pos[1] - current_pos[1]
    
    if direction_of_travel == "clockwise":
        # Clockwise movement preferences by segment
        if segment == 1:  # Top - prefer moving right
            return dx >= 0
        elif segment == 2:  # Right - prefer moving down
            return dy >= 0
        elif segment == 3:  # Bottom - prefer moving left
            return dx <= 0
        elif segment == 4:  # Left - prefer moving up
            return dy <= 0
    else:  # anticlockwise
        # Anticlockwise movement preferences by segment
        if segment == 1:  # Top - prefer moving left
            return dx <= 0
        elif segment == 2:  # Right - prefer moving up
            return dy <= 0
        elif segment == 3:  # Bottom - prefer moving right
            return dx >= 0
        elif segment == 4:  # Left - prefer moving down
            return dy >= 0
    
    return True  # Default to allowing movement

def group_nearby_obstacles(obstacles, max_distance=100):
    """
    Group obstacles that are within max_distance of each other.
    Returns both grouped obstacles and obstacles that should be removed.
    
    Args:
        obstacles: List of obstacle dictionaries with 'center' key
        max_distance: Maximum distance between obstacles to group them
    
    Returns:
        tuple: (obstacles_to_keep, obstacles_to_remove)
    """
    if not obstacles:
        return obstacles, []
    
    groups = []
    remaining_obstacles = obstacles.copy()
    
    while remaining_obstacles:
        current_group = [remaining_obstacles.pop(0)]
        
        # Keep adding obstacles that are close to any obstacle in the current group
        found_new = True
        while found_new:
            found_new = False
            for i in range(len(remaining_obstacles) - 1, -1, -1):
                candidate = remaining_obstacles[i]
                
                # Check if candidate is close to any obstacle in current group
                for group_obstacle in current_group:
                    dist = math.hypot(
                        candidate['center'][0] - group_obstacle['center'][0],
                        candidate['center'][1] - group_obstacle['center'][1]
                    )
                    if dist <= max_distance:
                        current_group.append(remaining_obstacles.pop(i))
                        found_new = True
                        break
                
                if found_new:
                    break
        
        groups.append(current_group)
    
    # Separate single obstacles from groups
    obstacles_to_keep = []
    obstacles_to_remove = []
    
    for group in groups:
        if len(group) == 1:
            # Single obstacle - keep it
            obstacles_to_keep.append(group[0])
        else:
            # Group of 2 or more - remove all obstacles in the group
            obstacles_to_remove.extend(group)
            print(f"Removing group of {len(group)} closely spaced obstacles")
    
    return obstacles_to_keep, obstacles_to_remove

# ...existing code...
class Pathfinder:
    def __init__(self, grid_size):
        self.grid_size, self.width, self.height = grid_size, SCREEN_WIDTH // grid_size, SCREEN_HEIGHT // grid_size
        self.collision_grid = np.zeros((self.width, self.height), dtype=bool)
        self.NUM_DIRECTIONS, self.TURN_PENALTY = 8, 1.5
    def build_collision_grid(self, walls, obstacles):
        self.collision_grid.fill(False)
        for wall in walls:
            p1, p2 = np.array(wall[:2]), np.array(wall[2:])
            length = np.linalg.norm(p2 - p1)
            direction = (p2 - p1) / length if length > 0 else np.array([0,0])
            for i in range(int(length)):
                point = p1 + direction * i
                gx, gy = int(point[0] // self.grid_size), int(point[1] // self.grid_size)
                if 0 <= gx < self.width and 0 <= gy < self.height: self.collision_grid[gx, gy] = True
        for obs in obstacles:
            obs_rect = pygame.Rect(0,0, obs['radius']*2, obs['radius']*2); obs_rect.center = obs['center']
            for gx in range(max(0, obs_rect.left // self.grid_size), min(self.width, (obs_rect.right // self.grid_size) + 1)):
                for gy in range(max(0, obs_rect.top // self.grid_size), min(self.height, (obs_rect.bottom // self.grid_size) + 1)): self.collision_grid[gx, gy] = True
    def _discretize_angle(self, angle):
        angle, angle_per_segment = (angle + 360) % 360, 360 / self.NUM_DIRECTIONS
        return int((angle + angle_per_segment / 2) / 360 * self.NUM_DIRECTIONS) % self.NUM_DIRECTIONS
    def find_path(self, start_pos, start_angle, end_pos, end_angle_degrees):
        start_grid_pos, end_grid_pos = (int(start_pos[0] // self.grid_size), int(start_pos[1] // self.grid_size)), (int(end_pos[0] // self.grid_size), int(end_pos[1] // self.grid_size))
        if not (0 <= start_grid_pos[0] < self.width and 0 <= start_grid_pos[1] < self.height and 0 <= end_grid_pos[0] < self.width and 0 <= end_grid_pos[1] < self.height) or self.collision_grid[start_grid_pos] or self.collision_grid[end_grid_pos]: return None
        start_angle_idx, end_angle_idx = self._discretize_angle(start_angle), self._discretize_angle(end_angle_degrees)
        start_node = (*start_grid_pos, start_angle_idx)
        open_set, came_from, g_score = [(0, start_node)], {}, {start_node: 0}; f_score = {start_node: self.heuristic(start_grid_pos, end_grid_pos)}
        while open_set:
            _, current_node = heapq.heappop(open_set)
            current_pos, current_angle_idx = current_node[:2], current_node[2]
            if current_pos == end_grid_pos and current_angle_idx == end_angle_idx: return self.reconstruct_path(came_from, current_node)
            
            # Get current segment for direction filtering
            current_segment = self.get_segment_for_position(current_pos)
            
            for dx, dy in [(0,1), (0,-1), (1,0), (-1,0), (1,1), (1,-1), (-1,1), (-1,-1)]:
                neighbor_pos = (current_pos[0] + dx, current_pos[1] + dy)
                if not (0 <= neighbor_pos[0] < self.width and 0 <= neighbor_pos[1] < self.height) or self.collision_grid[neighbor_pos]: continue
                
                # Check if this direction is allowed based on the direction of travel
                if not is_direction_allowed(current_pos, neighbor_pos, current_segment, DIRECTION_OF_TRAVEL):
                    continue
                
                move_angle = math.degrees(math.atan2(dy, dx))
                neighbor_angle_idx = self._discretize_angle(move_angle)
                neighbor_node, distance_cost = (*neighbor_pos, neighbor_angle_idx), math.hypot(dx, dy)
                angle_diff = abs(current_angle_idx - neighbor_angle_idx)
                if angle_diff > self.NUM_DIRECTIONS / 2: angle_diff = self.NUM_DIRECTIONS - angle_diff
                turning_cost = self.TURN_PENALTY * angle_diff
                tentative_g_score = g_score.get(current_node, float('inf')) + distance_cost + turning_cost
                if tentative_g_score < g_score.get(neighbor_node, float('inf')):
                    came_from[neighbor_node], g_score[neighbor_node] = current_node, tentative_g_score
                    f_score[neighbor_node] = tentative_g_score + self.heuristic(neighbor_pos, end_grid_pos); heapq.heappush(open_set, (f_score[neighbor_node], neighbor_node))
        return None
    def heuristic(self, a, b): return math.hypot(a[0] - b[0], a[1] - b[1])
    def reconstruct_path(self, came_from, current):
        path = [current[:2]];
        while current in came_from: current = came_from[current]; path.append(current[:2])
        return path[::-1]
    def get_segment_for_position(self, pos):
        """Get the segment (1=Top, 2=Right, 3=Bottom, 4=Left) for a given position."""
        x, y = pos[0] * self.grid_size, pos[1] * self.grid_size
        c1 = INNER_TR[1] + INNER_TR[0]
        side1 = y + x - c1
        c2 = INNER_TL[1] - INNER_TL[0]
        side2 = y - x - c2

        if side1 < 0 and side2 < 0: return 1 # Top
        elif side1 > 0 and side2 < 0: return 2 # Right
        elif side1 > 0 and side2 > 0: return 3 # Bottom
        elif side1 < 0 and side2 > 0: return 4 # Left
        else: return 1

def generate_equidistant_waypoints(path_nodes, grid_size, distance):
    if not path_nodes or len(path_nodes) < 2: return []
    world_path = [(n[0] * grid_size + grid_size / 2, n[1] * grid_size + grid_size / 2) for n in path_nodes]
    waypoints, dist_since_last_wp = [world_path[0]], 0.0
    for i in range(len(world_path) - 1):
        p1, p2 = np.array(world_path[i]), np.array(world_path[i+1])
        segment_vec, segment_len = p2 - p1, np.linalg.norm(p2 - p1)
        if segment_len == 0: continue
        segment_dir, current_dist_along_segment = segment_vec / segment_len, 0
        while dist_since_last_wp + (segment_len - current_dist_along_segment) >= distance:
            needed_dist = distance - dist_since_last_wp
            waypoints.append(tuple((p1 + segment_dir * current_dist_along_segment) + segment_dir * needed_dist))
            current_dist_along_segment += needed_dist; dist_since_last_wp = 0.0
        dist_since_last_wp += (segment_len - current_dist_along_segment)
    waypoints.append(world_path[-1])
    return waypoints

def main():
    global DIRECTION_OF_TRAVEL, DYNAMIC_WALL_RULE
    
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Autonomous Robot SLAM Simulation")
    clock = pygame.time.Clock()
    font, small_font = pygame.font.Font(None, 28), pygame.font.Font(None, 22)
    arc_surface, path_surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA), pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
    pathfinder = Pathfinder(PATHFINDING_GRID_SIZE)
    robot, heatmap, obstacle_manager = Robot(150, 150), Heatmap(SCREEN_WIDTH, SCREEN_HEIGHT, HEATMAP_CELL_SIZE), ObstacleManager()
    obstacles_for_drawing, obstacle_walls = generate_obstacles()
    pathfinding_state, pending_target_pos, final_goal_pos, final_goal_angle = "IDLE", None, None, None
    
    # Auto-target system
    auto_target_active = False
    previous_segment = robot.current_segment  # Track segment changes for auto-target
    
    running = True
    while running:
        dynamic_walls = [obs['dynamic_wall'] for obs in obstacle_manager.confirmed_obstacles if 'dynamic_wall' in obs]
        walls_for_lidar_simulation = COURSE_WALLS + obstacle_walls
        walls_for_robot_logic = COURSE_WALLS + obstacle_walls + dynamic_walls

        # Check for finish gate collision
        if check_finish_gate_collision((robot.x, robot.y), (robot.prev_x, robot.prev_y), robot.current_segment):
            print(f"Finish gate passed! Regenerating obstacles and setting new target.")
            # Regenerate obstacles
            obstacles_for_drawing, obstacle_walls = generate_obstacles()
            obstacle_manager.reset()
            
            # Swap dynamic wall rule
            DYNAMIC_WALL_RULE = "swapped" if DYNAMIC_WALL_RULE == "standard" else "standard"
            print(f"Dynamic wall rule changed to: {DYNAMIC_WALL_RULE}")
            
            # Set new target in opposite segment
            target_segment = get_opposite_segment(robot.current_segment)
            target_pos, target_angle = get_target_position_for_segment(target_segment)
            
            # Try to find path to new target
            pathfinder.build_collision_grid(walls_for_robot_logic, obstacle_manager.confirmed_obstacles)
            path_nodes = pathfinder.find_path((robot.x, robot.y), robot.angle, target_pos, target_angle)
            
            if path_nodes:
                raw_waypoints = generate_equidistant_waypoints(path_nodes, PATHFINDING_GRID_SIZE, WAYPOINT_DISTANCE)
                startup_wp = (robot.x + (WAYPOINT_DISTANCE / 2.0) * math.cos(math.radians(robot.angle)), 
                             robot.y + (WAYPOINT_DISTANCE / 2.0) * math.sin(math.radians(robot.angle)))
                robot.set_path([startup_wp] + raw_waypoints)
                final_goal_pos, final_goal_angle = target_pos, target_angle
                auto_target_active = True
                print(f"Auto-target set to opposite segment {target_segment}")
            else:
                print("Could not find path to new target")

        # Check for segment changes when auto-target is active
        if auto_target_active and robot.current_segment != previous_segment:
            print(f"Robot entered segment {robot.current_segment} from segment {previous_segment}")
            # Set target in opposite segment
            target_segment = get_opposite_segment(robot.current_segment)
            target_pos, target_angle = get_target_position_for_segment(target_segment)
            
            # Try to find path to opposite segment
            pathfinder.build_collision_grid(walls_for_robot_logic, obstacle_manager.confirmed_obstacles)
            path_nodes = pathfinder.find_path((robot.x, robot.y), robot.angle, target_pos, target_angle)
            
            if path_nodes:
                raw_waypoints = generate_equidistant_waypoints(path_nodes, PATHFINDING_GRID_SIZE, WAYPOINT_DISTANCE)
                startup_wp = (robot.x + (WAYPOINT_DISTANCE / 2.0) * math.cos(math.radians(robot.angle)), 
                             robot.y + (WAYPOINT_DISTANCE / 2.0) * math.sin(math.radians(robot.angle)))
                robot.set_path([startup_wp] + raw_waypoints)
                final_goal_pos, final_goal_angle = target_pos, target_angle
                print(f"Auto-target set to opposite segment {target_segment}")
            else:
                print(f"Could not find path to opposite segment {target_segment}")
            
            previous_segment = robot.current_segment

        # Check if robot reached target or lost target
        if auto_target_active and (robot.mode == 'MANUAL' or not robot.path):
            # Target reached or lost, set new target in opposite segment
            target_segment = get_opposite_segment(robot.current_segment)
            target_pos, target_angle = get_target_position_for_segment(target_segment)
            
            # Try to find path to opposite segment
            pathfinder.build_collision_grid(walls_for_robot_logic, obstacle_manager.confirmed_obstacles)
            path_nodes = pathfinder.find_path((robot.x, robot.y), robot.angle, target_pos, target_angle)
            
            if path_nodes:
                raw_waypoints = generate_equidistant_waypoints(path_nodes, PATHFINDING_GRID_SIZE, WAYPOINT_DISTANCE)
                startup_wp = (robot.x + (WAYPOINT_DISTANCE / 2.0) * math.cos(math.radians(robot.angle)), 
                             robot.y + (WAYPOINT_DISTANCE / 2.0) * math.sin(math.radians(robot.angle)))
                robot.set_path([startup_wp] + raw_waypoints)
                final_goal_pos, final_goal_angle = target_pos, target_angle
                print(f"New auto-target set to opposite segment {target_segment}")
            else:
                print(f"Could not find path to opposite segment {target_segment}, trying alternative position")
                # Try alternative position in current segment
                if robot.current_segment == 1:
                    alt_target = (COURSE_TOP_LEFT + COURSE_WIDTH // 4, COURSE_TOP_LEFT + 50)
                elif robot.current_segment == 2:
                    alt_target = (COURSE_TOP_LEFT + COURSE_WIDTH - 50, COURSE_TOP_LEFT + COURSE_WIDTH // 4)
                elif robot.current_segment == 3:
                    alt_target = (COURSE_TOP_LEFT + 3 * COURSE_WIDTH // 4, COURSE_TOP_LEFT + COURSE_WIDTH - 50)
                else:
                    alt_target = (COURSE_TOP_LEFT + 50, COURSE_TOP_LEFT + 3 * COURSE_WIDTH // 4)
                
                path_nodes = pathfinder.find_path((robot.x, robot.y), robot.angle, alt_target, target_angle)
                if path_nodes:
                    raw_waypoints = generate_equidistant_waypoints(path_nodes, PATHFINDING_GRID_SIZE, WAYPOINT_DISTANCE)
                    startup_wp = (robot.x + (WAYPOINT_DISTANCE / 2.0) * math.cos(math.radians(robot.angle)), 
                                 robot.y + (WAYPOINT_DISTANCE / 2.0) * math.sin(math.radians(robot.angle)))
                    robot.set_path([startup_wp] + raw_waypoints)
                    final_goal_pos, final_goal_angle = alt_target, target_angle
                    print("Alternative target set in current segment")
                else:
                    auto_target_active = False
                    print("Could not find any valid path, disabling auto-target")

        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                auto_target_active = False  # Disable auto-target when user clicks
                previous_segment = robot.current_segment  # Update tracking when manual control taken
                if pathfinding_state == "IDLE":
                    pending_target_pos, pathfinding_state = event.pos, "AWAITING_DIRECTION"
                    robot.set_path([]); robot.sharp_turn_locations.clear()
                elif pathfinding_state == "AWAITING_DIRECTION":
                    pathfinder.build_collision_grid(walls_for_robot_logic, obstacle_manager.confirmed_obstacles)
                    
                    dx, dy = event.pos[0] - pending_target_pos[0], event.pos[1] - pending_target_pos[1]
                    target_angle_degrees = math.degrees(math.atan2(dy, dx))
                    path_nodes = pathfinder.find_path((robot.x, robot.y), robot.angle, pending_target_pos, target_angle_degrees)
                    if path_nodes:
                        raw_waypoints = generate_equidistant_waypoints(path_nodes, PATHFINDING_GRID_SIZE, WAYPOINT_DISTANCE)
                        startup_wp = (robot.x + (WAYPOINT_DISTANCE / 2.0) * math.cos(math.radians(robot.angle)), robot.y + (WAYPOINT_DISTANCE / 2.0) * math.sin(math.radians(robot.angle)))
                        robot.set_path([startup_wp] + raw_waypoints)
                        final_goal_pos, final_goal_angle = pending_target_pos, target_angle_degrees
                    else:
                        print("No path found!"); robot.set_path([]); final_goal_pos, final_goal_angle = None, None
                    pathfinding_state, pending_target_pos = "IDLE", None
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    obstacles_for_drawing, obstacle_walls = generate_obstacles()
                    obstacle_manager.reset(); robot.set_path([]); robot.sharp_turn_locations.clear()
                    final_goal_pos, final_goal_angle, pathfinding_state = None, None, "IDLE"
                    auto_target_active = False
                if event.key == pygame.K_c:
                    obstacle_manager.reset()
                    if robot.mode == 'AUTO': robot.replan_needed = True
                if event.key == pygame.K_t:
                    # Toggle direction of travel
                    DIRECTION_OF_TRAVEL = "anticlockwise" if DIRECTION_OF_TRAVEL == "clockwise" else "clockwise"
                    print(f"Direction changed to: {DIRECTION_OF_TRAVEL}")
                    # Clear existing dynamic walls and force replan
                    obstacle_manager.reset()
                    if robot.mode == 'AUTO': robot.replan_needed = True
                if event.key == pygame.K_g:
                    # Toggle dynamic wall rule
                    DYNAMIC_WALL_RULE = "swapped" if DYNAMIC_WALL_RULE == "standard" else "standard"
                    print(f"Dynamic wall rule changed to: {DYNAMIC_WALL_RULE}")
                    # Clear existing dynamic walls and force replan
                    obstacle_manager.reset()
                    if robot.mode == 'AUTO': robot.replan_needed = True
                if event.key == pygame.K_SPACE:
                    # Toggle auto-target mode
                    auto_target_active = not auto_target_active
                    if auto_target_active:
                        print("Auto-target mode enabled")
                        # Set initial target to opposite segment
                        target_segment = get_opposite_segment(robot.current_segment)
                        target_pos, target_angle = get_target_position_for_segment(target_segment)
                        pathfinder.build_collision_grid(walls_for_robot_logic, obstacle_manager.confirmed_obstacles)
                        path_nodes = pathfinder.find_path((robot.x, robot.y), robot.angle, target_pos, target_angle)
                        if path_nodes:
                            raw_waypoints = generate_equidistant_waypoints(path_nodes, PATHFINDING_GRID_SIZE, WAYPOINT_DISTANCE)
                            startup_wp = (robot.x + (WAYPOINT_DISTANCE / 2.0) * math.cos(math.radians(robot.angle)), 
                                         robot.y + (WAYPOINT_DISTANCE / 2.0) * math.sin(math.radians(robot.angle)))
                            robot.set_path([startup_wp] + raw_waypoints)
                            final_goal_pos, final_goal_angle = target_pos, target_angle
                            print(f"Initial auto-target set to opposite segment {target_segment}")
                        else:
                            print("Could not find path to opposite segment")
                    else:
                        print("Auto-target mode disabled")
                        robot.set_path([])
                        final_goal_pos, final_goal_angle = None, None
                if event.key in [pygame.K_w, pygame.K_s, pygame.K_a, pygame.K_d]:
                    robot.set_path([]); robot.sharp_turn_locations.clear()
                    final_goal_pos, final_goal_angle, pathfinding_state = None, None, "IDLE"
                    auto_target_active = False
                    previous_segment = robot.current_segment  # Update tracking when manual control taken
                if robot.mode == 'MANUAL':
                    if event.key == pygame.K_w: robot.change_speed(1)
                    if event.key == pygame.K_s: robot.change_speed(-1)

        robot.update(pygame.key.get_pressed())
        
        # Handle path recalculation with timing
        current_time = pygame.time.get_ticks()
        if robot.replan_needed:
            # Only attempt replan if enough time has passed (500ms)
            if current_time - robot.last_replan_attempt >= 500:
                robot.last_replan_attempt = current_time
                robot.replan_needed = False
                
                if final_goal_pos and final_goal_angle is not None:
                    pathfinder.build_collision_grid(walls_for_robot_logic, obstacle_manager.confirmed_obstacles)
                    new_path_nodes = pathfinder.find_path((robot.x, robot.y), robot.angle, final_goal_pos, final_goal_angle)
                    
                    if new_path_nodes:
                        # Successfully found new path
                        raw_waypoints = generate_equidistant_waypoints(new_path_nodes, PATHFINDING_GRID_SIZE, WAYPOINT_DISTANCE)
                        startup_wp = (robot.x + (WAYPOINT_DISTANCE / 2.0) * math.cos(math.radians(robot.angle)), robot.y + (WAYPOINT_DISTANCE / 2.0) * math.sin(math.radians(robot.angle)))
                        robot.set_path(raw_waypoints)
                        print("Successfully recalculated path")
                    else:
                        # Failed to find new path - continue on current path and try again later
                        robot.replan_needed = True
                        print("Failed to find new path - continuing on current path, will retry in 500ms")
                else:
                    # No goal set - stop the robot
                    robot.set_path([])
       
        robot.simulate_lidar(walls_for_lidar_simulation)
        robot.estimate_walls()
        
        # --- LOGIC FIX: Create a specific list of walls for validating new clusters. ---
        # This list should only contain what the robot has learned (RANSAC walls) or created (dynamic walls).
        # It must NOT contain the ground-truth physical obstacle walls.
        walls_for_cluster_validation = robot.estimated_walls + dynamic_walls
        newly_confirmed = obstacle_manager.update(robot.cluster_unassociated_points(), walls_for_cluster_validation, (robot.x, robot.y), walls_for_lidar_simulation, obstacles_for_drawing)

        if newly_confirmed:
            robot.replan_needed = True
            
            # Group nearby obstacles - remove groups of 2 or more
            obstacles_to_keep, obstacles_to_remove = group_nearby_obstacles(newly_confirmed, max_distance=100)
            
            # Remove grouped obstacles
            if obstacles_to_remove:
                obstacle_manager.remove_obstacles(obstacles_to_remove)
            
            # Create dynamic walls for remaining single obstacles
            for obs in obstacles_to_keep:
                segment, (cx, cy) = obs['segment'], obs['center']
                obstacle_color = obs.get('color', COLOR_OBSTACLE_RED)
                
                print(f"Creating dynamic wall for single obstacle at ({cx}, {cy})")
                
                # Create dynamic wall using the direction-based logic
                obs['dynamic_wall'] = create_dynamic_wall_for_obstacle((cx, cy), obstacle_color, segment, DIRECTION_OF_TRAVEL)
                print(f"Created dynamic wall: {obs['dynamic_wall']}")
                print("---")
                    
        obstacle_manager.draw(screen)
        heatmap.add_points(robot.lidar_points); heatmap.decay(HEATMAP_DECAY_RATE)
        screen.fill(COLOR_BACKGROUND); heatmap.update_and_draw(screen)
        for wall in COURSE_WALLS: pygame.draw.line(screen, COLOR_WALL, (wall[0], wall[1]), (wall[2], wall[3]), 2)
        for obs in obstacles_for_drawing: pygame.draw.rect(screen, obs['color'], obs['rect']); pygame.draw.rect(screen, COLOR_WALL, obs['rect'], 1)
        for wall in dynamic_walls: pygame.draw.line(screen, COLOR_DYNAMIC_WALL, (wall[0], wall[1]), (wall[2], wall[3]), 4)
        pygame.draw.line(screen, COLOR_SEGMENT_BORDER, INNER_TR, INNER_BL, 1)
        pygame.draw.line(screen, COLOR_SEGMENT_BORDER, INNER_TL, INNER_BR, 1)
        
        # Draw finish gate
        draw_finish_gate(screen, robot.current_segment)
        
        for p in robot.inlier_points_for_viz: pygame.draw.circle(screen, COLOR_INLIER_POINTS, p, 3)
        # --- Optional Debugging: Uncomment the line below to see the points considered for clustering ---
        # for p in robot.unassociated_points: pygame.draw.circle(screen, COLOR_UNASSOCIATED_POINTS_DEBUG, p, 2)
        for wall in robot.estimated_walls: pygame.draw.line(screen, COLOR_ESTIMATED_WALL, wall[0], wall[1], 5)
        obstacle_manager.draw(screen)
        if SHOW_GRID:
            draw_pathfinding_grid(screen, pathfinder)
        robot_pos = (robot.x, robot.y)

        for wall_segment in robot.estimated_walls:
            p_start, p_end = wall_segment[0], wall_segment[1]
            closest_point = find_closest_point_on_segment(robot_pos, p_start, p_end)
            distance = math.hypot(robot_pos[0] - closest_point[0], robot_pos[1] - closest_point[1])
            vec_robot_to_point, vec_wall = np.array(closest_point) - np.array(robot_pos), np.array(p_end) - np.array(p_start)
            mag_robot_vec, mag_wall_vec = np.linalg.norm(vec_robot_to_point), np.linalg.norm(vec_wall)
            angle_deg = 90 if mag_robot_vec==0 or mag_wall_vec==0 else abs(90 - math.degrees(math.acos(np.clip(np.dot(vec_robot_to_point, vec_wall) / (mag_robot_vec * mag_wall_vec), -1.0, 1.0))))
            line_color = COLOR_PERPENDICULAR if angle_deg <= PERPENDICULAR_ANGLE_THRESHOLD else COLOR_NON_PERPENDICULAR
            pygame.draw.line(screen, line_color, robot_pos, closest_point, 2)
            text_surface = small_font.render(f"{distance:.0f}", True, COLOR_TEXT)
            mid_point = ((robot_pos[0] + closest_point[0]) / 2, (robot_pos[1] + closest_point[1]) / 2)
            text_rect = text_surface.get_rect(center=mid_point)
            pygame.draw.rect(screen, COLOR_BACKGROUND, text_rect.inflate(6, 4)); screen.blit(text_surface, text_rect)

        arc_surface.fill((0, 0, 0, 0))
        poly_points = [(robot.x, robot.y)]
        half_arc_rad, robot_angle_rad = math.radians(OBSTACLE_DETECTION_ARC_DEGREES / 2), math.radians(robot.angle)
        for i in range(21): poly_points.append((robot.x + UI_ARC_RADIUS * math.cos(robot_angle_rad - half_arc_rad + (half_arc_rad*2*i/20)), robot.y + UI_ARC_RADIUS * math.sin(robot_angle_rad - half_arc_rad + (half_arc_rad*2*i/20))))
        pygame.draw.polygon(arc_surface, COLOR_DETECTION_ARC, poly_points); screen.blit(arc_surface, (0, 0))

        path_surface.fill((0, 0, 0, 0))
        if robot.path:
            if robot.current_waypoint_index < len(robot.path): pygame.draw.lines(path_surface, COLOR_PLANNED_PATH, False, [robot_pos] + robot.path[robot.current_waypoint_index:], 3)
            for i in range(robot.current_waypoint_index, len(robot.path)): pygame.draw.circle(path_surface, COLOR_WAYPOINT, robot.path[i], 5)
            if final_goal_angle is not None:
                last_wp = robot.path[-1]; angle_rad = math.radians(final_goal_angle)
                pygame.draw.line(path_surface, COLOR_WAYPOINT, last_wp, (last_wp[0] + 25 * math.cos(angle_rad), last_wp[1] + 25 * math.sin(angle_rad)), 4)
        if pathfinding_state == "AWAITING_DIRECTION":
            pygame.draw.circle(path_surface, COLOR_WAYPOINT, pending_target_pos, 8); pygame.draw.line(path_surface, COLOR_WAYPOINT, pending_target_pos, pygame.mouse.get_pos(), 2)
        screen.blit(path_surface, (0,0))
        for pos in robot.sharp_turn_locations: pygame.draw.circle(screen, COLOR_SHARP_TURN, pos, 15, 2)

        pygame.draw.circle(screen, COLOR_ROBOT, robot_pos, ROBOT_SIZE); pygame.draw.circle(screen, (255,255,255), robot_pos, ROBOT_SIZE, 1)
        end_x, end_y = robot.x + ROBOT_SIZE * math.cos(math.radians(robot.angle)), robot.y + ROBOT_SIZE * math.sin(math.radians(robot.angle))
        pygame.draw.line(screen, (255, 255, 255), robot_pos, (end_x, end_y), 2)
        steer_angle_rad = math.radians(robot.angle - robot.steering_angle)
        steer_x, steer_y = robot.x + ROBOT_SIZE * 0.8 * math.cos(steer_angle_rad), robot.y + ROBOT_SIZE * 0.8 * math.sin(steer_angle_rad)
        pygame.draw.line(screen, (0, 255, 0), robot_pos, (steer_x, steer_y), 2)

        if robot.mode == 'MANUAL': 
            if auto_target_active:
                instruction_text_str = "Auto-target ACTIVE (opposite segment). Click: manual target. WASD: drive. Space: toggle auto. G: swap walls. R: obstacles. C: clear. T: direction"
            else:
                instruction_text_str = "Click: set dest. WASD: drive. Space: auto-target (opposite). G: swap walls. R: obstacles. C: clear. T: direction" if pathfinding_state=="IDLE" else "Click to set the desired FINAL DIRECTION."
        elif robot.mode == 'AUTO': 
            instruction_text_str = "Mode: AUTO. Navigating to opposite segment. Press WASD to take over. Space: toggle auto-target."
        elif robot.mode == 'REVERSING': 
            instruction_text_str = "Mode: REVERSING. Attempting to get unstuck..."
        if robot.replan_needed: 
            instruction_text_str = f"Mode: AUTO. Replanning path... Attempt #{robot.consecutive_replan_count}"
        screen.blit(font.render(instruction_text_str, True, COLOR_TEXT), (10, 10))
        segment_map = {1: "Top", 2: "Right", 3: "Bottom", 4: "Left"}
        segment_text = f"Segment: {robot.current_segment} ({segment_map.get(robot.current_segment, 'N/A')})"
        screen.blit(small_font.render(segment_text, True, COLOR_TEXT), (10, 40))
        
        # Display direction of travel
        direction_text = f"Direction: {DIRECTION_OF_TRAVEL.capitalize()}"
        screen.blit(small_font.render(direction_text, True, COLOR_TEXT), (10, 65))
        
        # Display dynamic wall rule
        rule_text = f"Wall Rule: {DYNAMIC_WALL_RULE.capitalize()}"
        screen.blit(small_font.render(rule_text, True, COLOR_TEXT), (10, 90))
        
        # Display obstacle handling info based on current rule
        if DYNAMIC_WALL_RULE == "standard":
            info_text = f"Red obstacles: pass on {'right' if DIRECTION_OF_TRAVEL == 'clockwise' else 'left'}, Green obstacles: pass on {'left' if DIRECTION_OF_TRAVEL == 'clockwise' else 'right'}"
        else:  # swapped
            info_text = f"Red obstacles: pass on {'left' if DIRECTION_OF_TRAVEL == 'clockwise' else 'right'}, Green obstacles: pass on {'right' if DIRECTION_OF_TRAVEL == 'clockwise' else 'left'}"
        screen.blit(small_font.render(info_text, True, COLOR_TEXT), (10, 115))
        
        # Display auto-target status
        if auto_target_active:
            target_segment = get_opposite_segment(robot.current_segment)
            segment_names = {1: "Top", 2: "Right", 3: "Bottom", 4: "Left"}
            auto_text = f"Auto-target: ON → {segment_names.get(target_segment, 'Unknown')}"
        else:
            auto_text = "Auto-target: OFF"
        screen.blit(small_font.render(auto_text, True, COLOR_TEXT), (10, 140))

        pygame.display.flip()
        clock.tick(60)
    pygame.quit()

if __name__ == '__main__':
    main()