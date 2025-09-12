# lidar_main.py
# This script is the final, high-performance implementation based on all our findings.
# It uses a checksum-agnostic reader in a separate thread and a fast Pygame renderer.
# MODIFIED to include an isolation filter and RANSAC for wall detection.
# CORRECTED to properly handle MAX_POINT_DISTANCE_MM for wall clustering.
# CORRECTED to show a static robot arrow, as the view is robot-centric.
# MODIFIED to highlight decision-making walls with translucent blocks.
# MODIFIED to categorize walls into Front, Right, Back, Left and measure distances.

import serial
import RPi.GPIO as GPIO
import time
import struct
import math
import pygame
import threading
import collections
import random
import numpy as np # Required for RANSAC's PCA line fitting

# --- Thread-Safe Shared Data Structure ---
lidar_points_polar = collections.deque(maxlen=450)
stop_event = threading.Event()

# --- LiDAR and GPIO Configuration ---
PWM_PIN = 18
SERIAL_PORT = '/dev/serial0'
BAUD_RATE = 230400
MOTOR_DUTY_CYCLE = 65

# --- Pygame Configuration ---
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 800
MAX_VIZ_DISTANCE_MM = 4000
SCALE_FACTOR = MAX_VIZ_DISTANCE_MM / (SCREEN_WIDTH / 2 - 20)
BACKGROUND_COLOR = (10, 10, 10)
POINT_COLOR = (100, 200, 255)
FILTERED_POINT_COLOR = (0, 100, 255)
WALL_COLOR = (100, 255, 100)
DISTANCE_LINE_COLOR = (0, 170, 255)
DISTANCE_TEXT_COLOR = (255, 255, 255)
ARROW_COLOR = (255, 165, 0) # Orange for the arrow
ARROW_LENGTH_PX = 60
ARROW_HEAD_LENGTH_PX = 20
ARROW_HEAD_ANGLE_DEG = 150
HIGHLIGHT_COLOR = (255, 0, 0, 60) # Red with 60/255 alpha (translucent)
HIGHLIGHT_THICKNESS_PX = 40

# --- Filter and RANSAC Configuration ---
LIDAR_RANGE_MM = 3000.0
ISOLATION_DISTANCE_MM = 60.0
ISOLATION_MIN_NEIGHBORS = 3
RANSAC_ITERATIONS = 25
RANSAC_THRESHOLD_MM = 80.0
RANSAC_MIN_INLIERS = 50
MAX_POINT_DISTANCE_MM = 80.0
MIN_WALL_LENGTH_MM = 150.0

# --- Direction Detection Configuration ---
DIRECTION_TOLERANCE_MM = 150.0 # Wall lengths must differ by this much to determine direction

### NEW ###
# --- Wall Categorization Configuration ---
WALL_ANGLE_TOLERANCE_DEG = 15 # Tolerance for a wall to be considered horizontal/vertical

# --- LiDAR Reader Thread (Unchanged) ---
def lidar_reader_thread():
    # ... (This function is unchanged)
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1)
    except serial.SerialException as e:
        print(f"FATAL: Could not open serial port: {e}")
        stop_event.set(); return

    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
    GPIO.setup(PWM_PIN, GPIO.OUT)
    motor_pwm = GPIO.PWM(PWM_PIN, 1000)
    motor_pwm.start(MOTOR_DUTY_CYCLE)
    print("Motor started. Reader thread running.")

    packet_buffer = bytearray()
    while not stop_event.is_set():
        try:
            data = ser.read(4096)
            if not data:
                time.sleep(0.001); continue
            packet_buffer.extend(data)
            while len(packet_buffer) >= 47:
                header_index = packet_buffer.find(0x54)
                if header_index == -1:
                    packet_buffer.clear(); break
                if header_index > 0:
                    packet_buffer = packet_buffer[header_index:]
                if len(packet_buffer) < 47: break
                packet = packet_buffer[:47]
                start_angle = struct.unpack('<H', bytes(packet[4:6]))[0]
                end_angle = struct.unpack('<H', bytes(packet[42:44]))[0]
                start_angle_deg = start_angle / 100.0
                end_angle_deg = end_angle / 100.0
                angle_diff = end_angle_deg - start_angle_deg
                if angle_diff < 0: angle_diff += 360.0
                for i in range(12):
                    offset = 6 + (i * 3)
                    distance_mm = struct.unpack('<H', bytes(packet[offset:offset+2]))[0]
                    confidence = packet[offset+2]
                    if distance_mm > 0 and confidence > 180:
                        angle_step = angle_diff / 11 if 11 > 0 else 0
                        current_angle_deg = start_angle_deg + i * angle_step
                        lidar_points_polar.append((math.radians(current_angle_deg % 360), distance_mm))
                packet_buffer = packet_buffer[47:]
        except Exception as e:
            print(f"Error in reader thread: {e}"); break
    motor_pwm.stop()
    GPIO.cleanup()
    ser.close()
    print("Reader thread has stopped.")

# --- Helper Functions (Existing functions are unchanged) ---
def filter_isolated_points(cartesian_points, min_neighbors, isolation_radius):
    if not cartesian_points: return []
    grid_cell_size = isolation_radius
    grid = collections.defaultdict(list)
    for i, point in enumerate(cartesian_points):
        cell = (int(point[0] // grid_cell_size), int(point[1] // grid_cell_size))
        grid[cell].append(i)
    filtered_points = []
    isolation_radius_sq = isolation_radius ** 2
    for i, p1 in enumerate(cartesian_points):
        neighbor_count = 0
        p1_cell = (int(p1[0] // grid_cell_size), int(p1[1] // grid_cell_size))
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                check_cell = (p1_cell[0] + dx, p1_cell[1] + dy)
                if check_cell in grid:
                    for neighbor_idx in grid[check_cell]:
                        p2 = cartesian_points[neighbor_idx]
                        dist_sq = (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2
                        if dist_sq < isolation_radius_sq:
                            neighbor_count += 1
        if neighbor_count >= min_neighbors:
            filtered_points.append(p1)
    return filtered_points

def distance_from_point_to_line(p, l1, l2):
    x0, y0 = p; x1, y1 = l1; x2, y2 = l2
    num = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)
    den = math.sqrt((y2 - y1)**2 + (x2 - x1)**2)
    return num / den if den != 0 else 0

def fit_line_with_pca(points):
    if len(points) < 2: return None
    data = np.array(points)
    mean = np.mean(data, axis=0)
    _, eigenvectors = np.linalg.eigh(np.cov(data.T))
    direction_vector = eigenvectors[:, -1]
    projections = np.dot(data - mean, direction_vector)
    p1 = mean + np.min(projections) * direction_vector
    p2 = mean + np.max(projections) * direction_vector
    return (p1.tolist(), p2.tolist())

def cluster_inliers_by_distance(inliers, max_distance):
    if not inliers: return []
    data = np.array(inliers)
    mean = np.mean(data, axis=0)
    _, eigenvectors = np.linalg.eigh(np.cov(data.T))
    direction_vector = eigenvectors[:, -1]
    projected = sorted([(np.dot(p - mean, direction_vector), p) for p in inliers], key=lambda x: x[0])
    sorted_inliers = [p for _, p in projected]
    clusters, current_cluster = [], [sorted_inliers[0]]
    for i in range(1, len(sorted_inliers)):
        p1, p2 = sorted_inliers[i-1], sorted_inliers[i]
        if math.hypot(p1[0] - p2[0], p1[1] - p2[1]) < max_distance:
            current_cluster.append(p2)
        else:
            clusters.append(current_cluster)
            current_cluster = [p2]
    clusters.append(current_cluster)
    return clusters

def run_ransac(points):
    estimated_walls, remaining_points = [], list(points)
    while len(remaining_points) > RANSAC_MIN_INLIERS:
        best_inliers = []
        for _ in range(RANSAC_ITERATIONS):
            if len(remaining_points) < 2: break
            p1, p2 = random.sample(remaining_points, 2)
            current_inliers = [p for p in remaining_points if distance_from_point_to_line(p, p1, p2) < RANSAC_THRESHOLD_MM]
            if len(current_inliers) > len(best_inliers):
                best_inliers = current_inliers
        if len(best_inliers) > RANSAC_MIN_INLIERS:
            contiguous_clusters = cluster_inliers_by_distance(best_inliers, MAX_POINT_DISTANCE_MM)
            points_to_remove = set()
            for cluster in contiguous_clusters:
                if len(cluster) >= RANSAC_MIN_INLIERS:
                    final_wall_segment = fit_line_with_pca(cluster)
                    if final_wall_segment:
                        p_start, p_end = final_wall_segment
                        length = math.hypot(p_end[0] - p_start[0], p_end[1] - p_start[1])
                        if length >= MIN_WALL_LENGTH_MM:
                            estimated_walls.append(final_wall_segment)
                            for point in cluster: points_to_remove.add(tuple(point))
            if not points_to_remove: break
            remaining_points = [p for p in remaining_points if tuple(p) not in points_to_remove]
        else: break
    return estimated_walls

def find_closest_point_on_segment(p, a, b):
    p_np, a_np, b_np = np.array(p), np.array(a), np.array(b)
    ab, ap = b_np - a_np, p_np - a_np
    ab_len_sq = np.dot(ab, ab)
    if ab_len_sq == 0: return tuple(a)
    t = np.clip(np.dot(ap, ab) / ab_len_sq, 0, 1)
    return tuple(a_np + t * ab)

def determine_direction(walls):
    """
    Determines direction and also returns the lists of walls used for the decision.
    """
    if not walls or len(walls) < 2:
        return "Undetermined", [], []

    upper_walls, lower_walls = [], []
    total_upper_length, total_lower_length = 0.0, 0.0

    for wall in walls:
        p1, p2 = wall
        length = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
        midpoint_y = (p1[1] + p2[1]) / 2.0
        
        if midpoint_y > 0:
            upper_walls.append(wall)
            total_upper_length += length
        elif midpoint_y < 0:
            lower_walls.append(wall)
            total_lower_length += length
    
    direction_string = "Undetermined / Straight"
    if total_lower_length > total_upper_length + DIRECTION_TOLERANCE_MM:
        direction_string = "Clockwise"
    elif total_upper_length > total_lower_length + DIRECTION_TOLERANCE_MM:
        direction_string = "Anticlockwise"
    
    return direction_string, upper_walls, lower_walls

### NEW FUNCTION ###
def categorize_walls_by_direction(walls, angle_tolerance_deg):
    """
    Categorizes walls into four directions relative to the robot and finds the closest in each.

    The robot is at (0,0) facing towards the positive Y axis (up on the screen).
    - Front (0 deg): Horizontal wall in the +Y direction.
    - Right (90 deg): Vertical wall in the +X direction.
    - Back (180 deg): Horizontal wall in the -Y direction.
    - Left (270 deg): Vertical wall in the -X direction.

    Args:
        walls (list): A list of wall segments, where each segment is ((x1, y1), (x2, y2)).
        angle_tolerance_deg (int): Tolerance in degrees for a wall to be horizontal/vertical.

    Returns:
        list: A 4-element list with distances for [front, right, back, left].
              -1 indicates no wall was detected in that direction.
    """
    # Indices for the directions array
    FRONT, RIGHT, BACK, LEFT = 0, 1, 2, 3
    distances = [-1.0] * 4

    for p1, p2 in walls:
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        
        # Get angle and normalize to [0, 180) to represent the line's orientation
        wall_angle_deg = math.degrees(math.atan2(dy, dx))
        normalized_angle_deg = (wall_angle_deg + 360) % 180

        # Calculate distance from the robot (0,0) to the wall segment
        dist = math.hypot(*find_closest_point_on_segment((0, 0), p1, p2))

        # Check for Horizontal-ish walls (potential Front or Back)
        if normalized_angle_deg < angle_tolerance_deg or normalized_angle_deg > (180 - angle_tolerance_deg):
            mid_y = (p1[1] + p2[1]) / 2
            index = FRONT if mid_y > 0 else BACK
            if distances[index] == -1 or dist < distances[index]:
                distances[index] = dist

        # Check for Vertical-ish walls (potential Left or Right)
        elif (90 - angle_tolerance_deg) < normalized_angle_deg < (90 + angle_tolerance_deg):
            mid_x = (p1[0] + p2[0]) / 2
            index = RIGHT if mid_x > 0 else LEFT
            if distances[index] == -1 or dist < distances[index]:
                distances[index] = dist
                
    return distances

def draw_highlight_blocks(screen, surface, walls, color, thickness):
    """Draws thick, translucent polygons over the given walls."""
    if not walls: return
    center_x, center_y = screen.get_width() // 2, screen.get_height() // 2
    half_thick = thickness / 2
    for p1_mm, p2_mm in walls:
        p1_screen = (center_x + p1_mm[0] / SCALE_FACTOR, center_y - p1_mm[1] / SCALE_FACTOR)
        p2_screen = (center_x + p2_mm[0] / SCALE_FACTOR, center_y - p2_mm[1] / SCALE_FACTOR)
        dx, dy = p2_screen[0] - p1_screen[0], p2_screen[1] - p1_screen[1]
        length = math.hypot(dx, dy)
        if length == 0: continue
        norm_x, norm_y = -dy / length, dx / length
        poly_points = [
            (p1_screen[0] + half_thick * norm_x, p1_screen[1] + half_thick * norm_y),
            (p2_screen[0] + half_thick * norm_x, p2_screen[1] + half_thick * norm_y),
            (p2_screen[0] - half_thick * norm_x, p2_screen[1] - half_thick * norm_y),
            (p1_screen[0] - half_thick * norm_x, p1_screen[1] - half_thick * norm_y)
        ]
        pygame.draw.polygon(surface, color, poly_points)

### MODIFIED FUNCTION ###
def draw_robot_direction_arrow(screen):
    # Arrow now points UP to represent the robot's forward direction (+Y axis).
    fixed_angle_rad = 0.0 # 0 radians points along the -Y screen axis (UP)
    center_x, center_y = screen.get_width() // 2, screen.get_height() // 2
    end_x = center_x + ARROW_LENGTH_PX * math.sin(fixed_angle_rad)
    end_y = center_y - ARROW_LENGTH_PX * math.cos(fixed_angle_rad)
    pygame.draw.line(screen, ARROW_COLOR, (center_x, center_y), (end_x, end_y), 5)
    head_angle1 = fixed_angle_rad + math.radians(ARROW_HEAD_ANGLE_DEG)
    head_angle2 = fixed_angle_rad - math.radians(ARROW_HEAD_ANGLE_DEG)
    head1_end_x = end_x + ARROW_HEAD_LENGTH_PX * math.sin(head_angle1)
    head1_end_y = end_y - ARROW_HEAD_LENGTH_PX * math.cos(head_angle1)
    pygame.draw.line(screen, ARROW_COLOR, (end_x, end_y), (head1_end_x, head1_end_y), 5)
    head2_end_x = end_x + ARROW_HEAD_LENGTH_PX * math.sin(head_angle2)
    head2_end_y = end_y - ARROW_HEAD_LENGTH_PX * math.cos(head_angle2)
    pygame.draw.line(screen, ARROW_COLOR, (end_x, end_y), (head2_end_x, head2_end_y), 5)

# --- Main Pygame Loop ---
def main():
    """Starts threads and runs the Pygame loop with RANSAC processing."""
    reader = threading.Thread(target=lidar_reader_thread)
    reader.daemon = True
    reader.start()

    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    highlight_surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
    pygame.display.set_caption("LiDAR Scan with Wall Categorization")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 24)
    
    center_x, center_y = SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2

    running = True
    while running and reader.is_alive():
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False

        # --- Data Processing Pipeline ---
        points_to_process = list(lidar_points_polar)
        cartesian_points = []
        for angle_rad, distance_mm in points_to_process:
            if 0 < distance_mm <= LIDAR_RANGE_MM:
                # Robot forward is +Y, right is +X
                x = distance_mm * math.sin(angle_rad)
                y = distance_mm * math.cos(angle_rad)
                cartesian_points.append((x, y))

        filtered_points = filter_isolated_points(cartesian_points, ISOLATION_MIN_NEIGHBORS, ISOLATION_DISTANCE_MM)
        estimated_walls = run_ransac(filtered_points)
        
        direction, upper_walls, lower_walls = determine_direction(estimated_walls)
        
        ### NEW ###
        # Categorize walls and get distances for the four cardinal directions
        wall_distances = categorize_walls_by_direction(estimated_walls, WALL_ANGLE_TOLERANCE_DEG)
        
        # --- Drawing ---
        screen.fill(BACKGROUND_COLOR)
        highlight_surface.fill((0, 0, 0, 0))
        decision_walls = upper_walls + lower_walls
        draw_highlight_blocks(screen, highlight_surface, decision_walls, HIGHLIGHT_COLOR, HIGHLIGHT_THICKNESS_PX)

        # Draw grid lines and labels
        for r in range(1000, int(MAX_VIZ_DISTANCE_MM) + 1, 1000):
            radius_px = int(r / SCALE_FACTOR)
            pygame.draw.circle(screen, (40, 40, 40), (center_x, center_y), radius_px, 1)
            label = font.render(f'{r/1000:.1f}m', True, (100, 100, 100))
            screen.blit(label, (center_x + 5, center_y - radius_px - 15))

        # Draw filtered points
        for x_mm, y_mm in filtered_points:
            screen_x = center_x + int(x_mm / SCALE_FACTOR)
            screen_y = center_y - int(y_mm / SCALE_FACTOR) # Invert Y for screen coordinates
            pygame.draw.circle(screen, FILTERED_POINT_COLOR, (screen_x, screen_y), 2)
        
        # Draw detected walls on top of the highlights
        for wall in estimated_walls:
            p1_mm, p2_mm = wall
            p1_screen = (center_x + int(p1_mm[0] / SCALE_FACTOR), center_y - int(p1_mm[1] / SCALE_FACTOR))
            p2_screen = (center_x + int(p2_mm[0] / SCALE_FACTOR), center_y - int(p2_mm[1] / SCALE_FACTOR))
            pygame.draw.line(screen, WALL_COLOR, p1_screen, p2_screen, 3)

        screen.blit(highlight_surface, (0, 0))

        # Draw distance lines to each wall
        for wall in estimated_walls:
            p1_mm, p2_mm = wall
            closest_point_mm = find_closest_point_on_segment((0, 0), p1_mm, p2_mm)
            distance_mm = math.hypot(closest_point_mm[0], closest_point_mm[1])
            closest_point_screen = (center_x + int(closest_point_mm[0] / SCALE_FACTOR), center_y - int(closest_point_mm[1] / SCALE_FACTOR))
            pygame.draw.line(screen, DISTANCE_LINE_COLOR, (center_x, center_y), closest_point_screen, 2)
            dist_text = f"{distance_mm:.1f} mm"
            text_surface = font.render(dist_text, True, DISTANCE_TEXT_COLOR)
            text_rect = text_surface.get_rect(center=closest_point_screen)
            text_rect.x -= 30
            pygame.draw.rect(screen, BACKGROUND_COLOR, text_rect.inflate(4, 4))
            screen.blit(text_surface, text_rect)
        
        draw_robot_direction_arrow(screen)

        # Display status text
        screen.blit(font.render(f'FPS: {int(clock.get_fps())}', True, (255, 255, 0)), (10, 10))
        screen.blit(font.render(f'Filtered Points: {len(filtered_points)}/{len(cartesian_points)}', True, (255, 255, 0)), (10, 30))
        screen.blit(font.render(f'Detected Walls: {len(estimated_walls)}', True, (255, 255, 0)), (10, 50))
        screen.blit(font.render(f'Direction Hint: {direction}', True, (255, 255, 0)), (10, 70))
        
        ### NEW ###
        # Display categorized wall distances
        pygame.draw.line(screen, (80, 80, 80), (10, 95), (200, 95), 1) # Separator
        wall_dist_text_f = f"Front: {wall_distances[0]:.0f} mm" if wall_distances[0] != -1 else "Front: N/A"
        wall_dist_text_r = f"Right: {wall_distances[1]:.0f} mm" if wall_distances[1] != -1 else "Right: N/A"
        wall_dist_text_b = f"Back:  {wall_distances[2]:.0f} mm" if wall_distances[2] != -1 else "Back:  N/A"
        wall_dist_text_l = f"Left:  {wall_distances[3]:.0f} mm" if wall_distances[3] != -1 else "Left:  N/A"
        screen.blit(font.render(wall_dist_text_f, True, (100, 255, 100)), (10, 100))
        screen.blit(font.render(wall_dist_text_r, True, (100, 255, 100)), (10, 120))
        screen.blit(font.render(wall_dist_text_b, True, (100, 255, 100)), (10, 140))
        screen.blit(font.render(wall_dist_text_l, True, (100, 255, 100)), (10, 160))

        pygame.display.flip()
        clock.tick(30)

    # --- Cleanup ---
    print("Main loop finished. Stopping reader thread...")
    stop_event.set()
    reader.join()
    pygame.quit()
    print("Program finished cleanly.")

if __name__ == '__main__':
    main()