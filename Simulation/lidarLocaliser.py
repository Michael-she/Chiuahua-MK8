import pygame
import numpy as np
import random
import math

# --- Configuration ---
# Screen
SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 700

# Colors
COLOR_BACKGROUND = (10, 10, 10)
COLOR_WALL = (200, 200, 200)
COLOR_ROBOT = (255, 0, 0)
COLOR_LIDAR_POINT = (0, 180, 0)
COLOR_ESTIMATED_WALL = (100, 255, 100)
COLOR_INLIER_POINTS = (0, 100, 255)
COLOR_TEXT = (255, 255, 255)
COLOR_PERPENDICULAR = (0, 170, 255)
COLOR_NON_PERPENDICULAR = (200, 160, 255)
COLOR_OBSTACLE_RED = (255, 50, 50)
COLOR_OBSTACLE_GREEN = (50, 255, 50)
COLOR_ESTIMATED_POS = (255, 0, 255) # Pink for estimated positions/lines

# Robot properties
ROBOT_SIZE = 15

# LiDAR properties
LIDAR_RANGE = 3000
LIDAR_RAYS = 360
LIDAR_INACCURACY = 8.0

# RANSAC & Clustering Parameters
RANSAC_ITERATIONS = 30
RANSAC_THRESHOLD_DISTANCE = 14.0
RANSAC_MIN_INLIERS = 10
MAX_POINT_DISTANCE = 60.0
PERPENDICULAR_ANGLE_THRESHOLD = 5.0
MIN_WALL_LENGTH = 40.0

# --- Localization Parameters & Known Map ---  <<< MODIFIED SECTION >>>
PARALLEL_THRESHOLD_DOT = 0.998
DISTANCE_MATCH_THRESHOLD = 30.0 # How close a measured distance must be to a known distance

# Known distances between parallel walls
KNOWN_DIST_INNER_OUTER = 200.0
KNOWN_DIST_OUTER_OUTER = 600.0

# Robot orientation
ORIENTATION_ACCURACY = 5.0  # Known to within 5 degrees

# Known absolute coordinates of the course walls
COURSE_TOP_LEFT = 50
KNOWN_VERTICAL_WALLS_X = [
    0 + COURSE_TOP_LEFT, 200 + COURSE_TOP_LEFT,
    400 + COURSE_TOP_LEFT, 600 + COURSE_TOP_LEFT
]
KNOWN_HORIZONTAL_WALLS_Y = [
    0 + COURSE_TOP_LEFT, 200 + COURSE_TOP_LEFT,
    400 + COURSE_TOP_LEFT, 600 + COURSE_TOP_LEFT
]

# Track bounds for confining position estimates
TRACK_BOUNDS = {
    'outer': {'x_min': 50, 'x_max': 650, 'y_min': 50, 'y_max': 650},
    'inner': {'x_min': 250, 'x_max': 450, 'y_min': 250, 'y_max': 450}
}
# --- End Modified Section ---

# Course & Obstacle Definition
OBSTACLE_SIZE = (10, 10)
COURSE_WALLS = [
    (50, 50, 650, 50), (650, 50, 650, 650), (650, 650, 50, 650), (50, 650, 50, 50),
    (250, 250, 450, 250), (450, 250, 450, 450), (450, 450, 250, 450), (250, 450, 250, 250),
]

def generate_obstacles():
    obstacle_slots = {
        "top": [(250, 130), (350, 130), (450, 130), (250, 170), (350, 170), (450, 170)],
        "bottom": [(250, 575), (350, 575), (450, 575), (250, 615), (350, 615), (450, 615)],
        "left": [(130, 250), (130, 375), (130, 500), (170, 250), (170, 375), (170, 500)],
        "right": [(575, 250), (575, 375), (575, 500), (615, 250), (615, 375), (615, 500)],
    }
    generated_obstacles, obstacle_walls = [], []
    for segment, slots in obstacle_slots.items():
        num_to_spawn = random.choice([0, 1, 1, 2])
        chosen_slots = []
        if num_to_spawn == 1: chosen_slots.append(random.choice(slots))
        elif num_to_spawn == 2:
            pair = random.choice([(0, 2), (3, 5)])
            chosen_slots.extend([slots[pair[0]], slots[pair[1]]])
        for slot_center in chosen_slots:
            rect = pygame.Rect((0, 0), OBSTACLE_SIZE); rect.center = slot_center
            color = random.choice([COLOR_OBSTACLE_RED, COLOR_OBSTACLE_GREEN])
            generated_obstacles.append({'rect': rect, 'color': color})
            tl, tr, bl, br = rect.topleft, rect.topright, rect.bottomleft, rect.bottomright
            obstacle_walls.extend([(tl[0], tl[1], tr[0], tr[1]), (tr[0], tr[1], br[0], br[1]), (br[0], br[1], bl[0], bl[1]), (bl[0], bl[1], tl[0], tl[1])])
    return generated_obstacles, obstacle_walls

class Robot:
    def __init__(self, x, y):
        self.x, self.y = x, y
        self.orientation = 0.0  # Robot orientation in degrees
        self.lidar_points, self.estimated_walls, self.inlier_points_for_viz = [], [], []
        self.position_estimates = [] # <<< NEW: Will store points OR lines
        self.is_in_corner = False  # Track if robot is in corner section

    def set_pos(self, x, y): self.x, self.y = x, y
    
    def set_orientation(self, angle): 
        self.orientation = angle % 360

    def simulate_lidar(self, walls):
        # ... (no changes in this method)
        self.lidar_points = []
        for i in range(LIDAR_RAYS):
            ray_angle_rad = math.radians((360 / LIDAR_RAYS) * i)
            end_x, end_y = self.x + LIDAR_RANGE * math.cos(ray_angle_rad), self.y + LIDAR_RANGE * math.sin(ray_angle_rad)
            closest_dist, hit_point = LIDAR_RANGE, None
            for wall in walls:
                p, d = line_intersection((self.x, self.y), (end_x, end_y), (wall[0], wall[1]), (wall[2], wall[3]))
                if p and d < closest_dist: closest_dist, hit_point = d, p
            if hit_point:
                dist_with_error = closest_dist + random.uniform(-LIDAR_INACCURACY, LIDAR_INACCURACY)
                self.lidar_points.append((self.x + dist_with_error * math.cos(ray_angle_rad), self.y + dist_with_error * math.sin(ray_angle_rad)))

    def estimate_walls(self):
        # ... (no changes in this method)
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
                            p_start, p_end = final_wall_segment[0], final_wall_segment[1]
                            length = math.hypot(p_end[0] - p_start[0], p_end[1] - p_start[1])
                            if length >= MIN_WALL_LENGTH:
                                self.estimated_walls.append(final_wall_segment)
                                self.inlier_points_for_viz.extend(cluster)
                        for point in cluster:
                            points_to_remove.add(point)
                if not points_to_remove: break
                remaining_points = [p for p in remaining_points if p not in points_to_remove]
            else:
                break

    # <<< METHOD COMPLETELY REWRITTEN >>>
    def estimate_position(self):
        self.position_estimates = []
        robot_pos = (self.x, self.y)

        # 1. Find and categorize all detected parallel pairs
        h_pairs, v_pairs = [], []
        all_pairs = find_parallel_pairs(self.estimated_walls)
        for pair in all_pairs:
            if pair['orientation'] == 'h': h_pairs.append(pair)
            else: v_pairs.append(pair)

        # 2. Determine if robot is in corner section based on wall pairs
        self.is_in_corner = self._detect_corner_section(h_pairs, v_pairs)

        # 3. Get X and Y constraints from these pairs
        x_constraints = self._get_constraints_from_pairs(v_pairs, 'v', robot_pos)
        y_constraints = self._get_constraints_from_pairs(h_pairs, 'h', robot_pos)

        # 4. Apply bounds checking to constraints
        x_constraints = self._apply_bounds_filtering(x_constraints, 'x')
        y_constraints = self._apply_bounds_filtering(y_constraints, 'y')

        # 5. Use robot orientation to further filter estimates
        if x_constraints: x_constraints = self._apply_orientation_filtering(x_constraints, 'x')
        if y_constraints: y_constraints = self._apply_orientation_filtering(y_constraints, 'y')

        # 6. Synthesize results into points (if both axes constrained) or lines (if one axis)
        if x_constraints and y_constraints:
            # We have both X and Y info, so we can find intersection points
            for x in x_constraints:
                for y in y_constraints:
                    if self._is_position_valid(x, y):
                        self.position_estimates.append((x, y))
        elif x_constraints:
            # Only X is constrained, so draw vertical lines of possibility
            for x in x_constraints:
                y_min, y_max = self._get_valid_y_range(x)
                if y_max > y_min:
                    self.position_estimates.append(((x, y_min), (x, y_max)))
        elif y_constraints:
            # Only Y is constrained, so draw horizontal lines of possibility
            for y in y_constraints:
                x_min, x_max = self._get_valid_x_range(y)
                if x_max > x_min:
                    self.position_estimates.append(((x_min, y), (x_max, y)))

    def _detect_corner_section(self, h_pairs, v_pairs):
        """Detect if robot is in corner section based on ~600px parallel wall pairs."""
        corner_pairs_count = 0
        
        for pair in h_pairs + v_pairs:
            measured_dist = distance_from_point_to_line(pair['center1'], pair['p1_2'], pair['p2_2'])
            if abs(measured_dist - KNOWN_DIST_OUTER_OUTER) < DISTANCE_MATCH_THRESHOLD:
                corner_pairs_count += 1
        
        return corner_pairs_count >= 2  # If we see two ~600px pairs, we're in corner

    def _apply_bounds_filtering(self, constraints, axis):
        """Filter constraints to valid track bounds."""
        if not constraints:
            return constraints
            
        filtered = []
        bounds = TRACK_BOUNDS['outer']
        
        for constraint in constraints:
            if axis == 'x':
                if bounds['x_min'] <= constraint <= bounds['x_max']:
                    filtered.append(constraint)
            else:  # axis == 'y'
                if bounds['y_min'] <= constraint <= bounds['y_max']:
                    filtered.append(constraint)
        
        return filtered

    def _apply_orientation_filtering(self, constraints, axis):
        """Use robot orientation to filter unlikely positions."""
        if not constraints or len(constraints) <= 1:
            return constraints
            
        # Use robot orientation to prefer certain position estimates
        # This is a simplified implementation - can be enhanced based on movement patterns
        robot_pos = (self.x, self.y)
        filtered = []
        
        for constraint in constraints:
            # Calculate expected position based on orientation and current robot position
            # For now, we accept all constraints but this could be enhanced to:
            # 1. Filter based on movement direction consistency
            # 2. Use orientation to resolve ambiguities between symmetric positions
            # 3. Apply probabilistic weighting based on orientation confidence
            filtered.append(constraint)
            
        return filtered

    def _is_position_valid(self, x, y):
        """Check if a position is within valid track bounds."""
        outer = TRACK_BOUNDS['outer']
        inner = TRACK_BOUNDS['inner']
        
        # Must be within outer bounds
        if not (outer['x_min'] <= x <= outer['x_max'] and outer['y_min'] <= y <= outer['y_max']):
            return False
            
        # If in corner section, cannot be in inner area
        if self.is_in_corner:
            if inner['x_min'] <= x <= inner['x_max'] and inner['y_min'] <= y <= inner['y_max']:
                return False
                
        return True

    def _get_valid_y_range(self, x):
        """Get valid Y range for a given X coordinate."""
        outer = TRACK_BOUNDS['outer']
        inner = TRACK_BOUNDS['inner']
        
        y_min, y_max = outer['y_min'], outer['y_max']
        
        # If in corner section and X is in inner range, adjust Y bounds
        if self.is_in_corner and inner['x_min'] <= x <= inner['x_max']:
            # Split into two ranges: top and bottom of inner area
            if y_min < inner['y_min']:
                y_max = inner['y_min']  # Use top section only for simplicity
            
        return y_min, y_max

    def _get_valid_x_range(self, y):
        """Get valid X range for a given Y coordinate."""
        outer = TRACK_BOUNDS['outer']
        inner = TRACK_BOUNDS['inner']
        
        x_min, x_max = outer['x_min'], outer['x_max']
        
        # If in corner section and Y is in inner range, adjust X bounds
        if self.is_in_corner and inner['y_min'] <= y <= inner['y_max']:
            # Split into two ranges: left and right of inner area
            if x_min < inner['x_min']:
                x_max = inner['x_min']  # Use left section only for simplicity
                
        return x_min, x_max

    def _get_constraints_from_pairs(self, pairs, orientation, robot_pos):
        """Helper to calculate possible coordinate values (X or Y) from parallel wall pairs."""
        constraints = set()
        known_distances = [KNOWN_DIST_INNER_OUTER, KNOWN_DIST_OUTER_OUTER]
        
        for pair in pairs:
            measured_dist = distance_from_point_to_line(pair['center1'], pair['p1_2'], pair['p2_2'])
            
            for known_dist in known_distances:
                if abs(measured_dist - known_dist) < DISTANCE_MATCH_THRESHOLD:
                    # Match found! Now calculate the robot's position relative to this hypothesis.
                    # This is the key: we get the robot's distance to one of the *detected* walls.
                    dist_robot_to_wall1 = distance_from_point_to_line(robot_pos, pair['p1_1'], pair['p2_1'])
                    
                    known_coords = KNOWN_VERTICAL_WALLS_X if orientation == 'v' else KNOWN_HORIZONTAL_WALLS_Y
                    
                    for coord in known_coords:
                        # The robot could be on either side of the known wall.
                        constraints.add(coord + dist_robot_to_wall1)
                        constraints.add(coord - dist_robot_to_wall1)
        return list(constraints)

    def cluster_inliers_by_distance(self, inliers):
        # ... (no changes in this method)
        if not inliers: return []
        data = np.array(inliers)
        mean = np.mean(data, axis=0)
        direction_vector = np.linalg.eigh(np.cov(data - mean, rowvar=False))[1][:, -1]
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
        # ... (no changes in this method)
        if len(points) < 2: return None
        data = np.array(points)
        mean = np.mean(data, axis=0)
        eigenvalues, eigenvectors = np.linalg.eigh(np.cov(data.T))
        direction_vector = eigenvectors[:, -1]
        projections = np.dot(data - mean, direction_vector)
        t_min, t_max = np.min(projections), np.max(projections)
        line_start = mean + t_min * direction_vector
        line_end = mean + t_max * direction_vector
        return (line_start.tolist(), line_end.tolist())

# Helper Functions
def get_wall_vector_and_orientation(wall):
    p1, p2 = np.array(wall[0]), np.array(wall[1])
    vec = p2 - p1
    orientation = 'v' if abs(vec[0]) < abs(vec[1]) else 'h'
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else np.array([0, 0]), orientation

def find_parallel_pairs(walls):
    pairs = []
    for i in range(len(walls)):
        for j in range(i + 1, len(walls)):
            wall1, wall2 = walls[i], walls[j]
            vec1, orient1 = get_wall_vector_and_orientation(wall1)
            vec2, orient2 = get_wall_vector_and_orientation(wall2)
            # Check for same orientation and high dot product (parallelism)
            if orient1 == orient2 and abs(np.dot(vec1, vec2)) > PARALLEL_THRESHOLD_DOT:
                p1_1, p2_1 = wall1[0], wall1[1]
                p1_2, p2_2 = wall2[0], wall2[1]
                pairs.append({
                    'p1_1': p1_1, 'p2_1': p2_1, 'center1': ((p1_1[0]+p2_1[0])/2, (p1_1[1]+p2_1[1])/2),
                    'p1_2': p1_2, 'p2_2': p2_2, 'center2': ((p1_2[0]+p2_2[0])/2, (p1_2[1]+p2_2[1])/2),
                    'orientation': orient1
                })
    return pairs

def distance_from_point_to_line(point, line_p1, line_p2):
    x0, y0 = point; x1, y1 = line_p1; x2, y2 = line_p2
    num = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)
    den = math.sqrt((y2 - y1)**2 + (x2 - x1)**2)
    return num / den if den != 0 else 0

def line_intersection(p1, p2, p3, p4):
    x1, y1 = p1; x2, y2 = p2; x3, y3 = p3; x4, y4 = p4
    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if den == 0: return None, float('inf')
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
    u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / den
    if 0 < t < 1 and 0 < u < 1:
        px, py = x1 + t * (x2 - x1), y1 + t * (y2 - y1)
        return (px, py), math.hypot(px - x1, py - y1)
    return None, float('inf')

def find_closest_point_on_segment(p, a, b):
    p_np, a_np, b_np = np.array(p), np.array(a), np.array(b)
    ab, ap = b_np - a_np, p_np - a_np
    ab_len_sq = np.dot(ab, ab)
    if ab_len_sq == 0: return a
    t = np.dot(ap, ab) / ab_len_sq
    return tuple(a_np + max(0, min(1, t)) * ab)

def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Localization with Lines of Possibility")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 28)
    small_font = pygame.font.Font(None, 22)
    robot = Robot(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2)

    obstacles_for_drawing, obstacle_walls = generate_obstacles()
    all_walls_for_lidar = COURSE_WALLS + obstacle_walls

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            if event.type == pygame.MOUSEMOTION: robot.set_pos(*event.pos)
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    obstacles_for_drawing, obstacle_walls = generate_obstacles()
                    all_walls_for_lidar = COURSE_WALLS + obstacle_walls
                elif event.key == pygame.K_q:
                    robot.set_orientation(robot.orientation - 10)  # Rotate left
                elif event.key == pygame.K_e:
                    robot.set_orientation(robot.orientation + 10)  # Rotate right

        robot.simulate_lidar(all_walls_for_lidar)
        robot.estimate_walls()
        robot.estimate_position() # <<< Call the new estimation method

        screen.fill(COLOR_BACKGROUND)

        # --- Draw track bounds (optional visualization) ---
        outer = TRACK_BOUNDS['outer']
        inner = TRACK_BOUNDS['inner']
        # Draw outer bounds as a subtle rectangle
        pygame.draw.rect(screen, (30, 30, 30), 
                        (outer['x_min'], outer['y_min'], 
                         outer['x_max'] - outer['x_min'], outer['y_max'] - outer['y_min']), 1)
        # Draw inner bounds
        pygame.draw.rect(screen, (30, 30, 30), 
                        (inner['x_min'], inner['y_min'], 
                         inner['x_max'] - inner['x_min'], inner['y_max'] - inner['y_min']), 1)

        # --- Drawing ---
        for wall in COURSE_WALLS: pygame.draw.line(screen, COLOR_WALL, (wall[0], wall[1]), (wall[2], wall[3]), 2)
        for obs in obstacles_for_drawing:
            pygame.draw.rect(screen, obs['color'], obs['rect'])
            pygame.draw.rect(screen, COLOR_WALL, obs['rect'], 1)

        for p in robot.lidar_points: pygame.draw.circle(screen, (50,50,50), p, 2)
        for p in robot.inlier_points_for_viz: pygame.draw.circle(screen, COLOR_INLIER_POINTS, p, 3)
        for wall in robot.estimated_walls: pygame.draw.line(screen, COLOR_ESTIMATED_WALL, wall[0], wall[1], 5)

        # <<< NEW: Draw possible positions (points) OR lines of possibility >>>
        for estimate in robot.position_estimates:
            # Check if the estimate is a line (tuple of two points) or a single point
            if isinstance(estimate[0], tuple):
                pygame.draw.line(screen, COLOR_ESTIMATED_POS, estimate[0], estimate[1], 2)
            else:
                pygame.draw.circle(screen, COLOR_ESTIMATED_POS, estimate, 8)
                pygame.draw.circle(screen, COLOR_TEXT, estimate, 8, 1)

        robot_pos = (robot.x, robot.y)
        for wall_segment in robot.estimated_walls:
            # ... (no changes to this drawing section)
            p_start, p_end = wall_segment[0], wall_segment[1]
            closest_point = find_closest_point_on_segment(robot_pos, p_start, p_end)
            distance = math.hypot(robot_pos[0] - closest_point[0], robot_pos[1] - closest_point[1])
            vec_robot_to_point, vec_wall = np.array(closest_point) - np.array(robot_pos), np.array(p_end) - np.array(p_start)
            angle_deg, mag_robot_vec, mag_wall_vec = 90, np.linalg.norm(vec_robot_to_point), np.linalg.norm(vec_wall)
            if mag_robot_vec > 0 and mag_wall_vec > 0:
                cos_theta = np.clip(np.dot(vec_robot_to_point, vec_wall) / (mag_robot_vec * mag_wall_vec), -1.0, 1.0)
                angle_deg = abs(90 - math.degrees(math.acos(cos_theta)))
            line_color = COLOR_PERPENDICULAR if angle_deg <= PERPENDICULAR_ANGLE_THRESHOLD else COLOR_NON_PERPENDICULAR
            pygame.draw.line(screen, line_color, robot_pos, closest_point, 2)
            dist_text_str = f"{distance:.0f}"
            text_surface = small_font.render(dist_text_str, True, COLOR_TEXT)
            mid_point = ((robot_pos[0] + closest_point[0]) / 2, (robot_pos[1] + closest_point[1]) / 2)
            text_rect = text_surface.get_rect(center=mid_point)
            pygame.draw.rect(screen, COLOR_BACKGROUND, text_rect.inflate(6, 4))
            screen.blit(text_surface, text_rect)

        pygame.draw.circle(screen, COLOR_ROBOT, robot_pos, ROBOT_SIZE)
        pygame.draw.circle(screen, (255,255,255), robot_pos, ROBOT_SIZE, 1)
        
        # Draw orientation indicator
        orientation_rad = math.radians(robot.orientation)
        end_x = robot.x + (ROBOT_SIZE - 3) * math.cos(orientation_rad)
        end_y = robot.y + (ROBOT_SIZE - 3) * math.sin(orientation_rad)
        pygame.draw.line(screen, (255, 255, 255), robot_pos, (end_x, end_y), 3)

        instruction_text = font.render("Mouse: move robot, Q/E: rotate, R: new obstacles", True, COLOR_TEXT)
        screen.blit(instruction_text, (10, 10))
        
        orientation_text = font.render(f"Orientation: {robot.orientation:.1f}°", True, COLOR_TEXT)
        screen.blit(orientation_text, (10, 40))
        
        corner_text = font.render(f"In corner: {'Yes' if robot.is_in_corner else 'No'}", True, COLOR_TEXT)
        screen.blit(corner_text, (10, 70))
        
        pos_text = font.render(f"Position estimates: {len(robot.position_estimates)}", True, COLOR_ESTIMATED_POS)
        screen.blit(pos_text, (10, SCREEN_HEIGHT - 35))

        pygame.display.flip()
        clock.tick(60)
    pygame.quit()

if __name__ == '__main__':
    main()