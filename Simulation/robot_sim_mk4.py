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
RANSAC_ITERATIONS = 30
RANSAC_THRESHOLD_DISTANCE = 14.0
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
OBSTACLE_SIZE = (10, 10)
COURSE_WALLS = [
    (0+COURSE_TOP_LEFT, 0+COURSE_TOP_LEFT, 600+COURSE_TOP_LEFT, 0+COURSE_TOP_LEFT),
    (600+COURSE_TOP_LEFT, 0+COURSE_TOP_LEFT, 600+COURSE_TOP_LEFT, 600+COURSE_TOP_LEFT),
    (600+COURSE_TOP_LEFT, 600+COURSE_TOP_LEFT, 0+COURSE_TOP_LEFT, 600+COURSE_TOP_LEFT),
    (0+COURSE_TOP_LEFT, 600+COURSE_TOP_LEFT, 0+COURSE_TOP_LEFT, 0+COURSE_TOP_LEFT),
    (200+COURSE_TOP_LEFT, 200+COURSE_TOP_LEFT, 400+COURSE_TOP_LEFT, 200+COURSE_TOP_LEFT),
    (400+COURSE_TOP_LEFT, 200+COURSE_TOP_LEFT, 400+COURSE_TOP_LEFT, 400+COURSE_TOP_LEFT),
    (400+COURSE_TOP_LEFT, 400+COURSE_TOP_LEFT, 200+COURSE_TOP_LEFT, 400+COURSE_TOP_LEFT),
    (200+COURSE_TOP_LEFT, 400+COURSE_TOP_LEFT, 200+COURSE_TOP_LEFT, 200+COURSE_TOP_LEFT),
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
    for wall_start, wall_end in walls:
        closest_point = find_closest_point_on_segment(center, wall_start, wall_end)
        dist_sq = (center[0] - closest_point[0])**2 + (center[1] - closest_point[1])**2
        if dist_sq < radius_sq: return True
    return False
def generate_obstacles():
    obstacle_slots = {k: [(v[0], v[1]) for v in vs] for k, vs in {
        "top": [(200+COURSE_TOP_LEFT, 80+COURSE_TOP_LEFT), (300+COURSE_TOP_LEFT, 80+COURSE_TOP_LEFT), (400+COURSE_TOP_LEFT, 80+COURSE_TOP_LEFT), (200+COURSE_TOP_LEFT, 120+COURSE_TOP_LEFT), (300+COURSE_TOP_LEFT, 120+COURSE_TOP_LEFT), (400+COURSE_TOP_LEFT, 120+COURSE_TOP_LEFT)],
        "bottom": [(200+COURSE_TOP_LEFT, 480+COURSE_TOP_LEFT), (300+COURSE_TOP_LEFT, 480+COURSE_TOP_LEFT), (400+COURSE_TOP_LEFT, 480+COURSE_TOP_LEFT), (200+COURSE_TOP_LEFT, 520+COURSE_TOP_LEFT), (300+COURSE_TOP_LEFT, 520+COURSE_TOP_LEFT), (400+COURSE_TOP_LEFT, 520+COURSE_TOP_LEFT)],
        "left": [(80+COURSE_TOP_LEFT, 200+COURSE_TOP_LEFT), (80+COURSE_TOP_LEFT, 325+COURSE_TOP_LEFT), (80+COURSE_TOP_LEFT, 450+COURSE_TOP_LEFT), (120+COURSE_TOP_LEFT, 200+COURSE_TOP_LEFT), (120+COURSE_TOP_LEFT, 325+COURSE_TOP_LEFT), (120+COURSE_TOP_LEFT, 450+COURSE_TOP_LEFT)],
        "right": [(480+COURSE_TOP_LEFT, 200+COURSE_TOP_LEFT), (480+COURSE_TOP_LEFT, 325+COURSE_TOP_LEFT), (480+COURSE_TOP_LEFT, 450+COURSE_TOP_LEFT), (520+COURSE_TOP_LEFT, 200+COURSE_TOP_LEFT), (520+COURSE_TOP_LEFT, 325+COURSE_TOP_LEFT), (520+COURSE_TOP_LEFT, 450+COURSE_TOP_LEFT)],
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

class Robot:
    def __init__(self, x, y):
        self.x, self.y = x, y
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

    def set_path(self, path):
        if path:
            self.path, self.current_waypoint_index, self.mode = path, 0, 'AUTO'
            self.replan_needed = False
        else:
            self.mode, self.path, self.speed_level = 'MANUAL', [], 0
            self.replan_needed = False
        self.consecutive_replan_count = 0
        # --- MODIFICATION: sharp_turn_locations is NOT cleared here anymore.
        # It's cleared on successful progress or manual reset.
            
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
                    print(f"Stuck after {self.consecutive_replan_count} attempts! Reversing...")
                    self.mode = 'REVERSING'
                    self.reverse_timer_start = pygame.time.get_ticks()
                    self.consecutive_replan_count = 0
                    self.path = [] 
                    self.sharp_turn_locations.clear()
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

    def update(self, keys_pressed):
        if self.mode == 'MANUAL':
            steer_input = 0
            if keys_pressed[pygame.K_a]: steer_input = 1
            if keys_pressed[pygame.K_d]: steer_input = -1
            if any(keys_pressed): 
                self.set_path([])
                self.sharp_turn_locations.clear()
        
            if steer_input != 0:
                self.steering_angle -= steer_input * ROBOT_STEERING_RATE
                self.steering_angle = np.clip(self.steering_angle, -ROBOT_MAX_STEERING_ANGLE, ROBOT_MAX_STEERING_ANGLE)
            else:
                self.steering_angle *= ROBOT_STEERING_RETURN_RATE
        
        elif self.mode == 'AUTO':
            self.follow_path()
            
        elif self.mode == 'REVERSING':
            self.speed_level = -1
            self.steering_angle = 0
            if pygame.time.get_ticks() - self.reverse_timer_start > REVERSE_DURATION_MS:
                self.speed_level = 0
                self.mode = 'AUTO'
                self.replan_needed = True

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
    def __init__(self): self.potential_obstacles, self.confirmed_obstacles = [], []
    def reset(self): self.potential_obstacles.clear(); self.confirmed_obstacles.clear()
    def update(self, current_frame_clusters, estimated_walls):
        invalidated_confirmed_indices = [i for i, obs in enumerate(self.confirmed_obstacles) if check_obstacle_wall_collision(obs, estimated_walls)]
        for i in sorted(invalidated_confirmed_indices, reverse=True): del self.confirmed_obstacles[i]
        for pot_obs in self.potential_obstacles: pot_obs['seen_this_frame'], pot_obs['confidence'] = False, pot_obs['confidence']*OBSTACLE_CONFIDENCE_DECAY;
        for cluster in current_frame_clusters:
            if check_obstacle_wall_collision(cluster, estimated_walls): continue
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
        newly_confirmed_indices = [i for i, pot_obs in enumerate(self.potential_obstacles) if pot_obs['confidence'] >= OBSTACLE_CONFIRMATION_THRESHOLD and not any(math.hypot(c['center'][0]-pot_obs['center'][0], c['center'][1]-pot_obs['center'][1]) < (c['radius']+pot_obs['radius']) for c in self.confirmed_obstacles)]
        for i in sorted(newly_confirmed_indices, reverse=True): self.confirmed_obstacles.append(self.potential_obstacles.pop(i))
        self.potential_obstacles = [p for p in self.potential_obstacles if p['confidence'] > 0.1]
    def draw(self, screen):
        for obs in self.confirmed_obstacles: pygame.draw.circle(screen, COLOR_OBSTACLE_DETECTED, obs['center'], obs['radius'], 2)
        s = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        for obs in self.potential_obstacles: alpha = min(200, 20 + int(obs['confidence'] * 40)); pygame.draw.circle(s, (*COLOR_OBSTACLE_POTENTIAL[:3], alpha), obs['center'], obs['radius'])
        screen.blit(s, (0,0))

class Pathfinder:
    def __init__(self, grid_size):
        self.grid_size = grid_size
        self.width = SCREEN_WIDTH // grid_size
        self.height = SCREEN_HEIGHT // grid_size
        self.collision_grid = np.zeros((self.width, self.height), dtype=bool)
        self.NUM_DIRECTIONS = 8
        self.TURN_PENALTY = 1.5

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
        angle = (angle + 360) % 360
        angle_per_segment = 360 / self.NUM_DIRECTIONS
        return int((angle + angle_per_segment / 2) / 360 * self.NUM_DIRECTIONS) % self.NUM_DIRECTIONS

    def find_path(self, start_pos, start_angle, end_pos, end_angle_degrees):
        start_grid_pos = (int(start_pos[0] // self.grid_size), int(start_pos[1] // self.grid_size))
        end_grid_pos = (int(end_pos[0] // self.grid_size), int(end_pos[1] // self.grid_size))
        if not (0 <= start_grid_pos[0] < self.width and 0 <= start_grid_pos[1] < self.height and 0 <= end_grid_pos[0] < self.width and 0 <= end_grid_pos[1] < self.height): return None
        if self.collision_grid[start_grid_pos] or self.collision_grid[end_grid_pos]: return None
        start_angle_idx, end_angle_idx = self._discretize_angle(start_angle), self._discretize_angle(end_angle_degrees)
        start_node = (*start_grid_pos, start_angle_idx)
        open_set, came_from = [(0, start_node)], {}
        g_score = {start_node: 0}; f_score = {start_node: self.heuristic(start_grid_pos, end_grid_pos)}
        while open_set:
            _, current_node = heapq.heappop(open_set)
            current_pos, current_angle_idx = current_node[:2], current_node[2]
            if current_pos == end_grid_pos and current_angle_idx == end_angle_idx: return self.reconstruct_path(came_from, current_node)
            for dx, dy in [(0,1), (0,-1), (1,0), (-1,0), (1,1), (1,-1), (-1,1), (-1,-1)]:
                neighbor_pos = (current_pos[0] + dx, current_pos[1] + dy)
                if not (0 <= neighbor_pos[0] < self.width and 0 <= neighbor_pos[1] < self.height) or self.collision_grid[neighbor_pos]: continue
                move_angle = math.degrees(math.atan2(dy, dx))
                neighbor_angle_idx = self._discretize_angle(move_angle)
                neighbor_node = (*neighbor_pos, neighbor_angle_idx)
                distance_cost = math.hypot(dx, dy)
                angle_diff = abs(current_angle_idx - neighbor_angle_idx)
                if angle_diff > self.NUM_DIRECTIONS / 2: angle_diff = self.NUM_DIRECTIONS - angle_diff
                turning_cost = self.TURN_PENALTY * angle_diff
                tentative_g_score = g_score.get(current_node, float('inf')) + distance_cost + turning_cost
                if tentative_g_score < g_score.get(neighbor_node, float('inf')):
                    came_from[neighbor_node], g_score[neighbor_node] = current_node, tentative_g_score
                    f_score[neighbor_node] = tentative_g_score + self.heuristic(neighbor_pos, end_grid_pos)
                    heapq.heappush(open_set, (f_score[neighbor_node], neighbor_node))
        return None
    def heuristic(self, a, b): return math.hypot(a[0] - b[0], a[1] - b[1])
    def reconstruct_path(self, came_from, current):
        path = [current[:2]];
        while current in came_from: current = came_from[current]; path.append(current[:2])
        return path[::-1]

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
            current_dist_along_segment += needed_dist
            dist_since_last_wp = 0.0
        dist_since_last_wp += (segment_len - current_dist_along_segment)
    waypoints.append(world_path[-1])
    return waypoints

def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Autonomous Robot SLAM Simulation")
    clock = pygame.time.Clock()
    font, small_font = pygame.font.Font(None, 28), pygame.font.Font(None, 22)

    arc_surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
    path_surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
    pathfinder = Pathfinder(PATHFINDING_GRID_SIZE)
    
    robot = Robot(150, 150)
    heatmap = Heatmap(SCREEN_WIDTH, SCREEN_HEIGHT, HEATMAP_CELL_SIZE)
    obstacle_manager = ObstacleManager()

    obstacles_for_drawing, obstacle_walls = generate_obstacles()
    all_walls_for_lidar = COURSE_WALLS + obstacle_walls
    
    pathfinding_state = "IDLE"
    pending_target_pos = None
    final_goal_pos = None
    final_goal_angle = None

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if pathfinding_state == "IDLE":
                    pending_target_pos = event.pos
                    pathfinding_state = "AWAITING_DIRECTION"
                    robot.set_path([]); robot.sharp_turn_locations.clear()
                
                elif pathfinding_state == "AWAITING_DIRECTION":
                    pathfinder.build_collision_grid(COURSE_WALLS + obstacle_walls, obstacle_manager.confirmed_obstacles)
                    dx = event.pos[0] - pending_target_pos[0]
                    dy = event.pos[1] - pending_target_pos[1]
                    target_angle_degrees = math.degrees(math.atan2(dy, dx))
                    
                    path_nodes = pathfinder.find_path(
                        start_pos=(robot.x, robot.y), start_angle=robot.angle,
                        end_pos=pending_target_pos, end_angle_degrees=target_angle_degrees
                    )
                    
                    if path_nodes:
                        # --- MODIFICATION: Prepend a "straight-ahead" waypoint ---
                        # 1. Generate the main path waypoints
                        raw_waypoints = generate_equidistant_waypoints(path_nodes, PATHFINDING_GRID_SIZE, WAYPOINT_DISTANCE)
                        # 2. Calculate a point directly in front of the robot
                        start_angle_rad = math.radians(robot.angle)
                        startup_wp = (robot.x + (WAYPOINT_DISTANCE / 2.0) * math.cos(start_angle_rad),
                                      robot.y + (WAYPOINT_DISTANCE / 2.0) * math.sin(start_angle_rad))
                        # 3. Prepend this startup waypoint to the list
                        final_path = [startup_wp] + raw_waypoints
                        
                        robot.set_path(final_path)
                        final_goal_pos = pending_target_pos
                        final_goal_angle = target_angle_degrees
                    else:
                        print("No path found!")
                        robot.set_path([]); final_goal_pos = None; final_goal_angle = None
                    pathfinding_state = "IDLE"
                    pending_target_pos = None
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    obstacles_for_drawing, obstacle_walls = generate_obstacles()
                    all_walls_for_lidar = COURSE_WALLS + obstacle_walls
                    obstacle_manager.reset(); robot.set_path([]); robot.sharp_turn_locations.clear()
                    final_goal_pos = None; final_goal_angle = None; pathfinding_state = "IDLE"
                if event.key in [pygame.K_w, pygame.K_s, pygame.K_a, pygame.K_d]:
                    robot.set_path([]); robot.sharp_turn_locations.clear()
                    final_goal_pos = None; final_goal_angle = None; pathfinding_state = "IDLE"
                if robot.mode == 'MANUAL':
                    if event.key == pygame.K_w: robot.change_speed(1)
                    if event.key == pygame.K_s: robot.change_speed(-1)

        keys = pygame.key.get_pressed()
        robot.update(keys)

        if robot.replan_needed:
            robot.replan_needed = False
            if final_goal_pos and final_goal_angle is not None:
                pathfinder.build_collision_grid(COURSE_WALLS + obstacle_walls, obstacle_manager.confirmed_obstacles)
                new_path_nodes = pathfinder.find_path(
                    start_pos=(robot.x, robot.y), start_angle=robot.angle,
                    end_pos=final_goal_pos, end_angle_degrees=final_goal_angle
                )
                if new_path_nodes:
                    # Same "straight-ahead" logic as above for replanned paths
                    raw_waypoints = generate_equidistant_waypoints(new_path_nodes, PATHFINDING_GRID_SIZE, WAYPOINT_DISTANCE)
                    start_angle_rad = math.radians(robot.angle)
                    startup_wp = (robot.x + (WAYPOINT_DISTANCE / 2.0) * math.cos(start_angle_rad),
                                  robot.y + (WAYPOINT_DISTANCE / 2.0) * math.sin(start_angle_rad))
                    final_path = [startup_wp] + raw_waypoints
                    robot.set_path(final_path)
                else:
                    robot.set_path([]); final_goal_pos = None; final_goal_angle = None
            else:
                robot.set_path([])

        robot.simulate_lidar(all_walls_for_lidar)
        robot.estimate_walls()
        obstacle_manager.update(robot.cluster_unassociated_points(), robot.estimated_walls)
        heatmap.add_points(robot.lidar_points)
        heatmap.decay(HEATMAP_DECAY_RATE)

        # --- Drawing ---
        screen.fill(COLOR_BACKGROUND)
        heatmap.update_and_draw(screen)
        for wall in COURSE_WALLS: pygame.draw.line(screen, COLOR_WALL, (wall[0], wall[1]), (wall[2], wall[3]), 2)
        for obs in obstacles_for_drawing:
            pygame.draw.rect(screen, obs['color'], obs['rect']); pygame.draw.rect(screen, COLOR_WALL, obs['rect'], 1)
        for p in robot.inlier_points_for_viz: pygame.draw.circle(screen, COLOR_INLIER_POINTS, p, 3)
        for wall in robot.estimated_walls: pygame.draw.line(screen, COLOR_ESTIMATED_WALL, wall[0], wall[1], 5)
        obstacle_manager.draw(screen)
        robot_pos = (robot.x, robot.y)

        # Distance measurement lines
        for wall_segment in robot.estimated_walls:
            p_start, p_end = wall_segment[0], wall_segment[1]
            closest_point = find_closest_point_on_segment(robot_pos, p_start, p_end)
            distance = math.hypot(robot_pos[0] - closest_point[0], robot_pos[1] - closest_point[1])
            vec_robot_to_point = np.array(closest_point) - np.array(robot_pos)
            vec_wall = np.array(p_end) - np.array(p_start)
            mag_robot_vec, mag_wall_vec = np.linalg.norm(vec_robot_to_point), np.linalg.norm(vec_wall)
            angle_deg = 90
            if mag_robot_vec > 0 and mag_wall_vec > 0:
                dot_product = np.dot(vec_robot_to_point, vec_wall)
                angle_rad = math.acos(np.clip(dot_product / (mag_robot_vec * mag_wall_vec), -1.0, 1.0))
                angle_deg = abs(90 - math.degrees(angle_rad))
            line_color = COLOR_PERPENDICULAR if angle_deg <= PERPENDICULAR_ANGLE_THRESHOLD else COLOR_NON_PERPENDICULAR
            pygame.draw.line(screen, line_color, robot_pos, closest_point, 2)
            text_surface = small_font.render(f"{distance:.0f}", True, COLOR_TEXT)
            mid_point = ((robot_pos[0] + closest_point[0]) / 2, (robot_pos[1] + closest_point[1]) / 2)
            text_rect = text_surface.get_rect(center=mid_point)
            pygame.draw.rect(screen, COLOR_BACKGROUND, text_rect.inflate(6, 4))
            screen.blit(text_surface, text_rect)

        # Detection Arc
        arc_surface.fill((0, 0, 0, 0))
        poly_points = [(robot.x, robot.y)]
        half_arc_rad, robot_angle_rad = math.radians(OBSTACLE_DETECTION_ARC_DEGREES / 2), math.radians(robot.angle)
        start_angle, end_angle = robot_angle_rad - half_arc_rad, robot_angle_rad + half_arc_rad
        for i in range(21):
            angle = start_angle + (end_angle - start_angle) * i / 20
            poly_points.append((robot.x + UI_ARC_RADIUS * math.cos(angle), robot.y + UI_ARC_RADIUS * math.sin(angle)))
        pygame.draw.polygon(arc_surface, COLOR_DETECTION_ARC, poly_points)
        screen.blit(arc_surface, (0, 0))
        
        # Path and Waypoints
        path_surface.fill((0, 0, 0, 0))
        if robot.path:
            if robot.current_waypoint_index < len(robot.path):
                points_to_draw = [robot_pos] + robot.path[robot.current_waypoint_index:]
                pygame.draw.lines(path_surface, COLOR_PLANNED_PATH, False, points_to_draw, 3)
            for i in range(robot.current_waypoint_index, len(robot.path)):
                pygame.draw.circle(path_surface, COLOR_WAYPOINT, robot.path[i], 5)
            if final_goal_angle is not None:
                last_wp = robot.path[-1]
                angle_rad = math.radians(final_goal_angle)
                arrow_len = 25
                end_arrow_x, end_arrow_y = last_wp[0] + arrow_len * math.cos(angle_rad), last_wp[1] + arrow_len * math.sin(angle_rad)
                pygame.draw.line(path_surface, COLOR_WAYPOINT, last_wp, (end_arrow_x, end_arrow_y), 4)
        
        if pathfinding_state == "AWAITING_DIRECTION":
            pygame.draw.circle(path_surface, COLOR_WAYPOINT, pending_target_pos, 8)
            pygame.draw.line(path_surface, COLOR_WAYPOINT, pending_target_pos, pygame.mouse.get_pos(), 2)
        screen.blit(path_surface, (0,0))
        
        # --- MODIFICATION: Draw sharp turn visualization on top of the path ---
        for pos in robot.sharp_turn_locations:
            pygame.draw.circle(screen, COLOR_SHARP_TURN, pos, 15, 2)
        
        # Robot Body
        pygame.draw.circle(screen, COLOR_ROBOT, robot_pos, ROBOT_SIZE)
        pygame.draw.circle(screen, (255,255,255), robot_pos, ROBOT_SIZE, 1)
        end_x = robot.x + ROBOT_SIZE * math.cos(math.radians(robot.angle))
        end_y = robot.y + ROBOT_SIZE * math.sin(math.radians(robot.angle))
        pygame.draw.line(screen, (255, 255, 255), robot_pos, (end_x, end_y), 2)
        steer_angle_rad = math.radians(robot.angle - robot.steering_angle)
        steer_x = robot.x + ROBOT_SIZE * 0.8 * math.cos(steer_angle_rad)
        steer_y = robot.y + ROBOT_SIZE * 0.8 * math.sin(steer_angle_rad)
        pygame.draw.line(screen, (0, 255, 0), robot_pos, (steer_x, steer_y), 2)
        
        # UI Text
        instruction_text_str = ""
        if robot.mode == 'MANUAL':
            if pathfinding_state == "IDLE":
                instruction_text_str = "Click to set destination. WASD to drive. 'R' for new obstacles."
            elif pathfinding_state == "AWAITING_DIRECTION":
                instruction_text_str = "Destination set. Now click to set the desired FINAL DIRECTION."
        elif robot.mode == 'AUTO':
            instruction_text_str = "Mode: AUTO. Navigating to target. Press WASD to take over."
        elif robot.mode == 'REVERSING':
            instruction_text_str = "Mode: REVERSING. Attempting to get unstuck..."
        if robot.replan_needed: # This state is transient but can be useful to show
            instruction_text_str = f"Mode: AUTO. Turn too sharp! Replan attempt #{robot.consecutive_replan_count}"
            
        screen.blit(font.render(instruction_text_str, True, COLOR_TEXT), (10, 10))

        pygame.display.flip()
        clock.tick(60)
    pygame.quit()

if __name__ == '__main__':
    main()