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
COLOR_TRUE_SENSOR = (0, 255, 0) # Green for actual mouse position
COLOR_ESTIMATED_POSE = (0, 150, 255) # Blue for estimated robot position
COLOR_PARTICLE = (255, 0, 255) # Pink for other possible points (particles)
COLOR_LIDAR_POINT = (255, 255, 255) # White for raw lidar hits
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
COLOR_DYNAMIC_WALL = (255, 165, 0) # Bright Orange for new walls
COLOR_SEGMENT_BORDER = (255, 0, 255) # Magenta for segment borders
COLOR_UNASSOCIATED_POINTS_DEBUG = (255, 128, 0) # Orange for debugging

ORANGE = (255, 0, 255)
BLACK =  (10, 10, 10)
# Sensor properties (visual representation only)
SENSOR_SIZE = 15
# LiDAR properties
LIDAR_RANGE = 5000
LIDAR_RAYS = 360
LIDAR_INACCURACY = 6.0
# RANSAC & Clustering Parameters
RANSAC_ITERATIONS = 20
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
# Course & Obstacle Definition
COURSE_TOP_LEFT = 50
COURSE_WIDTH = 600
OBSTACLE_SIZE = (10, 10)
INNER_TL = (200 + COURSE_TOP_LEFT, 200 + COURSE_TOP_LEFT)
INNER_TR = (400 + COURSE_TOP_LEFT, 200 + COURSE_TOP_LEFT)
INNER_BL = (200 + COURSE_TOP_LEFT, 400 + COURSE_TOP_LEFT)
INNER_BR = (400 + COURSE_TOP_LEFT, 400 + COURSE_TOP_LEFT)

DIRECTION_OF_TRAVEL = "clockwise" # Used for dynamic wall generation

# Particle Filter Parameters
NUM_PARTICLES = 500 # Number of particles for localization
PARTICLE_INITIAL_SPREAD = 100 # Initial random spread for particles
PARTICLE_NOISE_STD_DEV = {'x': 2.0, 'y': 2.0, 'angle': 1.0} # Noise added to particles after resampling
PARTICLE_RE_INITIALIZE_THRESHOLD = 0.05 # If effective number of particles drops too low, re-initialize some.

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
        if len(wall) != 4: # Assume it's a list of two points for dynamic walls
            wall_start = (wall[0][0], wall[0][1])
            wall_end = (wall[1][0], wall[1][1])
        else: # Standard wall tuple (x1, y1, x2, y2)
            wall_start = (wall[0], wall[1])
            wall_end = (wall[2], wall[3])
            
        closest_point = find_closest_point_on_segment(center, wall_start, wall_end)
        dist_sq = (center[0] - closest_point[0])**2 + (center[1] - closest_point[1])**2
        if dist_sq < radius_sq:
            return True
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

def get_obstacle_at_coord(x, y, obstacles_list):
    point_to_check = (x, y)
    for obstacle in obstacles_list:
        if obstacle['rect'].collidepoint(point_to_check):
            return (True, obstacle['color'])
    return (False, None)

# --- Lidar Simulator Class (combines relevant Robot methods) ---
class LidarSimulator:
    def __init__(self):
        self.lidar_points = []
        self.estimated_walls = []
        self.inlier_points_for_viz = []
        self.unassociated_points = []
        self.current_frame_clusters = []

    def simulate_lidar(self, x, y, angle, walls):
        self.lidar_points = []
        sensor_pos = (x, y)
        sensor_angle_rad = math.radians(angle)
        for i in range(LIDAR_RAYS):
            ray_angle_rad = sensor_angle_rad + math.radians((360 / LIDAR_RAYS) * i)
            end_x, end_y = x + LIDAR_RANGE * math.cos(ray_angle_rad), y + LIDAR_RANGE * math.sin(ray_angle_rad)
            closest_dist, hit_point = LIDAR_RANGE, None
            for wall in walls:
                p, d = line_intersection(sensor_pos, (end_x, end_y), (wall[0], wall[1]), (wall[2], wall[3]))
                if p and d < closest_dist: closest_dist, hit_point = d, p
            if hit_point:
                dist_with_error = closest_dist + random.uniform(-LIDAR_INACCURACY, LIDAR_INACCURACY)
                if dist_with_error > LIDAR_RANGE:
                    hit_point = (x + LIDAR_RANGE * math.cos(ray_angle_rad), y + LIDAR_RANGE * math.sin(ray_angle_rad))
                else:
                    hit_point = (x + dist_with_error * math.cos(ray_angle_rad), y + dist_with_error * math.sin(ray_angle_rad))
                self.lidar_points.append(hit_point)
        return self.lidar_points

    def cluster_inliers_by_distance(self, inliers):
        if not inliers: return []
        data = np.array(inliers)
        if data.ndim == 1: return [[tuple(data)]]
        if np.linalg.matrix_rank(np.cov(data.T)) < 2: return [inliers]
            
        mean = np.mean(data, axis=0)
        _, eigenvectors = np.linalg.eigh(np.cov(data.T))
        direction_vector = eigenvectors[:, -1]
        
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
        if np.linalg.matrix_rank(np.cov(data.T)) < 1: # All points are identical
            return ((mean[0] - 1, mean[1]), (mean[0] + 1, mean[1]))
            
        _, eigenvectors = np.linalg.eigh(np.cov(data.T))
        direction_vector = eigenvectors[:, -1]
        projections = np.dot(data - mean, direction_vector)
        p1 = (mean + np.min(projections) * direction_vector).tolist()
        p2 = (mean + np.max(projections) * direction_vector).tolist()
        return (p1, p2)

    def estimate_walls(self):
        self.estimated_walls, self.inlier_points_for_viz = [], []
        remaining_points = list(self.lidar_points)
        while len(remaining_points) > RANSAC_MIN_INLIERS:
            best_inliers = []
            for _ in range(RANSAC_ITERATIONS):
                if len(remaining_points) < 2: break
                sample_points = random.sample(remaining_points, 2)
                p1, p2 = sample_points[0], sample_points[1]
                if math.hypot(p1[0]-p2[0], p1[1]-p2[1]) < RANSAC_THRESHOLD_DISTANCE: continue
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

    def cluster_unassociated_points(self, x, y, angle):
        points_in_arc = [p for p in self.unassociated_points if abs((math.degrees(math.atan2(p[1] - y, p[0] - x)) - angle + 180) % 360 - 180) <= OBSTACLE_DETECTION_ARC_DEGREES / 2.0]
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
        self.current_frame_clusters = clusters_found
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

    def reset(self):
        self.potential_obstacles.clear(); self.confirmed_obstacles.clear()

    def getblockSegment(self, point):
        x, y = point[0], point[1]
        c1 = INNER_TR[1] + INNER_TR[0]
        side1 = y + x - c1
        c2 = INNER_TL[1] - INNER_TL[0]
        side2 = y - x - c2
        if side1 < 0 and side2 < 0: return 1 # Top
        elif side1 > 0 and side2 < 0: return 2 # Right
        elif side1 > 0 and side2 > 0: return 3 # Bottom
        elif side1 < 0 and side2 > 0: return 4 # Left
        else: return 1

    def update(self, current_frame_clusters, walls_for_validation, obstacles_for_drawing):
        invalidated_confirmed_indices = [i for i, obs in enumerate(self.confirmed_obstacles) if check_obstacle_wall_collision(obs, walls_for_validation)]
        for i in sorted(invalidated_confirmed_indices, reverse=True): del self.confirmed_obstacles[i]
        for pot_obs in self.potential_obstacles: pot_obs['seen_this_frame'], pot_obs['confidence'] = False, pot_obs['confidence']*OBSTACLE_CONFIDENCE_DECAY;
        
        newly_confirmed_obstacles = []
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
        
        newly_confirmed_indices = [i for i, pot_obs in enumerate(self.potential_obstacles) 
                                 if pot_obs['confidence'] >= OBSTACLE_CONFIRMATION_THRESHOLD 
                                 and not any(math.hypot(c['center'][0] - pot_obs['center'][0], 
                                                      c['center'][1] - pot_obs['center'][1]) < (c['radius'] + pot_obs['radius']) 
                                           for c in self.confirmed_obstacles)]
        
        for i in sorted(newly_confirmed_indices, reverse=True):
            obs_to_confirm = self.potential_obstacles.pop(i)
            obs_to_confirm['segment'] = self.getblockSegment(obs_to_confirm['center'])
            is_obstacle, obstacle_color = get_obstacle_at_coord(int(obs_to_confirm['center'][0]), int(obs_to_confirm['center'][1]), obstacles_for_drawing)
            obs_to_confirm['color'] = obstacle_color if is_obstacle else COLOR_OBSTACLE_RED
            self.confirmed_obstacles.append(obs_to_confirm)
            newly_confirmed_obstacles.append(obs_to_confirm)
        self.potential_obstacles = [p for p in self.potential_obstacles if p['confidence'] > 0.1]
        return newly_confirmed_obstacles

    def draw(self, screen):
        for obs in self.confirmed_obstacles: pygame.draw.circle(screen, COLOR_OBSTACLE_DETECTED, obs['center'], obs['radius'], 2)
        s = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        for obs in self.potential_obstacles: alpha = min(200, 20 + int(obs['confidence'] * 40)); pygame.draw.circle(s, (*COLOR_OBSTACLE_POTENTIAL[:3], alpha), obs['center'], obs['radius'])
        screen.blit(s, (0,0))

def create_dynamic_wall_for_obstacle(obstacle_center, obstacle_color, segment, direction_of_travel):
    cx, cy = obstacle_center
    MAX_WALL_LENGTH = 200
    pass_on_right = (obstacle_color == COLOR_OBSTACLE_RED if direction_of_travel == "clockwise" else obstacle_color == COLOR_OBSTACLE_GREEN)
    
    if segment == 1: wall_end = (cx, min(cy + MAX_WALL_LENGTH, COURSE_TOP_LEFT + COURSE_WIDTH) if pass_on_right else max(cy - MAX_WALL_LENGTH, COURSE_TOP_LEFT))
    elif segment == 2: wall_end = (max(cx - MAX_WALL_LENGTH, COURSE_TOP_LEFT) if pass_on_right else min(cx + MAX_WALL_LENGTH, COURSE_TOP_LEFT + COURSE_WIDTH), cy)
    elif segment == 3: wall_end = (cx, max(cy - MAX_WALL_LENGTH, COURSE_TOP_LEFT) if pass_on_right else min(cy + MAX_WALL_LENGTH, COURSE_TOP_LEFT + COURSE_WIDTH))
    elif segment == 4: wall_end = (min(cx + MAX_WALL_LENGTH, COURSE_TOP_LEFT + COURSE_WIDTH) if pass_on_right else max(cx - MAX_WALL_LENGTH, COURSE_TOP_LEFT), cy)
    else: wall_end = (cx, min(cy + MAX_WALL_LENGTH, COURSE_TOP_LEFT + COURSE_WIDTH) if abs(cy - COURSE_TOP_LEFT) < abs(cy - (COURSE_TOP_LEFT + COURSE_WIDTH)) else max(cy - MAX_WALL_LENGTH, COURSE_TOP_LEFT))
    
    return (cx, cy, wall_end[0], wall_end[1])

def group_nearby_obstacles(obstacles, max_distance=100):
    if not obstacles: return []
    groups, remaining_obstacles = [], obstacles.copy()
    while remaining_obstacles:
        current_group = [remaining_obstacles.pop(0)]
        found_new = True
        while found_new:
            found_new = False
            for i in range(len(remaining_obstacles) - 1, -1, -1):
                candidate = remaining_obstacles[i]
                for group_obstacle in current_group:
                    dist = math.hypot(candidate['center'][0] - group_obstacle['center'][0], candidate['center'][1] - group_obstacle['center'][1])
                    if dist <= max_distance: current_group.append(remaining_obstacles.pop(i)); found_new = True; break
                if found_new: break
        groups.append(current_group)
    return groups

def create_dynamic_wall_for_obstacle_group(obstacle_group, direction_of_travel):
    if not obstacle_group: return None
    representative_obstacle = obstacle_group[0]
    min_x = min(obs['center'][0] for obs in obstacle_group); max_x = max(obs['center'][0] for obs in obstacle_group)
    min_y = min(obs['center'][1] for obs in obstacle_group); max_y = max(obs['center'][1] for obs in obstacle_group)
    group_center = ((min_x + max_x) / 2, (min_y + max_y) / 2)
    return create_dynamic_wall_for_obstacle(
        group_center, representative_obstacle.get('color', COLOR_OBSTACLE_RED), representative_obstacle['segment'], direction_of_travel
    )

# --- Particle Filter Class ---
class ParticleFilter:
    def __init__(self, num_particles, initial_x, initial_y, initial_angle, initial_spread, lidar_simulator_instance, walls):
        self.num_particles = num_particles
        self.lidar_sim = lidar_simulator_instance
        self.walls = walls # Global map walls
        self.particles = []
        self.estimated_pose = {'x': initial_x, 'y': initial_y, 'angle': initial_angle}
        self.localization_score = 1.0 # Initial confidence

        self.initialize_particles(initial_x, initial_y, initial_angle, initial_spread)

    def initialize_particles(self, x, y, angle, spread):
        self.particles = []
        for _ in range(self.num_particles):
            px = x + random.uniform(-spread, spread)
            py = y + random.uniform(-spread, spread)
            p_angle = (angle + random.uniform(-spread / 2, spread / 2)) % 360
            self.particles.append({'x': px, 'y': py, 'angle': p_angle, 'weight': 1.0 / self.num_particles})

    def update(self, observed_lidar_points):
        # 1. Prediction (noise only, no motion model as mouse directly controls true position)
        for p in self.particles:
            p['x'] += random.gauss(0, PARTICLE_NOISE_STD_DEV['x'])
            p['y'] += random.gauss(0, PARTICLE_NOISE_STD_DEV['y'])
            p['angle'] = (p['angle'] + random.gauss(0, PARTICLE_NOISE_STD_DEV['angle'])) % 360

        # 2. Update weights based on observations
        total_weight = 0.0
        max_weight = 0.0

        if not observed_lidar_points: # No lidar data, all weights decay
            for p in self.particles:
                p['weight'] *= 0.8
            self.localization_score = max(0.0, self.localization_score * 0.95 - 0.01) # Decay confidence
        else:
            for p in self.particles:
                # Simulate lidar from particle's perspective using the global map
                predicted_lidar_points = self.lidar_sim.simulate_lidar(p['x'], p['y'], p['angle'], self.walls)
                
                # Compare predicted to observed points
                error = self.calculate_match_error(predicted_lidar_points, observed_lidar_points)
                
                # Convert error to weight (lower error = higher weight)
                p['weight'] *= math.exp(-error / (2 * (LIDAR_INACCURACY * 2)**2)) # sigma^2 based on lidar inaccuracy
                p['weight'] = max(p['weight'], 1e-10) # Prevent zero weights

                total_weight += p['weight']
                if p['weight'] > max_weight:
                    max_weight = p['weight']
            
            # Normalize weights
            if total_weight > 0:
                for p in self.particles:
                    p['weight'] /= total_weight
            else: # All weights are zero, re-initialize
                print("Particle filter lost, re-initializing.")
                self.initialize_particles(self.estimated_pose['x'], self.estimated_pose['y'], self.estimated_pose['angle'], PARTICLE_INITIAL_SPREAD * 2) # Larger spread
                self.localization_score = 0.1 # Mark as lost
                return

            # Effective Number of Particles (Neff) as a measure of degeneracy
            weights_np = np.array([p['weight'] for p in self.particles])
            neff = 1.0 / np.sum(weights_np**2) if np.sum(weights_np**2) > 0 else 0
            
            # Confidence update
            self.localization_score = np.clip(self.localization_score * 0.9 + (neff / self.num_particles) * 0.1, 0.0, 1.0) # Blend Neff into confidence

        # 3. Resample if Neff is low or routinely
        if neff < self.num_particles * PARTICLE_RE_INITIALIZE_THRESHOLD:
            # print(f"Low Neff ({neff:.2f}), re-sampling with noise.")
            self.resample_particles()
        elif random.random() < 0.1: # Resample periodically to introduce diversity
            self.resample_particles()

        # 4. Estimate pose
        self.estimated_pose = self.calculate_estimated_pose()
        
    def calculate_match_error(self, predicted_points, observed_points):
        # For each observed point, find the closest predicted point. Sum squared distances.
        if not observed_points or not predicted_points:
            return float('inf') # High error if no points or no predicted points
        
        error_sum_sq = 0.0
        for obs_p in observed_points:
            min_dist_sq = float('inf')
            for pred_p in predicted_points:
                dist_sq = (obs_p[0] - pred_p[0])**2 + (obs_p[1] - pred_p[1])**2
                min_dist_sq = min(min_dist_sq, dist_sq)
            error_sum_sq += min_dist_sq
        
        return error_sum_sq / len(observed_points) # Normalize by number of observed points

    def resample_particles(self):
        new_particles = []
        weights = np.array([p['weight'] for p in self.particles])
        if np.sum(weights) == 0:
            self.initialize_particles(self.estimated_pose['x'], self.estimated_pose['y'], self.estimated_pose['angle'], PARTICLE_INITIAL_SPREAD)
            return

        # Low Variance Resampling
        cumulative_sum = np.cumsum(weights)
        
        r = random.uniform(0, 1.0 / self.num_particles)
        j = 0
        for i in range(self.num_particles):
            while j < self.num_particles and r > cumulative_sum[j]:
                j += 1
            if j >= self.num_particles: # Fallback in case of floating point issues at end
                j = self.num_particles - 1
            new_particle = self.particles[j].copy()
            
            # Add small noise to new particles
            new_particle['x'] += random.gauss(0, PARTICLE_NOISE_STD_DEV['x'])
            new_particle['y'] += random.gauss(0, PARTICLE_NOISE_STD_DEV['y'])
            new_particle['angle'] = (new_particle['angle'] + random.gauss(0, PARTICLE_NOISE_STD_DEV['angle'])) % 360
            new_particle['weight'] = 1.0 / self.num_particles # Reset weights after resampling
            
            new_particles.append(new_particle)
            r += 1.0 / self.num_particles
        
        self.particles = new_particles

    def calculate_estimated_pose(self):
        # Weighted average of all particles
        x_sum, y_sum, sin_angle_sum, cos_angle_sum = 0.0, 0.0, 0.0, 0.0
        total_weight = sum(p['weight'] for p in self.particles)
        
        if total_weight == 0:
            return self.estimated_pose

        for p in self.particles:
            x_sum += p['x'] * p['weight']
            y_sum += p['y'] * p['weight']
            rad = math.radians(p['angle'])
            sin_angle_sum += math.sin(rad) * p['weight']
            cos_angle_sum += math.cos(rad) * p['weight']
        
        estimated_x = x_sum / total_weight
        estimated_y = y_sum / total_weight
        estimated_angle = math.degrees(math.atan2(sin_angle_sum, cos_angle_sum))
        
        return {'x': estimated_x, 'y': estimated_y, 'angle': estimated_angle}

def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Lidar Localization Module")
    clock = pygame.time.Clock()
    font, small_font = pygame.font.Font(None, 28), pygame.font.Font(None, 22)
    arc_surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
    
    lidar_simulator = LidarSimulator()
    heatmap = Heatmap(SCREEN_WIDTH, SCREEN_HEIGHT, HEATMAP_CELL_SIZE)
    obstacle_manager = ObstacleManager()
    obstacles_for_drawing, obstacle_walls = generate_obstacles()
    all_physical_walls = COURSE_WALLS + obstacle_walls # Ground truth map for lidar simulation

    # Initialize true sensor position and angle (mouse controlled)
    true_sensor_x, true_sensor_y = SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2
    last_mouse_pos = (true_sensor_x, true_sensor_y)
    true_sensor_angle = 0.0 # Will be updated based on mouse movement

    # Particle Filter
    particle_filter = ParticleFilter(NUM_PARTICLES, true_sensor_x, true_sensor_y, true_sensor_angle, PARTICLE_INITIAL_SPREAD, lidar_simulator, all_physical_walls)

    running = True
    while running:
        # Determine true sensor position and angle from mouse
        mouse_x, mouse_y = pygame.mouse.get_pos()
        dx, dy = mouse_x - last_mouse_pos[0], mouse_y - last_mouse_pos[1]
        
        # Only update angle if mouse has moved significantly
        if math.hypot(dx, dy) > 1:
            true_sensor_angle = (math.degrees(math.atan2(dy, dx)) + 360) % 360
            true_sensor_x, true_sensor_y = mouse_x, mouse_y
        last_mouse_pos = (true_sensor_x, true_sensor_y)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r: # Regenerate obstacles
                    obstacles_for_drawing, obstacle_walls = generate_obstacles()
                    all_physical_walls = COURSE_WALLS + obstacle_walls # Update ground truth map
                    obstacle_manager.reset() # Reset detected obstacles
                    # Re-initialize particle filter after map change
                    particle_filter.walls = all_physical_walls # Update filter's internal map reference
                    particle_filter.initialize_particles(true_sensor_x, true_sensor_y, true_sensor_angle, PARTICLE_INITIAL_SPREAD * 2) # Larger spread to adapt

                if event.key == pygame.K_c: # Clear detected obstacles
                    obstacle_manager.reset()
                if event.key == pygame.K_t: # Toggle direction for dynamic wall rule
                    global DIRECTION_OF_TRAVEL
                    DIRECTION_OF_TRAVEL = "anticlockwise" if DIRECTION_OF_TRAVEL == "clockwise" else "clockwise"
                    print(f"Dynamic Wall Rule Direction changed to: {DIRECTION_OF_TRAVEL}")
                    obstacle_manager.reset() # Reset detected obstacles when rule changes

        # 1. Simulate Lidar from true sensor position (mouse)
        # This is the "observation" step - what the sensor *actually* sees
        observed_lidar_points = lidar_simulator.simulate_lidar(true_sensor_x, true_sensor_y, true_sensor_angle, all_physical_walls)
        
        # 2. Particle Filter Update
        # The PF uses the observed lidar points to estimate its pose on the known map
        particle_filter.update(observed_lidar_points)
        estimated_pose = particle_filter.estimated_pose
        
        # 3. Lidar Processing (RANSAC, Clustering) based on the *observed* points
        # These operations act on the raw sensor data to build a local map/detect objects
        lidar_simulator.lidar_points = observed_lidar_points # Set the observed points as input for processing
        lidar_simulator.estimate_walls()
        # Clustering for obstacles is done from the *estimated* pose perspective
        lidar_simulator.cluster_unassociated_points(estimated_pose['x'], estimated_pose['y'], estimated_pose['angle'])
        
        # Determine walls for obstacle validation (based on current estimated map)
        dynamic_walls = [obs['dynamic_wall'] for obs in obstacle_manager.confirmed_obstacles if 'dynamic_wall' in obs]
        walls_for_cluster_validation = lidar_simulator.estimated_walls + dynamic_walls
        
        # Update obstacle manager based on current perception
        newly_confirmed = obstacle_manager.update(lidar_simulator.current_frame_clusters, walls_for_cluster_validation, obstacles_for_drawing)

        if newly_confirmed:
            obstacle_groups = group_nearby_obstacles(newly_confirmed, max_distance=100)
            for group in obstacle_groups:
                dynamic_wall = create_dynamic_wall_for_obstacle_group(group, DIRECTION_OF_TRAVEL)
                if dynamic_wall:
                    for obs in group:
                        obs['dynamic_wall'] = dynamic_wall
        
        heatmap.add_points(observed_lidar_points)
        heatmap.decay(HEATMAP_DECAY_RATE)

        # --- Drawing ---
        screen.fill(COLOR_BACKGROUND)
        heatmap.update_and_draw(screen)

        # Draw ground truth map
        for wall in COURSE_WALLS: pygame.draw.line(screen, COLOR_WALL, (wall[0], wall[1]), (wall[2], wall[3]), 2)
        for obs in obstacles_for_drawing: pygame.draw.rect(screen, obs['color'], obs['rect']); pygame.draw.rect(screen, COLOR_WALL, obs['rect'], 1)
        for wall in dynamic_walls: pygame.draw.line(screen, COLOR_DYNAMIC_WALL, (wall[0], wall[1]), (wall[2], wall[3]), 4)
        pygame.draw.line(screen, COLOR_SEGMENT_BORDER, INNER_TR, INNER_BL, 1)
        pygame.draw.line(screen, COLOR_SEGMENT_BORDER, INNER_TL, INNER_BR, 1)

        # Draw lidar and perception results (based on estimated pose for context)
        for p in observed_lidar_points: pygame.draw.circle(screen, COLOR_LIDAR_POINT, p, 2) # Raw lidar points from true position
        for p in lidar_simulator.inlier_points_for_viz: pygame.draw.circle(screen, COLOR_INLIER_POINTS, p, 3) # Inliers used for walls
        for wall in lidar_simulator.estimated_walls: pygame.draw.line(screen, COLOR_ESTIMATED_WALL, wall[0], wall[1], 5)
        obstacle_manager.draw(screen)

        # Draw perpendicular lines from ESTIMATED sensor to ESTIMATED walls
        estimated_sensor_pos_viz = (int(estimated_pose['x']), int(estimated_pose['y']))
        for wall_segment in lidar_simulator.estimated_walls:
            p_start, p_end = wall_segment[0], wall_segment[1]
            closest_point = find_closest_point_on_segment(estimated_sensor_pos_viz, p_start, p_end)
            distance = math.hypot(estimated_sensor_pos_viz[0] - closest_point[0], estimated_sensor_pos_viz[1] - closest_point[1])
            vec_sensor_to_point = np.array(closest_point) - np.array(estimated_sensor_pos_viz)
            vec_wall = np.array(p_end) - np.array(p_start)
            
            mag_sensor_vec = np.linalg.norm(vec_sensor_to_point)
            mag_wall_vec = np.linalg.norm(vec_wall)
            if mag_sensor_vec == 0 or mag_wall_vec == 0: angle_deg = 90
            else:
                cosine_angle = np.clip(np.dot(vec_sensor_to_point, vec_wall) / (mag_sensor_vec * mag_wall_vec), -1.0, 1.0)
                angle_deg = abs(90 - math.degrees(math.acos(cosine_angle)))
            
            line_color = COLOR_PERPENDICULAR if angle_deg <= PERPENDICULAR_ANGLE_THRESHOLD else COLOR_NON_PERPENDICULAR
            pygame.draw.line(screen, line_color, estimated_sensor_pos_viz, closest_point, 2)
            
            text_surface = small_font.render(f"{distance:.0f}", True, COLOR_TEXT)
            mid_point = ((estimated_sensor_pos_viz[0] + closest_point[0]) / 2, (estimated_sensor_pos_viz[1] + closest_point[1]) / 2)
            text_rect = text_surface.get_rect(center=mid_point)
            pygame.draw.rect(screen, COLOR_BACKGROUND, text_rect.inflate(6, 4)); screen.blit(text_surface, text_rect)

        # Draw the true sensor origin and heading (mouse position - green)
        pygame.draw.circle(screen, COLOR_TRUE_SENSOR, (int(true_sensor_x), int(true_sensor_y)), SENSOR_SIZE)
        pygame.draw.circle(screen, (255,255,255), (int(true_sensor_x), int(true_sensor_y)), SENSOR_SIZE, 1)
        end_x, end_y = true_sensor_x + SENSOR_SIZE * math.cos(math.radians(true_sensor_angle)), true_sensor_y + SENSOR_SIZE * math.sin(math.radians(true_sensor_angle))
        pygame.draw.line(screen, (255, 255, 255), (int(true_sensor_x), int(true_sensor_y)), (int(end_x), int(end_y)), 2)

        # Draw the detection arc (always centered at true_sensor_pos with true_sensor_angle)
        arc_surface.fill((0, 0, 0, 0))
        poly_points = [(true_sensor_x, true_sensor_y)]
        half_arc_rad, sensor_angle_rad = math.radians(OBSTACLE_DETECTION_ARC_DEGREES / 2), math.radians(true_sensor_angle)
        for i in range(21): poly_points.append((true_sensor_x + UI_ARC_RADIUS * math.cos(sensor_angle_rad - half_arc_rad + (half_arc_rad*2*i/20)), true_sensor_y + UI_ARC_RADIUS * math.sin(sensor_angle_rad - half_arc_rad + (half_arc_rad*2*i/20))))
        pygame.draw.polygon(arc_surface, COLOR_DETECTION_ARC, poly_points); screen.blit(arc_surface, (0, 0))

        # Draw estimated pose (blue circle with crosshair)
        pygame.draw.circle(screen, COLOR_ESTIMATED_POSE, estimated_sensor_pos_viz, SENSOR_SIZE + 5, 2)
        pygame.draw.circle(screen, COLOR_ESTIMATED_POSE, estimated_sensor_pos_viz, SENSOR_SIZE + 5) # Fill interior
        est_end_x = estimated_pose['x'] + SENSOR_SIZE * math.cos(math.radians(estimated_pose['angle']))
        est_end_y = estimated_pose['y'] + SENSOR_SIZE * math.sin(math.radians(estimated_pose['angle']))
        pygame.draw.line(screen, COLOR_TEXT, estimated_sensor_pos_viz, (est_end_x, est_end_y), 3)

        # Draw particles (pink dots)
        for p in particle_filter.particles:
            alpha = int(p['weight'] * 255 * 5) # Scale alpha by weight for visual emphasis
            alpha = np.clip(alpha, 10, 255) # Ensure minimum visibility
            particle_color = (*COLOR_PARTICLE[:3], alpha)
            s_particle = pygame.Surface((SENSOR_SIZE, SENSOR_SIZE), pygame.SRCALPHA)
            pygame.draw.circle(s_particle, particle_color, (s_particle.get_width()//2, s_particle.get_height()//2), SENSOR_SIZE // 4)
            screen.blit(s_particle, (int(p['x'] - s_particle.get_width()//2), int(p['y'] - s_particle.get_height()//2)))


        # UI Text
        info_text_str = "Lidar Localization Module. Mouse: True Sensor Pos. R: new obstacles. C: clear obstacles. T: toggle direction rule."
        screen.blit(font.render(info_text_str, True, COLOR_TEXT), (10, 10))
        
        estimated_pos_text = f"Estimated Pos: ({estimated_pose['x']:.0f}, {estimated_pose['y']:.0f}) Angle: {estimated_pose['angle']:.0f}°"
        screen.blit(small_font.render(estimated_pos_text, True, COLOR_TEXT), (10, 40))
        
        true_pos_text = f"True Pos (Mouse): ({true_sensor_x:.0f}, {true_sensor_y:.0f}) Angle: {true_sensor_angle:.0f}°"
        screen.blit(small_font.render(true_pos_text, True, COLOR_TEXT), (10, 65))

        localization_confidence_text = f"Localization Confidence: {particle_filter.localization_score:.2f}"
        screen.blit(small_font.render(localization_confidence_text, True, COLOR_TEXT), (10, 90))

        pygame.display.flip()
        clock.tick(60)
    pygame.quit()

if __name__ == '__main__':
    main()