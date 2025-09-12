import pygame
import numpy as np
import random
import math

# --- Configuration ---
# Screen
SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 700

# Colors (omitted for brevity, same as before)
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
COLOR_OBSTACLE_DETECTED = (255, 255, 0)
COLOR_OBSTACLE_POTENTIAL = (255, 255, 0, 100)

# Robot & LiDAR properties (omitted for brevity, same as before)
ROBOT_SIZE = 15
LIDAR_RANGE = 3000
LIDAR_RAYS = 360
LIDAR_INACCURACY = 6.0

# RANSAC & Clustering Parameters (omitted for brevity, same as before)
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

# Obstacle Persistence Parameters
OBSTACLE_CONFIDENCE_INCREMENT = 1.0
OBSTACLE_CONFIDENCE_DECAY = 0.95
OBSTACLE_CONFIRMATION_THRESHOLD = 5.0
OBSTACLE_MATCHING_DISTANCE = 20.0

# Heatmap Configuration
HEATMAP_CELL_SIZE = 8
HEATMAP_DECAY_RATE = 0.98
HEATMAP_MAX_HITS_ADJUST_RATE = 0.995

# --- Course & Obstacle Definition ---
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
    """Finds the point on segment AB closest to point P."""
    p_np, a_np, b_np = np.array(p), np.array(a), np.array(b)
    ab, ap = b_np - a_np, p_np - a_np
    ab_len_sq = np.dot(ab, ab)
    if ab_len_sq == 0: return a # A and B are the same point
    # Project P onto the line AB, clamping t between 0 (at A) and 1 (at B)
    t = np.clip(np.dot(ap, ab) / ab_len_sq, 0, 1)
    return tuple(a_np + t * ab)

# --- MODIFICATION START: New Collision Helper ---
def check_obstacle_wall_collision(obstacle, walls):
    """Checks if a circular obstacle intersects any wall segment."""
    center = obstacle['center']
    radius_sq = obstacle['radius']**2

    for wall_start, wall_end in walls:
        # Find the point on the wall segment closest to the obstacle center
        closest_point = find_closest_point_on_segment(center, wall_start, wall_end)
        
        # Calculate the squared distance between the center and the closest point
        dist_sq = (center[0] - closest_point[0])**2 + (center[1] - closest_point[1])**2
        
        # If the distance is less than the radius, they collide
        if dist_sq < radius_sq:
            return True
    return False
# --- MODIFICATION END ---

def generate_obstacles():
    # (generate_obstacles implementation omitted for brevity, same as before)
    obstacle_slots = {k: [(v[0], v[1]) for v in vs] for k, vs in {
        "top": [(200+COURSE_TOP_LEFT, 80+COURSE_TOP_LEFT), (300+COURSE_TOP_LEFT, 80+COURSE_TOP_LEFT), (400+COURSE_TOP_LEFT, 80+COURSE_TOP_LEFT), (200+COURSE_TOP_LEFT, 120+COURSE_TOP_LEFT), (300+COURSE_TOP_LEFT, 120+COURSE_TOP_LEFT), (400+COURSE_TOP_LEFT, 120+COURSE_TOP_LEFT)],
        "bottom": [(200+COURSE_TOP_LEFT, 525+COURSE_TOP_LEFT), (300+COURSE_TOP_LEFT, 525+COURSE_TOP_LEFT), (400+COURSE_TOP_LEFT, 525+COURSE_TOP_LEFT), (200+COURSE_TOP_LEFT, 565+COURSE_TOP_LEFT), (300+COURSE_TOP_LEFT, 565+COURSE_TOP_LEFT), (400+COURSE_TOP_LEFT, 565+COURSE_TOP_LEFT)],
        "left": [(80+COURSE_TOP_LEFT, 200+COURSE_TOP_LEFT), (80+COURSE_TOP_LEFT, 325+COURSE_TOP_LEFT), (80+COURSE_TOP_LEFT, 450+COURSE_TOP_LEFT), (120+COURSE_TOP_LEFT, 200+COURSE_TOP_LEFT), (120+COURSE_TOP_LEFT, 325+COURSE_TOP_LEFT), (120+COURSE_TOP_LEFT, 450+COURSE_TOP_LEFT)],
        "right": [(525+COURSE_TOP_LEFT, 200+COURSE_TOP_LEFT), (525+COURSE_TOP_LEFT, 325+COURSE_TOP_LEFT), (525+COURSE_TOP_LEFT, 450+COURSE_TOP_LEFT), (565+COURSE_TOP_LEFT, 200+COURSE_TOP_LEFT), (565+COURSE_TOP_LEFT, 325+COURSE_TOP_LEFT), (565+COURSE_TOP_LEFT, 450+COURSE_TOP_LEFT)],
    }.items()}
    generated_obstacles, obstacle_walls = [], []
    for _, slots in obstacle_slots.items():
        num_to_spawn = random.choice([0, 1, 1, 2])
        chosen_slots = []
        if num_to_spawn == 1: chosen_slots.append(random.choice(slots))
        elif num_to_spawn == 2: pair = random.choice([(0, 2), (3, 5)]); chosen_slots.extend([slots[pair[0]], slots[pair[1]]])
        for slot_center in chosen_slots:
            rect = pygame.Rect((0, 0), OBSTACLE_SIZE); rect.center = slot_center
            color = random.choice([COLOR_OBSTACLE_RED, COLOR_OBSTACLE_GREEN])
            generated_obstacles.append({'rect': rect, 'color': color})
            tl, tr, bl, br = rect.topleft, rect.topright, rect.bottomleft, rect.bottomright
            obstacle_walls.extend([(tl[0], tl[1], tr[0], tr[1]), (tr[0], tr[1], br[0], br[1]), (br[0], br[1], bl[0], bl[1]), (bl[0], bl[1], tl[0], tl[1])])
    return generated_obstacles, obstacle_walls


class Robot:
    # (Robot implementation omitted for brevity, same as previous version)
    def __init__(self, x, y):
        self.x, self.y = x, y
        self.lidar_points = []
        self.estimated_walls = []
        self.inlier_points_for_viz = []
        self.unassociated_points = [] 

    def set_pos(self, x, y): self.x, self.y = x, y

    def simulate_lidar(self, walls):
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
        data = np.array(points)
        mean = np.mean(data, axis=0)
        _, eigenvectors = np.linalg.eigh(np.cov(data.T))
        direction_vector = eigenvectors[:, -1]
        projections = np.dot(data - mean, direction_vector)
        line_start, line_end = mean + np.min(projections) * direction_vector, mean + np.max(projections) * direction_vector
        return (line_start.tolist(), line_end.tolist())

    def cluster_unassociated_points(self):
        points = self.unassociated_points
        clusters_found = []
        unvisited_points = set(points)
        while unvisited_points:
            queue = [unvisited_points.pop()]
            current_cluster = [queue[0]]
            head = 0
            while head < len(queue):
                current_point = queue[head]; head += 1
                neighbors = [p for p in unvisited_points if math.hypot(current_point[0] - p[0], current_point[1] - p[1]) < OBSTACLE_PROXIMITY]
                for neighbor in neighbors:
                    unvisited_points.remove(neighbor)
                    current_cluster.append(neighbor)
                    queue.append(neighbor)
            if len(current_cluster) >= OBSTACLE_MIN_POINTS:
                center = np.mean(np.array(current_cluster), axis=0)
                radius = max(math.hypot(p[0] - center[0], p[1] - center[1]) for p in current_cluster) + 5
                if radius <= OBSTACLE_MAX_SIZE:
                    clusters_found.append({'center': center.tolist(), 'radius': radius})
        return clusters_found


class Heatmap:
    # (Heatmap implementation omitted for brevity, same as previous version)
    def __init__(self, width, height, cell_size):
        self.cell_size, self.grid_width, self.grid_height = cell_size, width // cell_size, height // cell_size
        self.grid = np.zeros((self.grid_width, self.grid_height), dtype=float)
        self.surface = pygame.Surface((width, height), pygame.SRCALPHA)
        self.max_hits = 1.0

    def add_points(self, points):
        for x, y in points:
            grid_x, grid_y = int(x // self.cell_size), int(y // self.cell_size)
            if 0 <= grid_x < self.grid_width and 0 <= grid_y < self.grid_height: self.grid[grid_x, grid_y] += 1

    def decay(self, rate=0.99):
        self.grid *= rate; self.grid[self.grid < 0.1] = 0

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
        self.potential_obstacles = []
        self.confirmed_obstacles = []

    def reset(self):
        self.potential_obstacles.clear()
        self.confirmed_obstacles.clear()

    # --- MODIFICATION START: Update method to handle wall collisions ---
    def update(self, current_frame_clusters, estimated_walls):
        
        # 1. Invalidate Confirmed Obstacles colliding with walls
        invalidated_confirmed_indices = []
        for i, conf_obs in enumerate(self.confirmed_obstacles):
            if check_obstacle_wall_collision(conf_obs, estimated_walls):
                invalidated_confirmed_indices.append(i)

        # Remove invalidated confirmed obstacles
        for i in sorted(invalidated_confirmed_indices, reverse=True):
            del self.confirmed_obstacles[i]

        # 2. Decay confidence and check for wall collisions in potential obstacles
        for pot_obs in self.potential_obstacles:
            pot_obs['seen_this_frame'] = False
            pot_obs['confidence'] *= OBSTACLE_CONFIDENCE_DECAY
            
            # Check potential obstacles against current walls
            if check_obstacle_wall_collision(pot_obs, estimated_walls):
                # If it hits a wall, reset confidence to zero immediately
                pot_obs['confidence'] = 0

        # 3. Match current detections with potential obstacles
        for cluster in current_frame_clusters:
            # Optimization: Skip this cluster if it already overlaps a wall
            if check_obstacle_wall_collision(cluster, estimated_walls):
                continue
                
            best_match = None
            min_dist = OBSTACLE_MATCHING_DISTANCE
            for pot_obs in self.potential_obstacles:
                dist = math.hypot(cluster['center'][0] - pot_obs['center'][0], cluster['center'][1] - pot_obs['center'][1])
                if dist < min_dist:
                    min_dist = dist
                    best_match = pot_obs
            
            if best_match:
                # Matched: Update confidence and smooth position/radius
                best_match['confidence'] += OBSTACLE_CONFIDENCE_INCREMENT
                w = 0.2 # weight for new measurement
                best_match['center'][0] = best_match['center'][0] * (1-w) + cluster['center'][0] * w
                best_match['center'][1] = best_match['center'][1] * (1-w) + cluster['center'][1] * w
                best_match['radius'] = best_match['radius'] * (1-w) + cluster['radius'] * w
                best_match['seen_this_frame'] = True
            else:
                # No match found: create a new potential obstacle
                self.potential_obstacles.append({
                    'center': cluster['center'],
                    'radius': cluster['radius'],
                    'confidence': OBSTACLE_CONFIDENCE_INCREMENT,
                    'seen_this_frame': True
                })

        # 4. Confirm obstacles and cleanup
        newly_confirmed_indices = []
        for i, pot_obs in enumerate(self.potential_obstacles):
            if pot_obs['confidence'] >= OBSTACLE_CONFIRMATION_THRESHOLD:
                # Check for overlap with already confirmed obstacles
                is_overlapping = False
                for conf_obs in self.confirmed_obstacles:
                    dist = math.hypot(conf_obs['center'][0] - pot_obs['center'][0], conf_obs['center'][1] - pot_obs['center'][1])
                    if dist < (conf_obs['radius'] + pot_obs['radius']):
                        is_overlapping = True
                        break
                
                if not is_overlapping:
                    self.confirmed_obstacles.append(pot_obs)
                    newly_confirmed_indices.append(i)

        # Remove newly confirmed obstacles from the potential list
        for i in sorted(newly_confirmed_indices, reverse=True):
            del self.potential_obstacles[i]
            
        # Remove potential obstacles with very low confidence (including those reset due to wall collision)
        self.potential_obstacles = [p for p in self.potential_obstacles if p['confidence'] > 0.1]
    # --- MODIFICATION END ---

    def draw(self, screen):
        # Draw confirmed obstacles (solid)
        for obs in self.confirmed_obstacles:
            pygame.draw.circle(screen, COLOR_OBSTACLE_DETECTED, obs['center'], obs['radius'], 2)
            
        # Draw potential obstacles (transparent, with confidence-based alpha)
        s = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        for obs in self.potential_obstacles:
            alpha = min(200, 20 + int(obs['confidence'] * 40))
            color = (*COLOR_OBSTACLE_POTENTIAL[:3], alpha)
            pygame.draw.circle(s, color, obs['center'], obs['radius'])
        screen.blit(s, (0,0))


# --- Remaining Helper Functions ---

def distance_from_point_to_line(p, l1, l2):
    x0, y0 = p; x1, y1 = l1; x2, y2 = l2
    num = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)
    den = math.sqrt((y2-y1)**2 + (x2-x1)**2)
    return num/den if den != 0 else 0

def line_intersection(p1, p2, p3, p4):
    x1, y1=p1; x2, y2=p2; x3, y3=p3; x4, y4=p4
    den = (x1-x2)*(y3-y4)-(y1-y2)*(x3-x4)
    if den == 0: return None, float('inf')
    t = ((x1-x3)*(y3-y4)-(y1-y3)*(x3-x4))/den
    u = -((x1-x2)*(y1-y3)-(y1-y2)*(x1-x3))/den
    if 0<t<1 and 0<u<1:
        px, py = x1 + t * (x2 - x1), y1 + t * (y2 - y1)
        return (px, py), math.hypot(px-x1, py-y1)
    return None, float('inf')


def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Persistent Obstacle Detection with Wall Validation")
    clock = pygame.time.Clock()
    font, small_font = pygame.font.Font(None, 28), pygame.font.Font(None, 22)
    robot = Robot(SCREEN_WIDTH/2, SCREEN_HEIGHT/2)
    heatmap = Heatmap(SCREEN_WIDTH, SCREEN_HEIGHT, HEATMAP_CELL_SIZE)
    obstacle_manager = ObstacleManager()

    obstacles_for_drawing, obstacle_walls = generate_obstacles()
    all_walls_for_lidar = COURSE_WALLS + obstacle_walls
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            if event.type == pygame.MOUSEMOTION: robot.set_pos(*event.pos)
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obstacles_for_drawing, obstacle_walls = generate_obstacles()
                all_walls_for_lidar = COURSE_WALLS + obstacle_walls
                obstacle_manager.reset() # Reset confirmed obstacles with the map

        # --- Logic ---
        robot.simulate_lidar(all_walls_for_lidar)
        robot.estimate_walls()
        current_frame_clusters = robot.cluster_unassociated_points()
        
        # --- MODIFICATION START: Pass estimated walls to the manager ---
        obstacle_manager.update(current_frame_clusters, robot.estimated_walls)
        # --- MODIFICATION END ---
        
        heatmap.add_points(robot.lidar_points)
        heatmap.decay(HEATMAP_DECAY_RATE)

        # --- Drawing ---
        screen.fill(COLOR_BACKGROUND)
        heatmap.update_and_draw(screen)
        
        for wall in COURSE_WALLS: pygame.draw.line(screen, COLOR_WALL, (wall[0], wall[1]), (wall[2], wall[3]), 2)
        for obs in obstacles_for_drawing:
            pygame.draw.rect(screen, obs['color'], obs['rect'])
            pygame.draw.rect(screen, COLOR_WALL, obs['rect'], 1)

        for p in robot.inlier_points_for_viz: pygame.draw.circle(screen, COLOR_INLIER_POINTS, p, 3)
        for wall in robot.estimated_walls: pygame.draw.line(screen, COLOR_ESTIMATED_WALL, wall[0], wall[1], 5)
        
        obstacle_manager.draw(screen)
        
        robot_pos = (robot.x, robot.y)
        for wall_segment in robot.estimated_walls:
            p_start, p_end = wall_segment[0], wall_segment[1]
            closest_point = find_closest_point_on_segment(robot_pos, p_start, p_end)
            distance = math.hypot(robot_pos[0] - closest_point[0], robot_pos[1] - closest_point[1])
            
            # (Drawing distance lines - same as before)
            vec_robot_to_point, vec_wall = np.array(closest_point) - np.array(robot_pos), np.array(p_end) - np.array(p_start)
            mag_robot_vec, mag_wall_vec = np.linalg.norm(vec_robot_to_point), np.linalg.norm(vec_wall)
            angle_deg = 90
            if mag_robot_vec > 0 and mag_wall_vec > 0:
                angle_deg = abs(90 - math.degrees(math.acos(np.clip(np.dot(vec_robot_to_point, vec_wall)/(mag_robot_vec*mag_wall_vec),-1.0,1.0))))
            line_color = COLOR_PERPENDICULAR if angle_deg <= PERPENDICULAR_ANGLE_THRESHOLD else COLOR_NON_PERPENDICULAR
            pygame.draw.line(screen, line_color, robot_pos, closest_point, 2)
            text_surface = small_font.render(f"{distance:.0f}", True, COLOR_TEXT)
            mid_point = ((robot_pos[0] + closest_point[0]) / 2, (robot_pos[1] + closest_point[1]) / 2)
            text_rect = text_surface.get_rect(center=mid_point)
            pygame.draw.rect(screen, COLOR_BACKGROUND, text_rect.inflate(6, 4))
            screen.blit(text_surface, text_rect)
        
        pygame.draw.circle(screen, COLOR_ROBOT, robot_pos, ROBOT_SIZE)
        pygame.draw.circle(screen, (255,255,255), robot_pos, ROBOT_SIZE, 1)

        instruction_text = font.render("Move mouse to control robot. Press 'R' for new obstacles.", True, COLOR_TEXT)
        screen.blit(instruction_text, (10, 10))

        pygame.display.flip()
        clock.tick(60)
    pygame.quit()

if __name__ == '__main__':
    main()