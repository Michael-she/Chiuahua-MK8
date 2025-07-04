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
MIN_WALL_LENGTH = 40.0  # <<< NEW: Discard detected walls shorter than this length (in pixels)

# --- Course & Obstacle Definition ---
COURSE_TOP_LEFT = 50
OBSTACLE_SIZE = (10, 10) # Width, Height for new obstacles
COURSE_WALLS = [
    # outer boundary
    (0  + COURSE_TOP_LEFT, 0  + COURSE_TOP_LEFT, 600 + COURSE_TOP_LEFT, 0  + COURSE_TOP_LEFT),
    (600 + COURSE_TOP_LEFT, 0  + COURSE_TOP_LEFT, 600 + COURSE_TOP_LEFT, 600 + COURSE_TOP_LEFT),
    (600 + COURSE_TOP_LEFT, 600 + COURSE_TOP_LEFT, 0   + COURSE_TOP_LEFT, 600 + COURSE_TOP_LEFT),
    (0   + COURSE_TOP_LEFT, 600 + COURSE_TOP_LEFT, 0   + COURSE_TOP_LEFT, 0   + COURSE_TOP_LEFT),
    # inner square
    (200 + COURSE_TOP_LEFT, 200 + COURSE_TOP_LEFT, 400 + COURSE_TOP_LEFT, 200 + COURSE_TOP_LEFT),
    (400 + COURSE_TOP_LEFT, 200 + COURSE_TOP_LEFT, 400 + COURSE_TOP_LEFT, 400 + COURSE_TOP_LEFT),
    (400 + COURSE_TOP_LEFT, 400 + COURSE_TOP_LEFT, 200 + COURSE_TOP_LEFT, 400 + COURSE_TOP_LEFT),
    (200 + COURSE_TOP_LEFT, 400 + COURSE_TOP_LEFT, 200 + COURSE_TOP_LEFT, 200 + COURSE_TOP_LEFT),
]

def generate_obstacles():
    """Generates obstacles according to the specified rules."""
    obstacle_slots = {
        "top": [
            (200 + COURSE_TOP_LEFT,  80 + COURSE_TOP_LEFT),
            (300 + COURSE_TOP_LEFT,  80 + COURSE_TOP_LEFT),
            (400 + COURSE_TOP_LEFT,  80 + COURSE_TOP_LEFT),
            (200 + COURSE_TOP_LEFT, 120 + COURSE_TOP_LEFT),
            (300 + COURSE_TOP_LEFT, 120 + COURSE_TOP_LEFT),
            (400 + COURSE_TOP_LEFT, 120 + COURSE_TOP_LEFT),
        ],
        "bottom": [
            (200 + COURSE_TOP_LEFT, 525 + COURSE_TOP_LEFT),
            (300 + COURSE_TOP_LEFT, 525 + COURSE_TOP_LEFT),
            (400 + COURSE_TOP_LEFT, 525 + COURSE_TOP_LEFT),
            (200 + COURSE_TOP_LEFT, 565 + COURSE_TOP_LEFT),
            (300 + COURSE_TOP_LEFT, 565 + COURSE_TOP_LEFT),
            (400 + COURSE_TOP_LEFT, 565 + COURSE_TOP_LEFT),
        ],
        "left": [
            ( 80 + COURSE_TOP_LEFT, 200 + COURSE_TOP_LEFT),
            ( 80 + COURSE_TOP_LEFT, 325 + COURSE_TOP_LEFT),
            ( 80 + COURSE_TOP_LEFT, 450 + COURSE_TOP_LEFT),
            (120 + COURSE_TOP_LEFT, 200 + COURSE_TOP_LEFT),
            (120 + COURSE_TOP_LEFT, 325 + COURSE_TOP_LEFT),
            (120 + COURSE_TOP_LEFT, 450 + COURSE_TOP_LEFT),
        ],
        "right": [
            (525 + COURSE_TOP_LEFT, 200 + COURSE_TOP_LEFT),
            (525 + COURSE_TOP_LEFT, 325 + COURSE_TOP_LEFT),
            (525 + COURSE_TOP_LEFT, 450 + COURSE_TOP_LEFT),
            (565 + COURSE_TOP_LEFT, 200 + COURSE_TOP_LEFT),
            (565 + COURSE_TOP_LEFT, 325 + COURSE_TOP_LEFT),
            (565 + COURSE_TOP_LEFT, 450 + COURSE_TOP_LEFT),
        ],
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
            obstacle_walls.extend([
                (tl[0], tl[1], tr[0], tr[1]), (tr[0], tr[1], br[0], br[1]),
                (br[0], br[1], bl[0], bl[1]), (bl[0], bl[1], tl[0], tl[1])
            ])
    return generated_obstacles, obstacle_walls

class Robot:
    def __init__(self, x, y):
        self.x, self.y = x, y
        self.lidar_points, self.estimated_walls, self.inlier_points_for_viz = [], [], []

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

    # <<< METHOD MODIFIED TO FILTER WALLS BY LENGTH >>>
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
                        
                        # <<< MODIFIED SECTION START >>>
                        if final_wall_segment:
                            # Calculate the length of the detected wall segment
                            p_start, p_end = final_wall_segment[0], final_wall_segment[1]
                            length = math.hypot(p_end[0] - p_start[0], p_end[1] - p_start[1])

                            # Only add the wall if it's longer than the minimum required length
                            if length >= MIN_WALL_LENGTH:
                                self.estimated_walls.append(final_wall_segment)
                                self.inlier_points_for_viz.extend(cluster)
                        # <<< MODIFIED SECTION END >>>
                        
                        # Always remove the points from this cluster so we don't re-process them
                        for point in cluster:
                            points_to_remove.add(point)

                if not points_to_remove:
                    break
                remaining_points = [p for p in remaining_points if p not in points_to_remove]
            else:
                break

    def cluster_inliers_by_distance(self, inliers):
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
        if len(points) < 2: return None
        data = np.array(points)
        mean = np.mean(data, axis=0)
        # Use np.linalg.eigh for symmetric matrices like the covariance matrix
        eigenvalues, eigenvectors = np.linalg.eigh(np.cov(data.T))
        direction_vector = eigenvectors[:, -1] # Principal component is eigenvector with largest eigenvalue
        
        # Project points onto the principal component vector to find the extent
        projections = np.dot(data - mean, direction_vector)
        t_min, t_max = np.min(projections), np.max(projections)
        
        line_start = mean + t_min * direction_vector
        line_end = mean + t_max * direction_vector
        return (line_start.tolist(), line_end.tolist())

# Helper Functions
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
    pygame.display.set_caption("Wall Estimation with Random Obstacles")
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
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obstacles_for_drawing, obstacle_walls = generate_obstacles()
                all_walls_for_lidar = COURSE_WALLS + obstacle_walls

        robot.simulate_lidar(all_walls_for_lidar)
        robot.estimate_walls()

        screen.fill(COLOR_BACKGROUND)

        for wall in COURSE_WALLS: pygame.draw.line(screen, COLOR_WALL, (wall[0], wall[1]), (wall[2], wall[3]), 2)
        for obs in obstacles_for_drawing:
            pygame.draw.rect(screen, obs['color'], obs['rect'])
            pygame.draw.rect(screen, COLOR_WALL, obs['rect'], 1)

        for p in robot.lidar_points: pygame.draw.circle(screen, (50,50,50), p, 2)
        for p in robot.inlier_points_for_viz: pygame.draw.circle(screen, COLOR_INLIER_POINTS, p, 3)
        for wall in robot.estimated_walls: pygame.draw.line(screen, COLOR_ESTIMATED_WALL, wall[0], wall[1], 5)

        robot_pos = (robot.x, robot.y)
        for wall_segment in robot.estimated_walls:
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

        instruction_text = font.render("Move mouse to control robot. Press 'R' for new obstacles.", True, COLOR_TEXT)
        screen.blit(instruction_text, (10, 10))

        pygame.display.flip()
        clock.tick(60)
    pygame.quit()

if __name__ == '__main__':
    main()