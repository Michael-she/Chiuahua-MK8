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
# --- NEW COLORS ---
COLOR_PERPENDICULAR = (0, 170, 255)       # Light Blue for perpendicular lines
COLOR_NON_PERPENDICULAR = (200, 160, 255) # Light Purple for non-perpendicular

# Robot properties
ROBOT_SIZE = 15

# LiDAR properties
LIDAR_RANGE = 3000
LIDAR_RAYS = 360
LIDAR_INACCURACY = 8.0

# RANSAC & Clustering Parameters
RANSAC_ITERATIONS = 30
RANSAC_THRESHOLD_DISTANCE = 14.0
RANSAC_MIN_INLIERS = 20
MAX_POINT_DISTANCE = 50.0
# --- NEW ANGLE THRESHOLD ---
PERPENDICULAR_ANGLE_THRESHOLD = 5.0 # Degrees

# --- Course Definition ---
COURSE_TOP_LEFT = 50
WALLS = [
    # outer boundary
    (0  + COURSE_TOP_LEFT, 0  + COURSE_TOP_LEFT,
     600 + COURSE_TOP_LEFT, 0  + COURSE_TOP_LEFT),
    (600 + COURSE_TOP_LEFT, 0  + COURSE_TOP_LEFT,
     600 + COURSE_TOP_LEFT, 600 + COURSE_TOP_LEFT),
    (600 + COURSE_TOP_LEFT, 600 + COURSE_TOP_LEFT,
     0   + COURSE_TOP_LEFT, 600 + COURSE_TOP_LEFT),
    (0   + COURSE_TOP_LEFT, 600 + COURSE_TOP_LEFT,
     0   + COURSE_TOP_LEFT, 0   + COURSE_TOP_LEFT),

    # inner square
    (200 + COURSE_TOP_LEFT, 200 + COURSE_TOP_LEFT,
     400 + COURSE_TOP_LEFT, 200 + COURSE_TOP_LEFT),
    (400 + COURSE_TOP_LEFT, 200 + COURSE_TOP_LEFT,
     400 + COURSE_TOP_LEFT, 400 + COURSE_TOP_LEFT),
    (400 + COURSE_TOP_LEFT, 400 + COURSE_TOP_LEFT,
     200 + COURSE_TOP_LEFT, 400 + COURSE_TOP_LEFT),
    (200 + COURSE_TOP_LEFT, 400 + COURSE_TOP_LEFT,
     200 + COURSE_TOP_LEFT, 200 + COURSE_TOP_LEFT),

    # obstacle Top Left
    (195 + COURSE_TOP_LEFT,  80 + COURSE_TOP_LEFT,
     205 + COURSE_TOP_LEFT,  80 + COURSE_TOP_LEFT),
    (195 + COURSE_TOP_LEFT,  80 + COURSE_TOP_LEFT,
     195 + COURSE_TOP_LEFT,  90 + COURSE_TOP_LEFT),
    (205 + COURSE_TOP_LEFT,  80 + COURSE_TOP_LEFT,
     205 + COURSE_TOP_LEFT,  90 + COURSE_TOP_LEFT),
    (195 + COURSE_TOP_LEFT,  90 + COURSE_TOP_LEFT,
     205 + COURSE_TOP_LEFT,  90 + COURSE_TOP_LEFT),

     (195 + COURSE_TOP_LEFT,  120 + COURSE_TOP_LEFT,
     205 + COURSE_TOP_LEFT,  120 + COURSE_TOP_LEFT),
    (195 + COURSE_TOP_LEFT,  120 + COURSE_TOP_LEFT,
     195 + COURSE_TOP_LEFT,  130 + COURSE_TOP_LEFT),
    (205 + COURSE_TOP_LEFT,  120 + COURSE_TOP_LEFT,
     205 + COURSE_TOP_LEFT,  130 + COURSE_TOP_LEFT),
    (195 + COURSE_TOP_LEFT,  130 + COURSE_TOP_LEFT,
     205 + COURSE_TOP_LEFT,  130 + COURSE_TOP_LEFT),

     # obstacle Top Mid
    (295 + COURSE_TOP_LEFT,  80 + COURSE_TOP_LEFT,
    305 + COURSE_TOP_LEFT,  80 + COURSE_TOP_LEFT),
    (295 + COURSE_TOP_LEFT,  80 + COURSE_TOP_LEFT,
     295 + COURSE_TOP_LEFT,  90 + COURSE_TOP_LEFT),
    (305 + COURSE_TOP_LEFT,  80 + COURSE_TOP_LEFT,
     305 + COURSE_TOP_LEFT,  90 + COURSE_TOP_LEFT),
    (295 + COURSE_TOP_LEFT,  90 + COURSE_TOP_LEFT,
     305 + COURSE_TOP_LEFT,  90 + COURSE_TOP_LEFT),

    (295 + COURSE_TOP_LEFT,  120 + COURSE_TOP_LEFT,
     305 + COURSE_TOP_LEFT,  120 + COURSE_TOP_LEFT),
    (295 + COURSE_TOP_LEFT,  120 + COURSE_TOP_LEFT,
     295 + COURSE_TOP_LEFT,  130 + COURSE_TOP_LEFT),
    (305 + COURSE_TOP_LEFT,  120 + COURSE_TOP_LEFT,
     305 + COURSE_TOP_LEFT,  130 + COURSE_TOP_LEFT),
    (295 + COURSE_TOP_LEFT,  130 + COURSE_TOP_LEFT,
     305 + COURSE_TOP_LEFT,  130 + COURSE_TOP_LEFT),

    # obstacle Top Right
    (395 + COURSE_TOP_LEFT,  80 + COURSE_TOP_LEFT, 
     405 + COURSE_TOP_LEFT,  80 + COURSE_TOP_LEFT),
    (395 + COURSE_TOP_LEFT,  80 + COURSE_TOP_LEFT,
     395 + COURSE_TOP_LEFT,  90 + COURSE_TOP_LEFT),
    (405 + COURSE_TOP_LEFT,  80 + COURSE_TOP_LEFT,
     405 + COURSE_TOP_LEFT,  90 + COURSE_TOP_LEFT),
    (395 + COURSE_TOP_LEFT,  90 + COURSE_TOP_LEFT,
     405 + COURSE_TOP_LEFT,  90 + COURSE_TOP_LEFT),

    (395 + COURSE_TOP_LEFT,  120 + COURSE_TOP_LEFT,
     405 + COURSE_TOP_LEFT,  120 + COURSE_TOP_LEFT),
    (395 + COURSE_TOP_LEFT,  120 + COURSE_TOP_LEFT,
     395 + COURSE_TOP_LEFT,  130 + COURSE_TOP_LEFT),
    (405 + COURSE_TOP_LEFT,  120 + COURSE_TOP_LEFT,
     405 + COURSE_TOP_LEFT,  130 + COURSE_TOP_LEFT),
    (395 + COURSE_TOP_LEFT,  130 + COURSE_TOP_LEFT,
     405 + COURSE_TOP_LEFT,  130 + COURSE_TOP_LEFT),

     


     
]

class Robot:
    def __init__(self, x, y):
        self.x, self.y = x, y
        self.lidar_points = []
        self.estimated_walls = []
        self.inlier_points_for_viz = []

    def set_pos(self, x, y):
        self.x, self.y = x, y

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
                            self.estimated_walls.append(final_wall_segment)
                            self.inlier_points_for_viz.extend(cluster)
                        for point in cluster: points_to_remove.add(point)
                if not points_to_remove: break
                remaining_points = [p for p in remaining_points if p not in points_to_remove]
            else: break

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
        direction_vector = np.linalg.eigh(np.cov(data - mean, rowvar=False))[1][:, -1]
        t_start, t_end = np.dot(np.array(points[0]) - mean, direction_vector), np.dot(np.array(points[-1]) - mean, direction_vector)
        line_start, line_end = mean + t_start * direction_vector, mean + t_end * direction_vector
        return (line_start.tolist(), line_end.tolist())

# --- Helper Functions ---
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
    pygame.display.set_caption("Wall Estimation with Perpendicularity Check")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 28)
    small_font = pygame.font.Font(None, 22)
    robot = Robot(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2)
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            if event.type == pygame.MOUSEMOTION: robot.set_pos(*event.pos)

        robot.simulate_lidar(WALLS)
        robot.estimate_walls()

        screen.fill(COLOR_BACKGROUND)
        for wall in WALLS: pygame.draw.line(screen, COLOR_WALL, (wall[0], wall[1]), (wall[2], wall[3]), 2)
        
        for p in robot.lidar_points: pygame.draw.circle(screen, (50,50,50), p, 2)
        for p in robot.inlier_points_for_viz: pygame.draw.circle(screen, COLOR_INLIER_POINTS, p, 3)
        for wall in robot.estimated_walls: pygame.draw.line(screen, COLOR_ESTIMATED_WALL, wall[0], wall[1], 5)
        
        robot_pos = (robot.x, robot.y)
        for wall_segment in robot.estimated_walls:
            # --- NEW LOGIC FOR CONDITIONAL COLORING ---
            
            # 1. Find the closest point on the wall segment
            p_start, p_end = wall_segment[0], wall_segment[1]
            closest_point = find_closest_point_on_segment(robot_pos, p_start, p_end)
            distance = math.hypot(robot_pos[0] - closest_point[0], robot_pos[1] - closest_point[1])
            
            # 2. Define the vectors
            vec_robot_to_point = np.array(closest_point) - np.array(robot_pos)
            vec_wall = np.array(p_end) - np.array(p_start)
            
            # 3. Calculate the angle between them
            angle_deg = 90 # Default to non-perpendicular if vectors are zero
            mag_robot_vec = np.linalg.norm(vec_robot_to_point)
            mag_wall_vec = np.linalg.norm(vec_wall)
            
            if mag_robot_vec > 0 and mag_wall_vec > 0:
                # Find angle between robot-to-point and the wall itself
                dot_product = np.dot(vec_robot_to_point, vec_wall)
                cos_theta = dot_product / (mag_robot_vec * mag_wall_vec)
                cos_theta = np.clip(cos_theta, -1.0, 1.0)
                angle_deg = abs(90 - math.degrees(math.acos(cos_theta)))

            # 4. Choose color based on the angle
            line_color = COLOR_PERPENDICULAR if angle_deg <= PERPENDICULAR_ANGLE_THRESHOLD else COLOR_NON_PERPENDICULAR
            
            # 5. Draw the line and text
            pygame.draw.line(screen, line_color, robot_pos, closest_point, 2)
            
            dist_text_str = f"{distance:.0f}"
            text_surface = small_font.render(dist_text_str, True, COLOR_TEXT)
            mid_point = ((robot_pos[0] + closest_point[0]) / 2, (robot_pos[1] + closest_point[1]) / 2)
            text_rect = text_surface.get_rect(center=mid_point)
            pygame.draw.rect(screen, COLOR_BACKGROUND, text_rect.inflate(6, 4))
            screen.blit(text_surface, text_rect)
        
        pygame.draw.circle(screen, COLOR_ROBOT, robot_pos, ROBOT_SIZE)
        pygame.draw.circle(screen, (255,255,255), robot_pos, ROBOT_SIZE, 1)

        instruction_text = font.render("Blue: Perpendicular distance. Purple: Distance to corner.", True, COLOR_TEXT)
        screen.blit(instruction_text, (10, 10))

        pygame.display.flip()
        clock.tick(60)
    pygame.quit()

if __name__ == '__main__':
    main()