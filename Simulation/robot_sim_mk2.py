import pygame
import numpy as np
import random
import math

# --- Configuration ---
# Screen
SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 800

# Colors
COLOR_BACKGROUND = (10, 10, 10)
COLOR_WALL = (200, 200, 200)
COLOR_ROBOT = (255, 0, 0)
COLOR_LIDAR_POINT = (0, 180, 0)
COLOR_ESTIMATED_WALL = (100, 255, 100)
COLOR_INLIER_POINTS = (0, 100, 255) # Blue for inliers
COLOR_TEXT = (255, 255, 255)

# Robot properties
ROBOT_SIZE = 15

# LiDAR properties
LIDAR_RANGE = 3000
LIDAR_RAYS = 360
LIDAR_INACCURACY = 2.0

# --- RANSAC & Clustering Parameters ---
RANSAC_ITERATIONS = 80
RANSAC_THRESHOLD_DISTANCE = 5.0
RANSAC_MIN_INLIERS = 10

# --- NEW TUNABLE VARIABLE ---
# After RANSAC finds points on a line, this is the max distance between
# two of those points for them to be considered part of the same contiguous wall.
MAX_POINT_DISTANCE = 50.0

# --- Course Definition ---
WALLS = [
    (50, 50, 750, 50), (750, 50, 750, 750), (750, 750, 50, 750), (50, 750, 50, 50),
    (250, 250, 550, 250), (550, 250, 550, 550), (550, 550, 250, 550), (250, 550, 250, 250),
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
        # This function is unchanged
        self.lidar_points = []
        for i in range(LIDAR_RAYS):
            ray_angle_rad = math.radians((360 / LIDAR_RAYS) * i)
            end_x = self.x + LIDAR_RANGE * math.cos(ray_angle_rad)
            end_y = self.y + LIDAR_RANGE * math.sin(ray_angle_rad)
            closest_dist, hit_point = LIDAR_RANGE, None
            for wall in walls:
                p, d = line_intersection((self.x, self.y), (end_x, end_y), (wall[0], wall[1]), (wall[2], wall[3]))
                if p and d < closest_dist:
                    closest_dist, hit_point = d, p
            if hit_point:
                dist_with_error = closest_dist + random.uniform(-LIDAR_INACCURACY, LIDAR_INACCURACY)
                self.lidar_points.append((self.x + dist_with_error * math.cos(ray_angle_rad), self.y + dist_with_error * math.sin(ray_angle_rad)))

    def estimate_walls(self):
        """Processes LiDAR points using RANSAC followed by a contiguity check."""
        self.estimated_walls = []
        self.inlier_points_for_viz = []
        
        remaining_points = list(self.lidar_points)
        
        while len(remaining_points) > RANSAC_MIN_INLIERS:
            best_inliers = []
            
            for _ in range(RANSAC_ITERATIONS):
                if len(remaining_points) < 2: break
                p1, p2 = random.sample(remaining_points, 2)
                current_inliers = [p for p in remaining_points if distance_from_point_to_line(p, p1, p2) < RANSAC_THRESHOLD_DISTANCE]
                if len(current_inliers) > len(best_inliers):
                    best_inliers = current_inliers

            if len(best_inliers) > RANSAC_MIN_INLIERS:
                # --- NEW: Contiguity Check Step ---
                # Cluster the RANSAC inliers by distance to find contiguous segments
                contiguous_clusters = self.cluster_inliers_by_distance(best_inliers)
                
                points_to_remove = set()
                for cluster in contiguous_clusters:
                    if len(cluster) > RANSAC_MIN_INLIERS:
                        # This cluster is a valid wall segment
                        final_wall_segment = self.fit_line_with_pca(cluster)
                        if final_wall_segment:
                            self.estimated_walls.append(final_wall_segment)
                            self.inlier_points_for_viz.extend(cluster)
                        
                        # Mark these points for removal from the main pool
                        for point in cluster:
                            points_to_remove.add(point)
                
                if not points_to_remove: # No valid segments found, break loop
                    break
                    
                remaining_points = [p for p in remaining_points if p not in points_to_remove]
            else:
                break # No more significant lines found

    def cluster_inliers_by_distance(self, inliers):
        """Takes a set of inliers and splits them into contiguous groups."""
        if not inliers: return []
        
        # Project points onto their principal component axis and sort them
        data = np.array(inliers)
        mean = np.mean(data, axis=0)
        direction_vector = np.linalg.eigh(np.cov(data - mean, rowvar=False))[1][:, -1]
        
        # Create a list of (projection_value, original_point)
        projected = [(np.dot(p - mean, direction_vector), p) for p in inliers]
        projected.sort(key=lambda x: x[0])
        
        sorted_inliers = [p for _, p in projected]
        
        # Now cluster the sorted points by distance
        clusters = []
        current_cluster = [sorted_inliers[0]]
        
        for i in range(1, len(sorted_inliers)):
            p1 = sorted_inliers[i-1]
            p2 = sorted_inliers[i]
            dist = math.hypot(p1[0] - p2[0], p1[1] - p2[1])
            
            if dist < MAX_POINT_DISTANCE:
                current_cluster.append(p2)
            else:
                clusters.append(current_cluster)
                current_cluster = [p2]
        clusters.append(current_cluster)
        
        return clusters

    def fit_line_with_pca(self, points):
        # This function is unchanged
        if len(points) < 2: return None
        data = np.array(points)
        mean = np.mean(data, axis=0)
        direction_vector = np.linalg.eigh(np.cov(data - mean, rowvar=False))[1][:, -1]
        t_start = np.dot(np.array(points[0]) - mean, direction_vector)
        t_end = np.dot(np.array(points[-1]) - mean, direction_vector)
        line_start = mean + t_start * direction_vector
        line_end = mean + t_end * direction_vector
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

def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Wall Estimation using Hybrid RANSAC + Contiguity Check")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 28)
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
        
        # Draw all lidar points in a faded color
        for p in robot.lidar_points:
            pygame.draw.circle(screen, (50,50,50), p, 2)
        
        # Highlight the inlier points that belong to a final, contiguous wall
        for p in robot.inlier_points_for_viz:
            pygame.draw.circle(screen, COLOR_INLIER_POINTS, p, 3)

        for wall in robot.estimated_walls: pygame.draw.line(screen, COLOR_ESTIMATED_WALL, wall[0], wall[1], 5)
        
        pygame.draw.circle(screen, COLOR_ROBOT, (robot.x, robot.y), ROBOT_SIZE)
        pygame.draw.circle(screen, (255,255,255), (robot.x, robot.y), ROBOT_SIZE, 1)

        instruction_text = font.render("Move mouse to control robot. Blue dots are final, contiguous inliers.", True, COLOR_TEXT)
        screen.blit(instruction_text, (10, 10))

        pygame.display.flip()
        clock.tick(60)
    pygame.quit()

if __name__ == '__main__':
    main()