import pygame
import math
import random
import numpy as np
from enum import Enum

# --- Configuration Constants ---
WIDTH, HEIGHT = 800, 800
FPS = 60
BLACK, WHITE, RED, GREEN, BLUE, YELLOW, CYAN, MAGENTA, ORANGE = (0,0,0),(255,255,255),(255,0,0),(0,255,0),(0,0,255),(255,255,0),(0,255,255),(255,0,255),(255,165,0)
ROBOT_SIZE, MAX_SPEED, MAX_REVERSE_SPEED, MAX_STEERING = 15, 3.0, -1.5, math.radians(40)
LIDAR_RANGE, LIDAR_RAYS, LIDAR_INACCURACY = 1200, 180, 0.015
HEADING_ACCURACY_DEG = 5.0

# --- Wall & Course Info ---
WALLS = [((50,50),(750,50)),((750,50),(750,750)),((750,750),(50,750)),((50,750),(50,50)),((250,250),(550,250)),((550,250),(550,550)),((550,550),(250,550)),((250,550),(250,250))]
def get_line_eq(p1,p2): A=p2[1]-p1[1]; B=p1[0]-p2[0]; C=-A*p1[0]-B*p1[1]; return A,B,C
WALL_EQUATIONS = [get_line_eq(p1, p2) for p1, p2 in WALLS]

# --- Helper, Robot, Planner, PID classes ---
def normalize_angle(a): return(a+math.pi)%(2*math.pi)-math.pi
class Robot: # Unchanged
    def __init__(self,x,y,t): self.x,self.y,self.theta,self.speed,self.steering_angle,self.wheelbase = x,y,t,0,0,30
    def update_pose(self,dt):
        if dt==0:return
        self.x+=self.speed*math.cos(self.theta)*dt; self.y+=self.speed*math.sin(self.theta)*dt
        self.theta = normalize_angle(self.theta + (self.speed/self.wheelbase)*math.tan(self.steering_angle)*dt)
    def get_lidar_scan(self,walls):
        s=[]; [s.append(self._scan_ray(self.theta-math.radians(LIDAR_RAYS/2)+math.radians(i/LIDAR_RAYS * 180),walls)) for i in range(LIDAR_RAYS)]; return s
    def _scan_ray(self,angle,walls):
        re=(self.x+LIDAR_RANGE*math.cos(angle),self.y+LIDAR_RANGE*math.sin(angle)); md=LIDAR_RANGE
        for w in walls:
            d=self._get_intersect(w[0],w[1],(self.x,self.y),re)
            if d and d<md: md=d
        return md*(1+random.uniform(-LIDAR_INACCURACY,LIDAR_INACCURACY))
    def _get_intersect(self,p1,p2,p3,p4):
        d=(p1[0]-p2[0])*(p3[1]-p4[1])-(p1[1]-p2[1])*(p3[0]-p4[0])
        if d==0:return None
        t=(((p1[0]-p3[0])*(p3[1]-p4[1])-(p1[1]-p3[1])*(p3[0]-p4[0]))/d)
        u=(-((p1[0]-p2[0])*(p1[1]-p3[1])-(p1[1]-p2[1])*(p1[0]-p3[0]))/d)
        if 0<t<1 and u>0:return math.hypot((p1[0]+t*(p2[0]-p1[0]))-p3[0],(p1[1]+t*(p2[1]-p1[1]))-p3[1])
        return None
    def get_noisy_heading(self): return normalize_angle(self.theta + math.radians(random.uniform(-HEADING_ACCURACY_DEG, HEADING_ACCURACY_DEG)))

class PlannerState(Enum): IDLE,DRIVING_TO_PRE_TARGET,TURNING_FOR_APPROACH,FINAL_APPROACH=0,1,2,3
class Planner: # Unchanged
    def __init__(self): self.state=PlannerState.IDLE;self.waypoints,self.final_target_pose=[],None
    def set_target(self,tp): self.final_target_pose=tp;tx,ty,tt=tp; self.waypoints=[(tx-100*math.cos(tt),ty-100*math.sin(tt)),(tx,ty)];self.state=PlannerState.DRIVING_TO_PRE_TARGET
    def update(self,erp):
        if self.state==PlannerState.IDLE:return 0,0
        rx,ry,rt=erp
        if self.state==PlannerState.DRIVING_TO_PRE_TARGET:
            tp=self.waypoints[0]
            if math.hypot(tp[0]-rx,tp[1]-ry)<20:self.state=PlannerState.TURNING_FOR_APPROACH;return 0,0
            return 3.0,normalize_angle(math.atan2(tp[1]-ry,tp[0]-rx)-rt)
        elif self.state==PlannerState.TURNING_FOR_APPROACH:
            ta=self.final_target_pose[2];ae=normalize_angle(ta-rt)
            if abs(ae)<math.radians(5):self.state=PlannerState.FINAL_APPROACH;return 0,0
            return 0,ae*2.0
        elif self.state==PlannerState.FINAL_APPROACH:
            tp=self.waypoints[1]
            if math.hypot(tp[0]-rx,tp[1]-ry)<20:self.state=PlannerState.IDLE;return 0,0
            att=math.atan2(tp[1]-ry,tp[0]-rx);s=-1.5 if abs(normalize_angle(rt-att))>math.pi/2 else 3.0
            return s,normalize_angle(att-rt) if s>0 else normalize_angle(rt-att)
        return 0,0
class PIDController: # Unchanged
    def __init__(self,Kp,Ki,Kd):self.Kp,self.Ki,self.Kd=Kp,Ki,Kd;self.pe,self.i=0,0
    def update(self,e,dt): self.i+=e*dt;d=(e-self.pe)/dt if dt>0 else 0;o=self.Kp*e+self.Ki*self.i+self.Kd*d;self.pe=e;return o

# --- UPDATED LOCALIZATION SYSTEM with "Anti-Teleport" Filtering ---
class LocalizationSystem:
    def __init__(self, initial_pose):
        self.estimated_pose = list(initial_pose); self.wheelbase = 30
        self.last_fix_type = "None"; self.last_inliers = []
        # Weights for blending prediction and correction. Higher = more trust in LiDAR.
        self.CORNER_FIX_WEIGHT = 0.8
        self.CORRIDOR_FIX_WEIGHT = 0.3

    def predict(self, speed, steering_angle, dt):
        ex, ey, et = self.estimated_pose
        ex += speed * math.cos(et) * dt; ey += speed * math.sin(et) * dt
        et = normalize_angle(et + (speed / self.wheelbase) * math.tan(steering_angle) * dt)
        self.estimated_pose = [ex, ey, et]

    def correct(self, scan_data, noisy_robot_theta):
        lines = self._ransac_detect_lines(scan_data, 4, 5, 5.0)
        self.last_inliers = [line['inliers'] for line in lines]

        # ... line matching logic is the same ...
        matched_walls = []
        for line_data in lines:
            world_angle_est = normalize_angle(noisy_robot_theta + line_data['angle'])
            best_match_idx, min_angle_diff = -1, float('inf')
            for i, wall_eq in enumerate(WALL_EQUATIONS):
                wall_angle = math.atan2(wall_eq[0], -wall_eq[1])
                angle_diff = abs(normalize_angle(world_angle_est - wall_angle))
                if angle_diff < min_angle_diff and angle_diff < math.radians(25):
                    min_angle_diff = angle_diff; best_match_idx = i
            if best_match_idx != -1: matched_walls.append({'wall_idx': best_match_idx, 'dist': line_data['dist'], 'angle': line_data['angle']})
        
        is_corner = False
        if len(matched_walls) >= 2:
            wall_angles = [math.atan2(WALL_EQUATIONS[m['wall_idx']][0],-WALL_EQUATIONS[m['wall_idx']][1]) for m in matched_walls]
            for i in range(len(wall_angles)):
                for j in range(i+1,len(wall_angles)):
                    if abs(abs(normalize_angle(wall_angles[i]-wall_angles[j]))-math.pi/2)<math.radians(15): is_corner=True;break
                if is_corner:break
        
        # Determine the fix and apply it with filtering
        lidar_fix_pose, weight = None, 0
        if is_corner and len(matched_walls) >= 2:
            self.last_fix_type = "Corner Fix"
            # lidar_fix_pose = self._calculate_corner_fix(matched_walls)
            weight = self.CORNER_FIX_WEIGHT
        elif len(matched_walls) > 0:
            self.last_fix_type = "Corridor Fix"
            lidar_fix_pose = self._calculate_corridor_fix(matched_walls)
            weight = self.CORRIDOR_FIX_WEIGHT
        else:
            self.last_fix_type = "Dead Reckoning Only"

        # --- "ANTI-TELEPORT" FILTERING LOGIC ---
        if lidar_fix_pose and weight > 0:
            pred_x, pred_y, pred_theta = self.estimated_pose
            fix_x, fix_y, fix_theta = lidar_fix_pose

            # Blend positions (simple linear interpolation)
            final_x = (1 - weight) * pred_x + weight * fix_x if fix_x is not None else pred_x
            final_y = (1 - weight) * pred_y + weight * fix_y if fix_y is not None else pred_y
            
            # Blend angle carefully to handle wrapping around 2*pi
            if fix_theta is not None:
                angle_diff = normalize_angle(fix_theta - pred_theta)
                final_theta = normalize_angle(pred_theta + weight * angle_diff)
            else:
                final_theta = pred_theta
            
            self.estimated_pose = [final_x, final_y, final_theta]

    # These methods now RETURN a pose instead of setting it directly
    def _calculate_corner_fix(self, matched_walls):
        eqs=[]
        for match in matched_walls:
            A,B,C=WALL_EQUATIONS[match['wall_idx']];d=match['dist'];norm=math.sqrt(A**2+B**2)
            ex,ey,_=self.estimated_pose;c1=C-d*norm;c2=C+d*norm
            dist1=abs(A*ex+B*ey+c1);dist2=abs(A*ex+B*ey+c2)
            eqs.append((A,B,c1 if dist1<dist2 else c2))
        if len(eqs)<2: return None
        A1,B1,C1=eqs[0];A2,B2,C2=eqs[1];det=A1*B2-A2*B1
        if abs(det)<1e-6: return None
        x=(B1*C2-B2*C1)/det;y=(A2*C1-A1*C2)/det
        world_angles=[normalize_angle(math.atan2(WALL_EQUATIONS[m['wall_idx']][0],-WALL_EQUATIONS[m['wall_idx']][1])-m['angle']) for m in matched_walls]
        return (x, y, np.mean(world_angles))

    def _calculate_corridor_fix(self, matched_walls):
        world_angles=[normalize_angle(math.atan2(WALL_EQUATIONS[m['wall_idx']][0],-WALL_EQUATIONS[m['wall_idx']][1])-m['angle']) for m in matched_walls]
        # Corridor fix can only determine theta reliably, not x and y
        return (None, None, np.mean(world_angles))

    def _ransac_detect_lines(self, scan_data, num_lines, min_points, threshold): # Unchanged
        points=[]; [points.append((d*math.cos(-math.radians(90)+math.radians(i)),d*math.sin(-math.radians(90)+math.radians(i)))) for i,d in enumerate(scan_data) if d<1199]
        points=np.array(points); detected_lines=[]
        if len(points)<min_points:return[]
        for _ in range(num_lines):
            if len(points)<min_points:break
            best_inliers=np.array([])
            for _ in range(20):
                s=points[np.random.choice(len(points),2,replace=False)];p1,p2=s[0],s[1]
                A=p2[1]-p1[1];B=p1[0]-p2[0];C=-A*p1[0]-B*p1[1];norm=math.sqrt(A**2+B**2)
                if norm==0:continue
                dists=abs(A*points[:,0]+B*points[:,1]+C)/norm;inliers=points[dists<threshold]
                if len(inliers)>len(best_inliers):best_inliers=inliers
            if len(best_inliers)>min_points:
                x,y=best_inliers[:,0],best_inliers[:,1];m,c=np.polyfit(x,y,1) if np.std(x)>1e-3 else (float('inf'),x[0])
                angle=math.pi/2 if m==float('inf') else math.atan(-1/m);dist=abs(c) if m==float('inf') else abs(c)/math.sqrt(m**2+1)
                detected_lines.append({'angle':angle,'dist':dist,'inliers':best_inliers})
                dists=abs(points[:,0]-c) if m==float('inf') else abs(m*points[:,0]-points[:,1]+c)/math.sqrt(m**2+1)
                points=points[dists>=threshold]
            else:break
        return detected_lines

# --- ALL VISUALIZATION FUNCTIONS ---
def create_course_mask():
    mask = pygame.Surface((WIDTH,HEIGHT),pygame.SRCALPHA); mask.fill((0,0,0,180))
    pygame.draw.rect(mask,(0,0,0,0),(50,50,700,700)); pygame.draw.rect(mask,(0,0,0,180),(250,250,300,300))
    return mask
def draw_plan(screen, planner):
    if planner.state != PlannerState.IDLE:
        pre_target = planner.waypoints[0]; pygame.draw.circle(screen,CYAN,tuple(map(int,pre_target)),8); pygame.draw.circle(screen,WHITE,tuple(map(int,pre_target)),8,1)
        tx,ty,t_theta = planner.final_target_pose; pygame.draw.circle(screen,MAGENTA,(int(tx),int(ty)),8); pygame.draw.circle(screen,WHITE,(int(tx),int(ty)),8,1)
        end_x=tx+40*math.cos(t_theta);end_y=ty+40*math.sin(t_theta);pygame.draw.line(screen,GREEN,(tx,ty),(end_x,end_y),4)
def draw_robot(screen, robot):
    pygame.draw.circle(screen,RED,(int(robot.x),int(robot.y)),ROBOT_SIZE)
    end_x=robot.x+ROBOT_SIZE*2.0*math.cos(robot.theta);end_y=robot.y+ROBOT_SIZE*2.0*math.sin(robot.theta)
    pygame.draw.line(screen,WHITE,(int(robot.x),int(robot.y)),(int(end_x),int(end_y)),3)
def draw_estimated_pose(screen, pose):
    x,y,theta=pose; surface=pygame.Surface((ROBOT_SIZE*4,ROBOT_SIZE*4),pygame.SRCALPHA); center=(ROBOT_SIZE*2,ROBOT_SIZE*2)
    pygame.draw.circle(surface,(*BLUE,128),center,ROBOT_SIZE)
    end_x=center[0]+ROBOT_SIZE*2.0*math.cos(theta);end_y=center[1]+ROBOT_SIZE*2.0*math.sin(theta)
    pygame.draw.line(surface,(*WHITE,128),center,(end_x,end_y),3); screen.blit(surface,(int(x-center[0]),int(y-center[1])))
def draw_lidar_and_inliers(screen, robot, scan_data, inlier_sets):
    for i,dist in enumerate(scan_data):
        if dist < LIDAR_RANGE * 0.99:
            angle=robot.theta-math.radians(90)+math.radians(i); pos=(int(robot.x+dist*math.cos(angle)),int(robot.y+dist*math.sin(angle)))
            pygame.draw.circle(screen,(100,100,0),pos,2)
    colors=[CYAN,MAGENTA,GREEN,ORANGE]
    for i,inlier_group in enumerate(inlier_sets):
        color=colors[i%len(colors)]
        for p_x,p_y in inlier_group:
            world_x=robot.x+p_x*math.cos(robot.theta)-p_y*math.sin(robot.theta)
            world_y=robot.y+p_x*math.sin(robot.theta)+p_y*math.cos(robot.theta)
            pygame.draw.circle(screen,color,(int(world_x),int(world_y)),3)

def main():
    pygame.init(); screen=pygame.display.set_mode((WIDTH,HEIGHT)); font=pygame.font.SysFont("monospace",16); clock=pygame.time.Clock()
    robot=Robot(300,150,0); localization=LocalizationSystem((robot.x+10,robot.y+10,robot.theta-0.1)); planner=Planner(); steering_pid=PIDController(1.1,0.01,0.4)
    course_mask = create_course_mask()
    setting_target_pos=None; running=True
    while running:
        dt = clock.tick(FPS) / 1000.0
        for event in pygame.event.get():
            if event.type==pygame.QUIT:running=False
            if event.type==pygame.MOUSEBUTTONDOWN:setting_target_pos=event.pos
            if event.type==pygame.MOUSEBUTTONUP and setting_target_pos:
                fx,fy=event.pos;dx,dy=fx-setting_target_pos[0],fy-setting_target_pos[1]
                planner.set_target((setting_target_pos[0],setting_target_pos[1],math.atan2(dy,dx)));setting_target_pos=None
        
        scan_data=robot.get_lidar_scan(WALLS); noisy_theta=robot.get_noisy_heading()
        localization.predict(robot.speed,robot.steering_angle,dt)
        localization.correct(scan_data,noisy_theta)
        estimated_pose=localization.estimated_pose
        target_speed,steer_err=planner.update(estimated_pose)
        robot.steering_angle=max(-math.radians(40),min(math.radians(40),steering_pid.update(steer_err,dt)))
        robot.speed=target_speed; robot.update_pose(dt)
        
        screen.fill(BLACK); [pygame.draw.line(screen,WHITE,w[0],w[1],3) for w in WALLS]
        draw_lidar_and_inliers(screen,robot,scan_data,localization.last_inliers)
        draw_plan(screen,planner)
        draw_estimated_pose(screen,estimated_pose)
        draw_robot(screen,robot)
        screen.blit(course_mask,(0,0))
        if setting_target_pos: pygame.draw.line(screen,GREEN,setting_target_pos,pygame.mouse.get_pos(),2)
        
        info_y=10
        def draw_text(t): nonlocal info_y; screen.blit(font.render(t,True,WHITE),(10,info_y)); info_y+=20
        draw_text(f"Real:({robot.x:.1f},{robot.y:.1f})@{math.degrees(robot.theta):.1f}"); draw_text(f"Est: ({estimated_pose[0]:.1f},{estimated_pose[1]:.1f})@{math.degrees(estimated_pose[2]):.1f}")
        draw_text(f"Fix Type: {localization.last_fix_type}")
        pygame.display.flip()
    pygame.quit()

if __name__ == "__main__": main()