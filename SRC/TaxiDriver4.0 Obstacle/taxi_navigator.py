#!/usr/bin/env python3
"""
Simple demonstration of the TaxiDriver class
Shows how to control both motor and steering programmatically
"""

import time
from taxi_driver import TaxiDriver

def main():
    """Demonstration of programmatic motor and steering control"""
    taxi = None
   
    map = [[None for _ in range(2)] for _ in range(4)]
    turnExitDist90 = 340
    turnExitDist60 = 180
    parkingAdjust = 200
    turnDist = 950

    current_lane = "center"
    print("=== Taxi Driver Demo with Steering ===")
    
    # Initialize taxi driver
    taxi = TaxiDriver()
    taxi.set_servo_angle_pin_26(0)
    taxi.start_all_controllers()
    
    # Give controllers time to initialize
    print("Initializing controllers...")
    time.sleep(3)
    
    # Wait for camera to be ready
    print("Waiting for camera to initialize...")
    camera_ready = False
    wait_count = 0
    max_wait_time = 30  # Maximum wait time in seconds
    
    while not camera_ready and wait_count < max_wait_time:
        try:
            # Try to get camera detections - if successful, camera is ready
            test_detections = taxi.get_camera_detections()
            if test_detections is not None:  # Camera is responding
                camera_ready = True
                print(test_detections)
                print("Camera is ready!")
            else:
                print(f"Camera not ready yet... waiting ({wait_count + 1}s)")
                time.sleep(1)
                wait_count += 1
        except Exception as e:
            print(f"Camera not ready yet... waiting ({wait_count + 1}s) - Error: {e}")
            time.sleep(1)
            wait_count += 1
    
    if not camera_ready:
        print("Warning: Camera failed to initialize within the timeout period!")
    
    blocks = taxi.get_camera_detections()
    # while len(blocks) == 0:
    #     print("No blocks detected, waiting...")
    #     time.sleep(0.5)
    #     blocks = taxi.get_camera_detections()

    #Note intial position
    initial_left_distance = taxi.get_distance(-90)
    initial_right_distance = taxi.get_distance(90)
    initial_front_distance = taxi.get_distance(0)
    print(f"Initial left distance (-90°): {initial_left_distance} mm")
    print(f"Initial right distance (90°): {initial_right_distance} mm")
    print(f"Initial front distance (0°): {initial_front_distance} mm")

    turning_angle = 89

    if initial_left_distance == 9999 or initial_left_distance < initial_right_distance:
        turning_right = True
        current_lane = "outside"
        turning_angle = -turning_angle
    else:
        turning_right = False
        current_lane = "outside"

    taxi.set_target_angle(90)

    time.sleep(1)
    taxi.set_target_angle(-90)
    time.sleep(1)
    taxi.set_target_angle(0)
    time.sleep(1)


    while(taxi.get_button_state_pin_23() == False):
        
    
        time.sleep(0.1)
    time.sleep(1)

    taxi.set_servo_angle_pin_26(90)
    taxi.move_reverse(100)
    for i in range (7):
        taxi.set_target_angle(taxi.get_current_angle())
        time.sleep(0.1)

    
    
    #start driving forwards
    if turning_right:
        taxi.set_target_angle(-45)
    else:
        taxi.set_target_angle(45)
    taxi.move_forward(99)  # Move forward at 100% speed
    time.sleep(1)
    taxi.stop_motor() #Stop and smell the block
    time.sleep(1)
    if turning_right:
        initial_front_distance = taxi.get_distance(-45)
    else:
        initial_front_distance = taxi.get_distance(45)
    
    parkinZone = 0

    if initial_front_distance < 1500:
        parkinZone = 1
    else:
        parkinZone = 2

    print("Initial front distance at turning angle:", initial_front_distance, "mm, parking zone:", parkinZone)


    time.sleep(1)
    taxi.set_target_angle(0)
    blocks = taxi.get_camera_detections()
    old_lane = current_lane
    taxi.move_forward(99)
    # If a course correction is needed at the start
    if len(blocks) > 0:
        if (old_lane == "outside" and ((not turning_right) and blocks[0]['color'] == 'green' ) or (turning_right and blocks[0]['color'] == 'red' )):
            print("Correcting to inside lane")
            old_target = taxi.get_target_angle()
            taxi.set_target_angle(old_target + turning_angle)
            taxi.wait_for_turn_completion()
            current_lane = "inside"
            while taxi.get_distance(0) > turnExitDist90:
                print("Waiting to reach turn exit distance, current front distance:", taxi.get_distance(0))
                time.sleep(0.1)
            taxi.set_target_angle(old_target)
            taxi.wait_for_turn_completion()
    



    #Constantly check distances at +-100 degrees, until one of them is longer than 1.5m

    left_distance = taxi.get_distance(-90)
    right_distance = taxi.get_distance(90)
    front_distance = taxi.get_distance(0)

    print(f"Left distance (-90°): {left_distance} mm")
    print(f"Right distance (90°): {right_distance} mm")
    print(f"Front distance (0°): {front_distance} mm")

   
       
    for i in range(11):
        outer_wall_addon = 0
        if (i+1)%4 == 0:
            outer_wall_addon = parkingAdjust
        consecutive_clear_count = 0
        while consecutive_clear_count < 3:
            if front_distance < turnDist:
                consecutive_clear_count += 1
                print(f"front is Not clear, moving forward (count: {consecutive_clear_count}/3)")
            else:
                consecutive_clear_count = 0
                print("front blocked, resetting count")
            time.sleep(0.1)
            front_distance = taxi.get_distance(taxi.get_current_angle()- taxi.get_target_angle())
        #turning lane change

        old_lane = current_lane

        old_target = taxi.get_target_angle()

        blocks = taxi.get_camera_detections()
        print("-------------------------Turning-------------------------")
        for block in blocks:
            print(f"Detected block: {block}")
        print(f"Total detected blocks: {len(blocks)}")

        if len(blocks) > 0:
            #Inside lane to outside lane change
            if (current_lane == "inside" and ((not turning_right) and blocks[0]['color'] == 'red' ) or (turning_right and blocks[0]['color'] == 'green' )):
                print("Wait here")
                while taxi.get_distance(0) > turnExitDist90 + outer_wall_addon:
                    print("Waiting to reach turn exit distance, current front distance:", taxi.get_distance(0))
                    current_lane = "outside"
                    time.sleep(0.1)
            
        
            else:
                current_lane = "inside"
        else:
            old_lane = "outside"
            current_lane = "inside"



        taxi.set_target_angle(old_target + turning_angle)

        taxi.wait_for_turn_completion(30)

        print("old lane:", old_lane)
        if old_lane == "outside":
            blocks = taxi.get_camera_detections()
            for block in blocks:
                print(f"Detected block: {block}")
            
            #outside to outside lane change
            if (blocks[0]['color'] == 'red' and not turning_right) or (blocks[0]['color'] == 'green' and turning_right):
                taxi.set_target_angle(old_target)
                taxi.wait_for_turn_completion()
                current_lane = "outside"
                while taxi.get_distance(0) > turnExitDist90 + outer_wall_addon:
                    print("Waiting to reach turn exit distance, current front distance:", taxi.get_distance(0))
                    current_lane = "outside"
                    time.sleep(0.1)
        taxi.set_target_angle(old_target + turning_angle)
        taxi.wait_for_turn_completion()
        
        # time.sleep(1)
        
        print("---------------Driving Forward---------------")
        front_distance = taxi.get_distance(taxi.get_current_angle()- taxi.get_target_angle())
        print("Current front distance:", front_distance, "at angle:",taxi.get_current_angle()- taxi.get_target_angle() )
        consecutive_clear_count = 0
        while consecutive_clear_count < 5:
            if front_distance < 1700:
                consecutive_clear_count += 1
                print(f"front is Not clear, moving forward (count: {consecutive_clear_count}/3) Front distance: {front_distance} mm")
            else:
                consecutive_clear_count = 0
                print("front blocked, resetting count")
            
            time.sleep(0.1)
            front_distance = taxi.get_distance(taxi.get_current_angle()- taxi.get_target_angle())
        
        print("---------------------Mid lane correction---------------------")
        blocks = taxi.get_camera_detections()
        for block in blocks:
            print(f"Detected block: {block}")
        old_lane = current_lane
        if len(blocks) > 0:
            if (old_lane == "outside" and ((not turning_right) and blocks[0]['color'] == 'green' ) or (turning_right and blocks[0]['color'] == 'red' )):
                proceed  = False
                if turning_right:
                    if taxi.get_closest_point(90) > 600:
                        proceed = True
                else:
                    if taxi.get_closest_point(-90) > 600:
                        proceed = True
                if proceed:
                    print("Correcting to inside lane Closest pt - ", taxi.get_closest_point(-90))
                    old_target = taxi.get_target_angle()
                    taxi.set_target_angle(old_target + turning_angle)
                    taxi.wait_for_turn_completion()
                    current_lane = "inside"
                    while taxi.get_distance(0) > turnExitDist90:
                        print("Waiting to reach turn exit distance, current front distance:", taxi.get_distance(0))
                        time.sleep(0.1)
                    taxi.set_target_angle(old_target)
                    taxi.wait_for_turn_completion()
                else:
                    print(f"MID LANE CORRECTION REJECTED {taxi.get_closest_point(90)}------------------------")
            elif(old_lane == "inside" and ((not turning_right) and blocks[0]['color'] == 'red' ) or (turning_right and blocks[0]['color'] == 'green' )):
                print("Correcting to outside lane")
                old_target = taxi.get_target_angle()
                taxi.set_target_angle(old_target - turning_angle)
                taxi.wait_for_turn_completion()
                current_lane = "outside"
                while taxi.get_distance(0) > turnExitDist90 + outer_wall_addon:
                    print("Waiting to reach turn exit distance, current front distance:", taxi.get_distance(0))
                    time.sleep(0.1)
                taxi.set_target_angle(old_target)
                taxi.wait_for_turn_completion()
        
        distance_to_wall = 0
        
    taxi.stop_motor()
        
        
    

    

if __name__ == "__main__":
    main()
