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

    print("=== Taxi Driver Demo with Steering ===")
    
    # Initialize taxi driver
    taxi = TaxiDriver()
    taxi.start_all_controllers()
    
    # Give controllers time to initialize
    print("Initializing controllers...")
    time.sleep(4)

    #Note intial position
    initial_left_distance = taxi.get_distance(-90)
    initial_right_distance = taxi.get_distance(90)
    iniital_front_distance = taxi.get_distance(0)
    print(f"Initial left distance (-90°): {initial_left_distance} mm")
    print(f"Initial right distance (90°): {initial_right_distance} mm")
    print(f"Initial front distance (0°): {iniital_front_distance} mm")

    #start driving forwards
    taxi.move_forward(90)  # Move forward at 90% speed

    #Constantly check distances at +-90 degrees, until one of them is longer than 1.5m

    turningRight = False
    while True:
        left_distance = taxi.get_distance(-90)
        right_distance = taxi.get_distance(90)
        front_distance = taxi.get_distance(0)

        print(f"Left distance (-90°): {left_distance} mm")
        print(f"Right distance (90°): {right_distance} mm")
        print(f"Front distance (0°): {front_distance} mm")

        if left_distance > 1500:
            print("wall fell away on left...")
            turningRight = False
            break
        if right_distance > 1500:
            print("wall fell away on right...")
            turningRight = True
            
            break
    time.sleep(0.5) 
    for i in range(10):
        taxi.set_motor_speed(100) #keep moving forward

        if turningRight:
            taxi.turn_right(89)
            if i%4 == 0:
                taxi.turn_right(1) #correct for drift
        else:
            taxi.turn_left(89)
            if i%4 == 0:
                taxi.turn_left(1) #correct for drift
        time.sleep(3) #Wait for the turn to complete
        front_distance = taxi.get_distance(0)
        while front_distance > 1500:
            print("front is clear, moving forward")
            
            time.sleep(0.1)
            front_distance = taxi.get_distance(0)
        distance_to_wall = 0
        taxi.set_motor_speed(90) #slow down when approaching wall
        while distance_to_wall < 1500:
            time.sleep(0.5)
            if turningRight:
                distance_to_wall = taxi.get_distance(90)
            else:
                distance_to_wall = taxi.get_distance(-90)
            
            print(f"Distance to wall on {'left' if turningRight else 'right'}: {distance_to_wall} mm")
        
    target_distance_from_wall = 0
    time.sleep(0.5)
    if turningRight:
        taxi.turn_right(90)
        target_distance_from_wall = initial_left_distance
    else:
        taxi.turn_left(90)
        target_distance_from_wall = initial_right_distance

    time.sleep(3)
    while taxi.get_distance(0) > target_distance_from_wall+130:
        print(f"Adjusting to target distance from wall: {target_distance_from_wall} mm, current front distance: {taxi.get_distance(0)} mm")
     

    
    if turningRight:
        taxi.turn_right(90)
    else:
        taxi.turn_left(90)

    time.sleep(3)
    while taxi.get_distance(0) > iniital_front_distance+160:
        time.sleep(0.1)
    taxi.stop_motor()

    

if __name__ == "__main__":
    main()
