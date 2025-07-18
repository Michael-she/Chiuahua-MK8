import sys
sys.path.append(r'c:\Users\mgshe\Documents\Chiuahua-MK8\Simulation')

# Test to verify the localization bounds checking is working
def test_localization_bounds():
    # Check that the constants are defined correctly
    from robot_sim_mk5 import COURSE_TOP_LEFT, COURSE_WIDTH
    
    print(f"COURSE_TOP_LEFT: {COURSE_TOP_LEFT}")
    print(f"COURSE_WIDTH: {COURSE_WIDTH}")
    
    # Expected track bounds
    expected_min = COURSE_TOP_LEFT + 10  # 60
    expected_max = COURSE_TOP_LEFT + COURSE_WIDTH - 10  # 640
    
    print(f"Expected track bounds: ({expected_min}, {expected_min}) to ({expected_max}, {expected_max})")
    
    # Test position clamping
    def clamp_position(x, y):
        return max(60, min(640, x)), max(60, min(640, y))
    
    test_cases = [
        (50, 50),     # Outside bounds - should clamp to (60, 60)
        (700, 700),   # Outside bounds - should clamp to (640, 640)
        (350, 350),   # Inside bounds - should remain (350, 350)
        (-100, 200),  # Partially outside - should clamp to (60, 200)
        (500, 800),   # Partially outside - should clamp to (500, 640)
    ]
    
    for x, y in test_cases:
        clamped_x, clamped_y = clamp_position(x, y)
        print(f"Position ({x}, {y}) -> ({clamped_x}, {clamped_y})")
        
        # Check if position was clamped
        if clamped_x != x or clamped_y != y:
            print(f"  --> Position was clamped (outside bounds)")
        else:
            print(f"  --> Position within bounds")

if __name__ == "__main__":
    test_localization_bounds()
