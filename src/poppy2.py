from pypot.dynamixel.io import DxlIO
import pybullet as p
import pybullet_data
import os
import time
import math
# Initialize PyBullet

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.loadURDF("plane.urdf")

startPos = [0, 0, 0]
startOrientation = p.getQuaternionFromEuler([0, 0, 0])

# Load the Poppy robot URDF
p.setAdditionalSearchPath(pybullet_data.getDataPath())
robot_urdf = os.path.join(os.path.dirname(__file__), 'poppy_torso.urdf')
robot = p.loadURDF(robot_urdf, basePosition=startPos, baseOrientation=startOrientation, useFixedBase=True)
      
    


right_arm = [51 , 52, 53, 54]
left_arm = [41, 42, 43, 44]
bust_y = 35
bust = 34
abz = 33

right_arm_p = [10, 11, 12, 13]
left_arm_p = [5, 6, 7, 8]
bust_p = 1
abz_p = 0


with DxlIO('/dev/ttyACM0', baudrate=1000000) as dxl_io:
    # Read the current position of the motor with ID 1
    motor_id = dxl_io.scan()
    print(f"Motor IDs found: {motor_id}")

    # Set all joints to initial position (0 degrees/radians) using dict and zip
    all_joints = right_arm + left_arm + [ bust, abz]
    all_joints_n_bust = right_arm + left_arm + [bust_y, bust, abz]
    initial_positions = dict(zip(all_joints_n_bust, [0] * len(all_joints)))

    # Set hardware motors to initial positions
    dxl_io.set_goal_position(initial_positions)

    print("initialised")

    time.sleep(2)


    def ActionSpace(action_value):
        initial_positions = dict(zip(all_joints, [action_value*180/math.pi]))
        dxl_io.set_goal_position(initial_positions)

        p.setJointMotorControlArray(robot, all_joints, p.POSITION_CONTROL, targetPosition=action_value)
        p.stepSimulation()

        print(f"Setting goal position to {action_value} for joints,{all_joints}")
        print(dxl_io.get_present_position(motor_id))
        end_effector_state_l = p.getLinkState(robot, left_arm_p[3])
        end_effector_state_r = p.getLinkState(robot, right_arm_p[3])

        end_effector_pos_l = end_effector_state_l  # Position (x, y, z)
        end_effector_pos_r = end_effector_state_r
        end_effector_pos = end_effector_pos_l + end_effector_pos_r  # This concatenates the two tuples/lists
        
        time.sleep(1/30)
        return end_effector_pos





  