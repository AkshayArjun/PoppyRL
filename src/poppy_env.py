import os
import random

import pybullet as p
import pybullet_data

import gymnasium as gym
from gymnasium import spaces
import numpy as np

class PoppyEnv(gym.Env):
    def __init__(self, render=True):
        super(PoppyEnv, self).__init__()
        self.render = render
        if render:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)

        self.reach_threshold=0.1
        self.startPos = [0, 0, 0]
        self.startOrientation = p.getQuaternionFromEuler([0, 0, 0])

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.robot_urdf = os.path.join(os.path.dirname(__file__), 'poppy_torso.urdf')
        self.reset()

        self.num_joints = p.getNumJoints(self.robot)
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.num_joints,), dtype=np.float32)
        self.observation_space = spaces.Dict({
            "observation": spaces.Box(low=-1, high=1, shape=(self.num_joints,), dtype=np.float32),
            "achieved_goal": spaces.Box(low=-10.0, high=10.0, shape=(6,), dtype=np.float32),
            "desired_goal": spaces.Box(low=-10.0, high=10.0, shape=(6,), dtype=np.float32),
        })

        self._target_location=np.array([self.startPos[0]+0.2, self.startPos[1] -0.25, self.startPos[2]+0.2,self.startPos[0]+0.25, self.startPos[1] + 0.25, self.startPos[2]+0.2])
        self.cubeId1 = self.create_cube(self._target_location[:3], color=[1, 0, 0, 1])  # Red cube
        self.cubeId2 = self.create_cube(self._target_location[3:], color=[0, 0, 1, 1])  # Green cube

    def create_cube(self,startPos, color):
        cube_pos = [startPos[0], startPos[1], startPos[2]]
        cube_orientation = p.getQuaternionFromEuler([0, 0, 0])
        cube_scale =  [0.01, 0.01, 0.01]
        cube_collision_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=cube_scale)
        cube_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=cube_collision_id, basePosition=cube_pos, baseOrientation=cube_orientation) 
        p.changeVisualShape(cube_id, -1, rgbaColor= color)  # Set color to blue
        return cube_id
    

    def get_reward(self, achieved_goal, desired_goal):
        # Calculate the Euclidean distance between the achieved goal and the desired goal
        distance = np.linalg.norm(achieved_goal - desired_goal)

        # Define the reward: negative distance to encourage minimizing the distance
        reward = -distance

        # Add a success bonus if the distance is within the reach threshold
        if distance < self.reach_threshold:
            reward += 10.0  # Success bonus

        return reward

    def step(self, action):
        p.setJointMotorControlArray(self.robot, range(self.num_joints), p.POSITION_CONTROL, targetPositions=action)
        p.stepSimulation()
        joint_states = [p.getJointState(self.robot, i)[0] for i in range(self.num_joints)]
        ee_state = self.get_arm_positions(self.robot)
        obs = {
            "observation": np.array(joint_states, dtype=np.float32),
            "achieved_goal": np.array(ee_state, dtype=np.float32),
            "desired_goal": self._target_location,
        }
        reward = self.get_reward(ee_state, self._target_location)
        terminated = reward > 0  # Terminate if the success bonus is achieved
        truncated = False  # No truncation logic implemented
        info = {
            "distance_to_target": np.linalg.norm(ee_state - self._target_location),
            "target_location": self._target_location,
            "end_effector_position": ee_state,
        }
        return obs, reward, terminated, truncated, info


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        plane = p.loadURDF("plane.urdf")
        self.robot = p.loadURDF(self.robot_urdf, useFixedBase=True)
        self.num_joints = p.getNumJoints(self.robot)
        self._target_location = np.array([random.uniform(self.startPos[0], self.startPos[0] + 0.2),
                                          random.uniform(self.startPos[1] - 0.2, self.startPos[1] - 0.15),
                                          random.uniform(self.startPos[2], self.startPos[2] + 0.2),
                                          random.uniform(self.startPos[0] - 0.2, self.startPos[0]),
                                          random.uniform(self.startPos[1] - 0.2, self.startPos[1] - 0.15),
                                          random.uniform(self.startPos[2], self.startPos[2] + 0.2)])
        self.cubeId1 = self.create_cube(self._target_location[:3], color=[1, 0, 0, 1])  # Red cube
        self.cubeId2 = self.create_cube(self._target_location[3:], color=[0, 0, 1, 1])
        
        obs = self._get_obs()
        info = {
            "target_location": self._target_location,
        }
        return obs, info
    
    def _get_obs(self):
        joint_states = [p.getJointState(self.robot, i)[0] for i in range(self.num_joints)]
        ee_state = self.get_arm_positions(self.robot)
        return {
            "observation": np.array(joint_states, dtype=np.float32),
            "achieved_goal": np.array(ee_state, dtype=np.float32),
            "desired_goal": self._target_location,
        }

    def get_arm_positions(self,body_id):
        # Link IDs for left and right forearms
        link_id_left_forearm = 8
        link_id_right_forearm = 12
        
        # Get the link state for the left forearm
        link_state_left_forearm = p.getLinkState(body_id, link_id_left_forearm)
        
        position_left_forearm = link_state_left_forearm[0]  # COM position of left forearm
        
        # Get the link state for the right forearm
        link_state_right_forearm = p.getLinkState(body_id, link_id_right_forearm)
        position_right_forearm = link_state_right_forearm[0]  # COM position of right forearm
        
        return np.concatenate([position_right_forearm, position_left_forearm])

    def render(self, mode='human'):
        if mode == "human":
            if not self.render:
                p.disconnect()
                p.connect(p.GUI)
                self.render = True
        elif mode == "rgb_array":
            width, height, rgbImg, _, _ = p.getCameraImage(
            width=640,
            height=480,
            viewMatrix=p.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=self.startPos,
                distance=1.5,
                yaw=50,
                pitch=-35,
                roll=0,
                upAxisIndex=2,
            ),
            projectionMatrix=p.computeProjectionMatrixFOV(
                fov=60, aspect=1.0, nearVal=0.1, farVal=100.0
            ),
            )
            return np.array(rgbImg, dtype=np.uint8)
        else:
            raise NotImplementedError(f"Render mode '{mode}' is not supported.")

    def close(self):
        p.disconnect()
        self.robot = None
        self.cubeId1 = None
        self.cubeId2 = None
        self._target_location = None
        self.startPos = None
        self.startOrientation = None
        self.num_joints = None
        self.action_space = None
        self.observation_space = None
        self.reach_threshold = None
        if self.render:
            self.render = False
        else:
            self.render = False
    