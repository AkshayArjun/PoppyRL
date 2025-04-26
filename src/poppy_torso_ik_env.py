import pybullet as p
import pybullet_data
import gym
from gym import spaces
import numpy as np
import os
import random

class PoppyTorsoIKEnv(gym.Env):
    def __init__(self, render=False):
        super(PoppyTorsoIKEnv, self).__init__()
        self.render = render
        if render:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)

        self.startPos = [0, 0, 0]
        self.startOrientation = p.getQuaternionFromEuler([0, 0, 0])

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.robot_urdf = os.path.join(os.path.dirname(__file__), 'poppy_torso.urdf')
        self.reset_sim()

        self.num_joints = p.getNumJoints(self.robot)
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.num_joints,), dtype=np.float32)
        self.observation_space = spaces.Dict({
            "observation": spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_joints,), dtype=np.float32),
            "achieved_goal": spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32),
            "desired_goal": spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32),
        })
        self.desired_goal=np.array([self.startPos[0]+0.2, self.startPos[1] -0.25, self.startPos[2]+0.2,self.startPos[0], self.startPos[1] -0.25, self.startPos[2]+0.2])
        self.cubeId1 = self.create_cube(self.desired_goal[:3])
        self.cubeId2 = self.create_cube(self.desired_goal[3:])
        self.reach_threshold=0.1
    

    def create_cube(self,startPos):
            cube_pos = [startPos[0], startPos[1], startPos[2]]  # Move 1.5 units in front of the robot
            cube_orientation = p.getQuaternionFromEuler([0, 0, 0])
            cube_scale =  [0.01, 0.01, 0.01]  # Scale down to 20% of the original size        
            cube_collision_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=cube_scale)
            cube_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=cube_collision_id, basePosition=cube_pos, baseOrientation=cube_orientation)
            
            return cube_id


    def reset_sim(self):
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        plane = p.loadURDF("plane.urdf")
        self.robot = p.loadURDF(self.robot_urdf, useFixedBase=True)
        self.num_joints = p.getNumJoints(self.robot)  # <-- Add this line here
        self.target_position = np.array([random.uniform(self.startPos[0] , self.startPos[0] + 0.2),
                                    random.uniform(self.startPos[1] - 0.2, self.startPos[1] -0.15),
                                    random.uniform(self.startPos[2] , self.startPos[2] + 0.2),
                                    random.uniform(self.startPos[0] - 0.2, self.startPos[0] ),
                                    random.uniform(self.startPos[1] - 0.2, self.startPos[1] - 0.15),
                                    random.uniform(self.startPos[2] , self.startPos[2] + 0.2)])
        return self._get_obs()

    def compute_dense_reward(self, achieved_goal, desired_goal):
        return -np.linalg.norm(achieved_goal - desired_goal)

    def compute_sparse_reward(self, achieved_goal, desired_goal):
        return 0.0 if np.linalg.norm(achieved_goal - desired_goal) < 0.05 else -1.0

    def compute_reward(self, achieved_goal, desired_goal, info):
        r_dense = self.compute_dense_reward(achieved_goal, desired_goal)
        r_sparse = self.compute_sparse_reward(achieved_goal, desired_goal)
        alpha = 0.3
        return alpha * r_dense + (1 - alpha) * r_sparse

    def _get_obs(self):
        joint_states = [p.getJointState(self.robot, i)[0] for i in range(self.num_joints)]
        ee_state = self.get_arm_positions(self.robot) # Get left end-effector position
        return {
            "observation": np.array(joint_states, dtype=np.float32),
            "achieved_goal": np.array(ee_state, dtype=np.float32),
            "desired_goal": self.target_position,
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
        
        return np.concatenate([position_left_forearm, position_right_forearm])

    def reset(self):
        return self.reset_sim()

    def step(self, action):
        for i in range(self.num_joints):
            p.setJointMotorControl2(self.robot, i, p.POSITION_CONTROL, targetPosition=action[i])
        for _ in range(10):
            p.stepSimulation()
        obs = self._get_obs()
        reward = self.compute_reward(obs["achieved_goal"], obs["desired_goal"], None)
        done = reward == 0
        return obs, reward, done, {}

    def render(self, mode='human'):
        pass

    def close(self):
        p.disconnect()
