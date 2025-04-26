import pybullet as p
import time
import pybullet_data
import random  # To generate random values
import math  # To use math.sin()

import copy

import torch
import itertools
import random
import gym
from gym import spaces
import numpy as np
import torch.nn.functional as F

from collections import deque, namedtuple
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import IterableDataset
from torch.optim import AdamW
from torch.distributions.normal import Normal

from pytorch_lightning import LightningModule, Trainer

from gym.wrappers import RecordVideo, RecordEpisodeStatistics


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class RobotGraspingEnv(gym.Env):
    def __init__(self):
        super(RobotGraspingEnv, self).__init__()

        # Initialize the PyBullet simulation
        # self.physicsClient = p.connect(p.DIRECT)  # Use p.GUI for visualization
        print("just connected")
        self.physicsClient = p.connect(p.GUI)
        
        p.setAdditionalSearchPath(pybullet_data.getDataPath())  # Standard path
        p.setGravity(0, 0, -10)
        
        # Load plane and robot
        self.planeId = p.loadURDF("plane.urdf")
        
        # Starting position for the Poppy robot
        self.startPos = [0, 0, 0]  # Adjusted to be above the ground
        self.startOrientation = p.getQuaternionFromEuler([0, 0, 0])  # Robot's orientation (no rotation)
        self.robotId = p.loadURDF("poppy_torso.urdf", self.startPos, self.startOrientation, useFixedBase=True)
        
        # Load the cube (target object to hold)
        

        # Define action and observation space
        low = np.array([-1.57] * 8)  # Default range for all elements
        high = np.array([1.57] * 8)  # Default range for all elements

        # Set specific bounds for the 4th element (index 3) and 8th element (index 7)
        low[3] = -1.57  # Fourth element ranges from 0 to -1.57
        high[3] = 0
       

        low[7] = 0  # Eighth element ranges from 0 to 1.57
        high[7] = 1.57

        # Define action space using the low and high bounds
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.observation_space = spaces.Dict({
            'observation': spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32),  # Joint states + cube position
            'achieved_goal': spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32),  # Cube position (3D)
            'desired_goal': spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)   # Desired goal for the cube position (3D)
        })
        self.desired_goal=np.array([self.startPos[0]+0.2, self.startPos[1] -0.25, self.startPos[2]+0.2,self.startPos[0], self.startPos[1] -0.25, self.startPos[2]+0.2])
        self.cubeId1 = self.create_cube(self.desired_goal[:3])
        self.cubeId2 = self.create_cube(self.desired_goal[3:])
        self.reach_threshold=0.1# for compute reward

    def joint_name():
        # Print all joint information to identify the hand-related joints
        for i in range(num_joints):
            joint_info = p.getJointInfo(poppyId, i)
            joint_name = joint_info[1].decode("utf-8")
            # print(f"Joint {i}: {joint_name}")
            # The link name is stored at index 12 in the joint_info tuple
            link_name = joint_info[12].decode('utf-8')  # Decoding from bytes to string
            
            print(f"Joint {i}: {joint_name}, Link: {link_name}")


    def create_cube(self,startPos):
            cube_pos = [startPos[0], startPos[1], startPos[2]]  # Move 1.5 units in front of the robot
            cube_orientation = p.getQuaternionFromEuler([0, 0, 0])
            cube_scale =  [0.01, 0.01, 0.01]  # Scale down to 20% of the original size        
            cube_collision_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=cube_scale)
            cube_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=cube_collision_id, basePosition=cube_pos, baseOrientation=cube_orientation)
            
            return cube_id

    def is_forearm_touching_cube(self,poppyId, forearm_link_id, cubeId):
        # Get contact points
        contact_points = p.getContactPoints(bodyA=poppyId, bodyB=cubeId)
        
        
        # Loop through the contact points to check if the forearm link is involved
        for contact in contact_points:
            
            
            if contact[4] == forearm_link_id:  # contact[4] is the link index
                print("hi")
                return True  # The forearm is touching the cube
        return False  # No contact with the forearm

    # Function to generate random target positions within a specific range
    def get_random_position(min_val, max_val):
        return random.uniform(min_val, max_val)
    # Function to control joints with a sine wave motion
    def sine_wave_position(self,amplitude, frequency, time_step):
        return amplitude * math.sin(frequency * time_step)
    
    def reset(self,succ=False):
        print("reset")
        p.resetBasePositionAndOrientation(self.robotId, self.startPos,self.startOrientation)
        if succ:
            self.desired_goal = np.array([random.uniform(self.startPos[0] , self.startPos[0] + 0.2),
                                    random.uniform(self.startPos[1] - 0.2, self.startPos[1] -0.15),
                                    random.uniform(self.startPos[2] , self.startPos[2] + 0.2),
                                    random.uniform(self.startPos[0] - 0.2, self.startPos[0] ),
                                    random.uniform(self.startPos[1] - 0.2, self.startPos[1] - 0.15),
                                    random.uniform(self.startPos[2] , self.startPos[2] + 0.2)])
        p.resetBasePositionAndOrientation(self.cubeId1,self.desired_goal[:3], self.startOrientation)
        p.resetBasePositionAndOrientation(self.cubeId2,self.desired_goal[3:], self.startOrientation)
        
        return self.get_observation()
    
    def get_observation(self):
        # joint_positions = [p.getJointState(self.robotId, i)[0] for i in range(5, 13) if i not in [5, 9]]
        joint_positions = [p.getJointState(self.robotId, i)[0] for i in range(5, 13) ]
        
        #cube_position, _ = p.getBasePositionAndOrientation(self.cubeId)
        
        # Define achieved_goal (cube position)
        achieved_goal = self.get_arm_positions(self.robotId)  ### make achieved goal to arm position 
        
        
        desired_goal = self.desired_goal  # Example desired goal (could be set as needed)
        
        # Return the combined observation as a dictionary
        # observation = {
        #     'observation': np.array(joint_positions + list(cube_position)),
        #     'achieved_goal': achieved_goal,
        #     'desired_goal': desired_goal
        # }
        # Return the combined observation as a dictionary
        #observation': np.array(joint_positions),
        observation = {
            'observation':achieved_goal,
            'achieved_goal': achieved_goal,
            'desired_goal': desired_goal
        }
        
        return observation
    
    def step(self, action,info):
        # Apply action (move robot joints)
        # for i, joint_action in enumerate(action):
        #     p.setJointMotorControl2(self.robotId, i, p.POSITION_CONTROL, targetPosition=joint_action)

        # left_shoulder_target_pos = 0
        # left_shoulder_x_target_pos = action[0]
        # left_arm_z_target_pos = action[1]
        # left_elbow_target_pos = action[2]
        # right_shoulder_target_pos = 0
        # right_shoulder_x_target_pos = action[3]
        # right_arm_z_target_pos = action[4]
        # right_elbow_target_pos = action[5]

        left_shoulder_target_pos =action[0]
        left_shoulder_x_target_pos = action[1]
        left_arm_z_target_pos = action[2]
        left_elbow_target_pos = action[3]
        right_shoulder_target_pos = action[4]
        right_shoulder_x_target_pos = action[5]
        right_arm_z_target_pos = action[6]
        right_elbow_target_pos = action[7]

        # left_shoulder_target_pos = 0
        # left_shoulder_x_target_pos =0
        # left_arm_z_target_pos = 0
        # left_elbow_target_pos = 0
        # right_shoulder_target_pos = 0
        # right_shoulder_x_target_pos = 0
        # right_arm_z_target_pos = 0
        # right_elbow_target_pos = 0


        # Use the move_arms function to apply the actions to the robot's joints
        self.move_arms(self.robotId, left_shoulder_target_pos, left_shoulder_x_target_pos, left_arm_z_target_pos,
                       left_elbow_target_pos, right_shoulder_target_pos, right_shoulder_x_target_pos,
                       right_arm_z_target_pos, right_elbow_target_pos)

        
        # Step the simulation
        # print("before")
        # print(np.round(action, 1))
        # observation = self.get_observation()
        # print(np.round(observation['observation'], 1))

        tolerance = 0.01  # radians or meters, depending on your joints
        max_steps = 50    # safety limit so you don't get stuck in infinite loop
        reached_target = False
        for _ in range(max_steps):
            p.stepSimulation()

            # self.move_arms(self.robotId, left_shoulder_target_pos, left_shoulder_x_target_pos, left_arm_z_target_pos,
            #            left_elbow_target_pos, right_shoulder_target_pos, right_shoulder_x_target_pos,
            #            right_arm_z_target_pos, right_elbow_target_pos)
            
            # Get joint positions
            
            joint_positions = [p.getJointState(self.robotId, i)[0] for i in range(5, 13) ]
            current_joint_pos=np.array(joint_positions)
            
            # Check if all joints are within tolerance
            if np.all(np.abs(action - current_joint_pos) < tolerance):
                reached_target = True     
                
                break

        # Get observations
        #print(np.round(np.abs((action - current_joint_pos)),1))     
        # if not reached_target:
        #     print("⚠️  Reached max_steps without hitting target joint positions.") 
        observation = self.get_observation()
        #print(np.round(action-observation['observation'], 1))
        info=info+1
        # Calculate reward
        reward,done,succ = self.compute_reward(observation['achieved_goal'], observation['desired_goal'], info)
        # succ tell if it was succesful both arm to reach goal
        
        
        # Done condition: Check if robot successfully holds the box

        ##### make done true if reaches position
        
        
        return observation, reward, done,info,succ
    
    
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


    def compute_reward(self,achieved_goal,desired_goal,info):
        """
        change this to target position values
        """
        left_target_position = desired_goal[:3]  # First 3 elements of desired goal
        right_target_position = desired_goal[3:]  # Last 3 elements of desired goal
        
       
        # Get the positions of the left and right arms
        left_arm_position = achieved_goal[:3]  # First 3 elements of achieved goal
        right_arm_position = achieved_goal[3:]  # Last 3 elements of achieved goal
        
        # if info % 300 == 0:
        #     #print("entered")
        #     self.reach_threshold = max(self.reach_threshold - 0.05, 0.1)
        # Calculate the distance of the left arm and right arm from their respective target points
        distance_left = np.linalg.norm(left_arm_position - left_target_position)
        distance_right = np.linalg.norm(right_arm_position - right_target_position)
        
        # # Define rewards for reaching the targets
        # left_arm_reward = 10 if distance_left < reach_threshold else 0  # Reward for left arm if within threshold
        # right_arm_reward = 10 if distance_right < reach_threshold else 0  # Reward for right arm if within threshold
        
        # # Higher reward if both arms reach their targets
        # both_arms_reached = 20 if distance_left < reach_threshold and distance_right < reach_threshold else 0
        # Define rewards for reaching the targets

        # left_arm_reward = 1.0 / (distance_left + 1e-3)  # Avoid division by zero
        # right_arm_reward = 1.0 / (distance_right + 1e-3)
        left_arm_reward=-5*distance_left
        right_arm_reward=-5*distance_right
        

        done=False
        succ=False

        # if info>50000:
        #     done=True
        
        # left_arm_reward=-10
        # right_arm_reward=-10
        # Higher reward if both arms reach their targets
        both_arms_reached = 0
        # if self.is_forearm_touching_cube(self.robotId,8,self.cubeId):
            
        #     print("left")
        #     left_arm_reward=0.5
        # if self.is_forearm_touching_cube(self.robotId,12,self.cubeId):
            
        #     print("right")
        #     right_arm_reward=0.5
        # if (left_arm_reward+right_arm_reward)==1:
        #     done=True
        #     both_arms_reached=3
        #print(self.reach_threshold)
        # Total reward
        # if distance_left < self.reach_threshold:
            
            
        #     left_arm_reward=10
        #     done=True
        # if distance_right < self.reach_threshold:
            
            
        #     right_arm_reward=10
        #     done=True
        total_reward = left_arm_reward + right_arm_reward + both_arms_reached
        
        # done = True if distance_left < self.reach_threshold and distance_right < self.reach_threshold else False
        

        # if distance_left < self.reach_threshold+0.15 and distance_right < self.reach_threshold+0.15 :
            
        #     total_reward=total_reward+2

        # if distance_left < self.reach_threshold+0.1 and distance_right < self.reach_threshold+0.1 :
             
        #     total_reward=total_reward+4


        if distance_left < self.reach_threshold and distance_right < self.reach_threshold :
            done = True 
            succ=True
            total_reward=total_reward+10

        if achieved_goal[1]>0:
            total_reward=total_reward-10
            done=True
        if achieved_goal[4]>0:
            total_reward=total_reward-10
            done=True


        if info > 50:
            done=True
            print("too many steps")

        
        
        return total_reward , done ,succ

    
   


    def move_arms(self,poppyId, left_shoulder_target_pos, left_shoulder_x_target_pos, left_arm_z_target_pos, 
                left_elbow_target_pos, right_shoulder_target_pos, right_shoulder_x_target_pos, 
                right_arm_z_target_pos, right_elbow_target_pos):
        
        # Left arm joints control (each joint gets a specific target position)
        p.setJointMotorControl2(poppyId, 0, p.POSITION_CONTROL, targetPosition=0)  # Left shoulder Y
        p.setJointMotorControl2(poppyId, 1, p.POSITION_CONTROL, targetPosition=0)  # Left shoulder X
        p.setJointMotorControl2(poppyId, 2, p.POSITION_CONTROL, targetPosition=0)  # Left arm Z
        p.setJointMotorControl2(poppyId, 3, p.POSITION_CONTROL, targetPosition=0)    # Left elbow Y
        p.setJointMotorControl2(poppyId, 4, p.POSITION_CONTROL, targetPosition=0)    # Left elbow Y
    
    
        p.setJointMotorControl2(poppyId, 5, p.POSITION_CONTROL, targetPosition=left_shoulder_target_pos)  # Left shoulder Y
        p.setJointMotorControl2(poppyId, 6, p.POSITION_CONTROL, targetPosition=left_shoulder_x_target_pos)  # Left shoulder X
        p.setJointMotorControl2(poppyId, 7, p.POSITION_CONTROL, targetPosition=left_arm_z_target_pos)  # Left arm Z
        p.setJointMotorControl2(poppyId, 8, p.POSITION_CONTROL, targetPosition=left_elbow_target_pos)    # Left elbow Y

        # Right arm joints control (each joint gets a specific target position)
        p.setJointMotorControl2(poppyId, 9, p.POSITION_CONTROL, targetPosition=right_shoulder_target_pos) # Right shoulder Y
        p.setJointMotorControl2(poppyId, 10, p.POSITION_CONTROL, targetPosition=right_shoulder_x_target_pos) # Right shoulder X
        p.setJointMotorControl2(poppyId, 11, p.POSITION_CONTROL, targetPosition=right_arm_z_target_pos)  # Right arm Z
        p.setJointMotorControl2(poppyId, 12, p.POSITION_CONTROL, targetPosition=right_elbow_target_pos)  # Right elbow Y

class ReplayBuffer:
    def __init__(self, capacity, her_probability=0.8):
        self.her_probability = her_probability
        self.buffer = deque(maxlen=capacity//2)
        self.her_buffer = deque(maxlen=capacity//2)

    def __len__(self):
        return len(self.buffer) + len(self.her_buffer)

    def append(self, experience, her=False):
        if her:
            self.her_buffer.append(experience)
        else:
            self.buffer.append(experience)

    def sample(self, batch_size):
        her_batch_size = int(batch_size * self.her_probability)
        regular_batch_size = batch_size - her_batch_size

        batch = random.sample(self.buffer, regular_batch_size)
        her_batch = random.sample(self.her_buffer, her_batch_size)
        full_batch = list(batch + her_batch)
        random.shuffle(full_batch)
        return full_batch
  
class RLDataset(IterableDataset):
    def __init__(self, buffer, sample_size=400):
        self.buffer = buffer
        self.sample_size = sample_size

    def __iter__(self):
        for experience in self.buffer.sample(self.sample_size):
            yield experience

class DQN(nn.Module):

    def __init__(self, hidden_size, obs_size, out_dims):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size + out_dims, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, state, action):
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).to(device)
        if isinstance(action, np.ndarray):
            action = torch.from_numpy(action).to(device)
        in_vector = torch.hstack((state, action))
        return self.net(in_vector.float())

class GradientPolicy(nn.Module):

    def __init__(self, hidden_size, obs_size, out_dims, max):
        super().__init__()

        self.max = torch.from_numpy(max).to(device)

        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        self.linear_mu = nn.Linear(hidden_size, out_dims)
        self.linear_log_std = nn.Linear(hidden_size, out_dims)

    def forward(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).to(device)
        x = self.net(obs.float())
        mu = self.linear_mu(x)

        log_std = self.linear_log_std(x)
        log_std = log_std.clamp(-20, 2)
        std = log_std.exp()

        dist = Normal(mu, std)
        action = dist.rsample()
        log_prob = dist.log_prob(action)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        log_prob -= (2* (np.log(2) - action - F.softplus(-2*action))).sum(dim=-1, keepdim=True)

        action = torch.tanh(action) * self.max
        return action, log_prob
    
class SACHER(LightningModule):

    def __init__(self, capacity=100_000, batch_size=256, lr=1e-3,
                hidden_size=256, gamma=0.99, loss_fn=F.smooth_l1_loss, optim=AdamW,
                samples_per_epoch=1_000, tau=0.05, alpha=0.02, her=0.8):

        super().__init__()

        self.env = RobotGraspingEnv()
        

        ag_size = self.env.observation_space['achieved_goal'].shape[0]
        dg_size = self.env.observation_space['desired_goal'].shape[0]
        obs_size = self.env.observation_space['observation'].shape[0]
       

        action_dims = self.env.action_space.shape[0]
        
        

        max_action = self.env.action_space.high

        self.q_net1 = DQN(hidden_size, obs_size + dg_size, action_dims)
        self.q_net2 = DQN(hidden_size, obs_size + dg_size, action_dims)
        self.policy = GradientPolicy(hidden_size, obs_size + dg_size, action_dims, max_action)

        self.target_policy = copy.deepcopy(self.policy)
        self.target_q_net1 = copy.deepcopy(self.q_net1)
        self.target_q_net2 = copy.deepcopy(self.q_net2)
        self.succ=False

        self.buffer = ReplayBuffer(capacity=capacity, her_probability=her)
        self.info=0 # no of times the loop has worked, decresase the reach threshold based on this value in compute reward
        self.save_hyperparameters()

        while len(self.buffer) < self.hparams.samples_per_epoch * 2:
            print(f"{len(self.buffer)} samples in experience buffer. Filling...")
            self.play_episodes()

    @torch.no_grad()
    def play_episodes(self, policy=None):
        state = self.env.reset(self.succ)
        done = False
        self.info=0
        while not done:
            
            desired_state = np.hstack([state['observation'], state['desired_goal']])
            
            achieved_state = np.hstack([state['observation'], state['achieved_goal']])

            if policy and random.random() > 0.1:
                action, _ = self.policy(desired_state)
                action = action.cpu().numpy()
            else:
                action = self.env.action_space.sample()
                
                         
            next_state, reward, done, self.info,self.succ = self.env.step(action,self.info)

            
            next_desired_state = np.hstack([next_state['observation'], next_state['desired_goal']])
            next_achieved_state = np.hstack([next_state['observation'], next_state['achieved_goal']])
            

            # Desired goal.
            exp = (desired_state, action, reward, done, next_desired_state)
            self.buffer.append(exp)

            # Achieved goal.
            reward , _,_ = self.env.compute_reward(next_state['achieved_goal'], next_state['achieved_goal'], self.info)
            exp = (achieved_state, action, reward, done, next_achieved_state)
            self.buffer.append(exp, her=True)

            state = next_state

    def forward(self, x):
        output = self.policy(x)
        return output

    def configure_optimizers(self):
        q_net_parameters = itertools.chain(self.q_net1.parameters(), self.q_net2.parameters())
        q_net_optimizer = self.hparams.optim(q_net_parameters, lr=self.hparams.lr)
        policy_optimizer = self.hparams.optim(self.policy.parameters(), lr=self.hparams.lr)
        return [q_net_optimizer, policy_optimizer]

    def train_dataloader(self):
        dataset = RLDataset(self.buffer, self.hparams.samples_per_epoch)
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.hparams.batch_size,
            num_workers=0
        )
        return dataloader

    def training_step(self, batch, batch_idx, optimizer_idx):
        states, actions, rewards, dones, next_states = batch
        rewards = rewards.unsqueeze(1)
        dones = dones.unsqueeze(1)

        if optimizer_idx == 0:

            action_values1 = self.q_net1(states, actions)
            action_values2 = self.q_net2(states, actions)

            target_actions, target_log_probs = self.target_policy(next_states)

            next_action_values = torch.min(
                self.target_q_net1(next_states, target_actions),
                self.target_q_net2(next_states, target_actions)
            )
            next_action_values[dones] = 0.0

            expected_action_values = rewards + self.hparams.gamma * (next_action_values - self.hparams.alpha * target_log_probs)

            q_loss1 = self.hparams.loss_fn(action_values1, expected_action_values)
            q_loss2 = self.hparams.loss_fn(action_values2, expected_action_values)

            q_loss_total = q_loss1 + q_loss2
            
            self.log("episode/Q-Loss", q_loss_total)
            return q_loss_total

        elif optimizer_idx == 1:

            actions, log_probs = self.policy(states)

            action_values = torch.min(
                self.q_net1(states, actions),
                self.q_net2(states, actions)
            )

            policy_loss = (self.hparams.alpha * log_probs - action_values).mean()
            self.log("episode/Policy Loss", policy_loss)
            return policy_loss

    def training_epoch_end(self, training_step_outputs):
        self.play_episodes(policy=self.policy)

        polyak_average(self.q_net1, self.target_q_net1, tau=self.hparams.tau)
        polyak_average(self.q_net2, self.target_q_net2, tau=self.hparams.tau)
        polyak_average(self.policy, self.target_policy, tau=self.hparams.tau)

        #self.log("episode/episode_return", self.env.return_queue[-1])
        
def polyak_average(net, target_net, tau=0.01):
    for qp, tp in zip(net.parameters(), target_net.parameters()):
        tp.data.copy_(tau * qp.data + (1 - tau) * tp.data)


current_time=0




def test_trained_agent(policy, env, episodes=10):
    
    
    for ep in range(episodes):
        state = env.reset(True)
        done = False
        total_reward = 0
        info=0
        print(f"\n🚀 Starting test episode {ep + 1}")

        while not done:
            # Combine observation and desired goal for the input
            input_state = np.hstack([state['observation'], state['desired_goal']])
            #input_state=torch.from_numpy(input_state).to(device)
            
            # Get action from policy (no exploration noise here)
            with torch.no_grad():
                action, _ = policy(input_state)
                

            action = action.cpu().numpy()

            # Step in the environment
            next_state, reward, done, info,succ = env.step(action, info)
            total_reward += reward
            

            print(info)

            state = next_state
            time.sleep(0.05)  # Slow down for visualization if GUI is used

        print(f"✅ Episode {ep + 1} finished with total reward: {total_reward:.2f}")

        try:
            input("⏸️  Last frame displayed. Press Enter to continue...")
        except KeyboardInterrupt:
            print("⏹️  Interrupted by user.")


def main():

    if torch.cuda.is_available():
        # Get the number of GPUs available
        num_gpus = torch.cuda.device_count()
        print(f'Number of GPUs available: {num_gpus}')
    else:
        print('CUDA is not available. Using CPU.')
    algo = SACHER( lr=1e-3, alpha=0.2, tau=0.1)
    
    if num_gpus > 0:
        print("gpu using")
        trainer = Trainer(
            devices=num_gpus,  # Use 'devices' for number of GPUs
            accelerator="gpu",  # Specify GPU as the accelerator
            max_epochs=1_000,
            log_every_n_steps=1
        )
    else:
        trainer = Trainer(
            devices=1,  # You can set devices=1 to use CPU in PyTorch Lightning.
            accelerator="cpu",  # Specify CPU as the accelerator
            max_epochs=4_000,
            log_every_n_steps=1
        )

    trainer.fit(algo)
    
    algo.policy = algo.policy.to(device)

    test_trained_agent(algo.policy, algo.env)
    # Disconnect from the simulation
    p.disconnect()

# # Create the environment
# env = RobotGraspingEnv()

# # Create the RL agent (PPO)
# model = PPO("MlpPolicy", env, verbose=1)

# # Train the agent
# model.learn(total_timesteps=100000)

# # Save the model
# model.save("robot_grasping_model")

# # Evaluate the agent
# obs = env.reset()
# done = False
# while not done:
#     action, _states = model.predict(obs)
#     obs, rewards, done, info = env.step(action)
#     env.render()  # Optionally render the environment


# Protect the entry point for multiprocessing
if __name__ == '__main__':
    main()










