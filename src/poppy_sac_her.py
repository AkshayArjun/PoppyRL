from stable_baselines3 import HerReplayBuffer, SAC
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy
from poppy_env import PoppyEnv
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import VecVideoRecorder


# Define the model class (SAC in this case)
model_class = SAC

# Create the environment
env = PoppyEnv(render=True, max_episode_length=500, alpha= 0.5 )  # Wrap the environment in a DummyVecEnv for vectorized training


# HER parameters
goal_selection_strategy = 'future'  # Equivalent to GoalSelectionStrategy.FUTURE
online_sampling = True  # HER transitions will be sampled online


# Initialize the model
model = model_class(
    "MultiInputPolicy",  # Use a policy that supports multi-input observations
    env,
    replay_buffer_class=HerReplayBuffer,  # Use HER replay buffer
    replay_buffer_kwargs=dict(
        n_sampled_goal=4,  # Number of HER samples per transition
        goal_selection_strategy=goal_selection_strategy,  # Strategy for HER
    ),
    verbose=1,  # Print training progress
    learning_starts= 2000
)


# Train the model
model.learn(total_timesteps=300000, log_interval= 4)  # Adjust the number of timesteps as needed

# Save the trained model
model.save("./poppy_sac_her_model")

# Load the model for evaluation
model = model_class.load('./poppy_sac_her_model', env=env)

# Evaluate the trained model
obs, info = env.reset()
for _ in range(100):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        obs, info = env.reset()