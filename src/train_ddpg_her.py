import torch
from stable_baselines3 import DDPG, HerReplayBuffer
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy
from stable_baselines3.common.env_util import make_vec_env
from poppy_torso_ik_env import PoppyTorsoIKEnv

# Create environment
env = PoppyTorsoIKEnv(render=True)

# Print available devices
print("CUDA Available:", torch.cuda.is_available())
print("Device being used:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

# Initialize model with GPU support
model = DDPG(
    "MultiInputPolicy",
    env,
    replay_buffer_class=HerReplayBuffer,
    replay_buffer_kwargs=dict(
        n_sampled_goal=4,
        goal_selection_strategy=GoalSelectionStrategy.FUTURE,
    ),
    verbose=1,
    device="cuda",
    learning_rate=1e-3,
    buffer_size=100000,
    learning_starts=1000,
    batch_size=256,
    gamma=0.98,
    tau=0.05,
    tensorboard_log="./ddpg_her_tensorboard/"  # <-- this adds TensorBoard logging
)



model.learn(total_timesteps=300_000, tb_log_name="poppy_ddpg_her")
model.save("ddpg_her_poppy")
