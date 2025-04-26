from stable_baselines3 import SAC
from stable_baselines3.her import HerReplayBuffer
from poppy_torso_ik_env import PoppyTorsoIKEnv

# Use the same CustomReachEnv from earlier
print("Initializing PoppyTorsoIKEnv...")
env = PoppyTorsoIKEnv()

model = SAC(
    "MultiInputPolicy",
    env,
    replay_buffer_class=HerReplayBuffer,
    replay_buffer_kwargs=dict(
        n_sampled_goal=4,
        goal_selection_strategy="future",  # HER strategy
       
    ),
    verbose=1,
    buffer_size=100_000,  # You can increase this for more stability
    learning_rate=3e-4,
    batch_size=256,
    gamma=0.98,
    tau=0.05,
    train_freq=(1, "episode"),
    gradient_steps=1,
)

model.learn(total_timesteps=100_000)

print("Training complete. Saving model...")
model.save("poppy_sac_her")
print("Model saved.")
