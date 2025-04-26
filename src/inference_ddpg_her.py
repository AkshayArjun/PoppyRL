import time
import torch
from stable_baselines3 import DDPG
from stable_baselines3.common.vec_env import DummyVecEnv
from poppy_torso_ik_env import PoppyTorsoIKEnv



def main():
    # Check for GPU
    if torch.cuda.is_available():
        print("CUDA Available:", True)
        print("Using device:", torch.cuda.get_device_name(0))
        device = "cuda"
    else:
        print("CUDA Available:", False)
        print("Using device: CPU")
        device = "cpu"

    # Create dual-arm Poppy Torso environment
    env = PoppyTorsoIKEnv(render=True)

    # Vectorize the environment for SB3 compatibility
    env = DummyVecEnv([lambda: env])

    # Load the trained model
    model_path = "ddpg_her_poppy"
    print(f"Loading model from: {model_path}")
    model = DDPG.load(model_path, env=env, device=device)

    # Run inference loop
    obs = env.reset()
    timestep = 0
    max_timesteps = 1000

    while timestep < max_timesteps:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)

        # Log useful info
        print(f"Step {timestep} | Reward: {reward[0]:.4f} | Done: {done[0]}")

        # Sleep to match control frequency (~60Hz)
        time.sleep(1/60)

        if done[0]:
            print("Episode done. Resetting environment.")
            obs = env.reset()

        timestep += 1

    print("Inference complete.")


if __name__ == "__main__":
    main()