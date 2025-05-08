from gymnasium.utils.env_checker import check_env
from poppy_torso_ik_env import PoppyTorsoIKEnv
from poppy_env import PoppyEnv

check_env(PoppyEnv(), warn=True)
print("Environment check complete.")

