# view_policy.py
import os
os.environ.setdefault("MUJOCO_GL", "osmesa")  # offscreen software renderer — sidesteps the
                                               # broken onscreen GLFW/GLEW path entirely

import torch
import gym
import imageio
from rob831.policies.MLP_policy import MLPPolicySL
from rob831.infrastructure import pytorch_util as ptu
from rob831.infrastructure.utils import sample_trajectory

HW1_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CHECKPOINT = os.path.join(HW1_DIR, "data/q1_bc_ant_Ant-v2_17-09-2026_02-16-05/policy_itr_0.pt")
OUTPUT_VIDEO = os.path.join(HW1_DIR, "rollout.mp4")
N_LAYERS = 5   # <- match what you trained with
SIZE = 64      # <- match what you trained with

ptu.init_gpu(use_gpu=False)

env = gym.make('Ant-v2')
discrete = isinstance(env.action_space, gym.spaces.Discrete)
ob_dim = env.observation_space.shape[0]
ac_dim = env.action_space.n if discrete else env.action_space.shape[0]

policy = MLPPolicySL(ac_dim, ob_dim, n_layers=N_LAYERS, size=SIZE, discrete=discrete)
policy.load_state_dict(torch.load(CHECKPOINT, map_location="cpu"))
policy.eval()

path = sample_trajectory(env, policy, max_path_length=1000, render=True, render_mode='rgb_array')
env.close()

print(f"Episode length: {len(path['reward'])} steps, return: {path['reward'].sum():.2f}")
imageio.mimsave(OUTPUT_VIDEO, path['image_obs'], fps=30)
print(f"Saved rollout video to {OUTPUT_VIDEO}")
