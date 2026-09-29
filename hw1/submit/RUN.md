# Reproducing results

All commands are run from the `hw1/` directory with the `rob831` conda env active,
using `--no_gpu` and `--video_log_freq -1`.

## Table 1 (Section 1, Question 2): expert return over 2 trajectories

Run the BC script once per environment with default hyperparameters
(`eval_batch_size=5000`, `num_agent_train_steps_per_iter=2000`, same as
Table 2) -- Table 1's numbers come from the `Train_AverageReturn` /
`Train_StdReturn` logged from the loaded `--expert_data`, and the resulting
`run_logs` are also real BC training runs, not throwaway expert-only logging
runs:

```
python rob831/scripts/run_hw1.py \
  --expert_policy_file rob831/policies/experts/Ant.pkl \
  --env_name Ant-v2 --exp_name bc_ant_default_5000_2000 --n_iter 1 \
  --expert_data rob831/expert_data/expert_data_Ant-v2.pkl \
  --eval_batch_size 5000 --num_agent_train_steps_per_iter 2000 \
  --video_log_freq -1 --no_gpu

python rob831/scripts/run_hw1.py \
  --expert_policy_file rob831/policies/experts/Humanoid.pkl \
  --env_name Humanoid-v2 --exp_name bc_humanoid_default_5000_2000 --n_iter 1 \
  --expert_data rob831/expert_data/expert_data_Humanoid-v2.pkl \
  --eval_batch_size 5000 --num_agent_train_steps_per_iter 2000 \
  --video_log_freq -1 --no_gpu

python rob831/scripts/run_hw1.py \
  --expert_policy_file rob831/policies/experts/Walker2d.pkl \
  --env_name Walker2d-v2 --exp_name bc_walker2d_default_5000_2000 --n_iter 1 \
  --expert_data rob831/expert_data/expert_data_Walker2d-v2.pkl \
  --eval_batch_size 5000 --num_agent_train_steps_per_iter 2000 \
  --video_log_freq -1 --no_gpu

python rob831/scripts/run_hw1.py \
  --expert_policy_file rob831/policies/experts/Hopper.pkl \
  --env_name Hopper-v2 --exp_name bc_hopper_default_5000_2000 --n_iter 1 \
  --expert_data rob831/expert_data/expert_data_Hopper-v2.pkl \
  --eval_batch_size 5000 --num_agent_train_steps_per_iter 2000 \
  --video_log_freq -1 --no_gpu

python rob831/scripts/run_hw1.py \
  --expert_policy_file rob831/policies/experts/HalfCheetah.pkl \
  --env_name HalfCheetah-v2 --exp_name bc_halfcheetah_default_5000_2000 --n_iter 1 \
  --expert_data rob831/expert_data/expert_data_HalfCheetah-v2.pkl \
  --eval_batch_size 5000 --num_agent_train_steps_per_iter 2000 \
  --video_log_freq -1 --no_gpu
```


## Table 2 (Section 1, Question 3): BC vs. expert on Ant-v2 and Humanoid-v2

Default network (2 layers, 64 units), default learning rate (5e-3), 1 BC
iteration, `eval_batch_size=5000` (5 eval rollouts), `num_agent_train_steps_per_iter=2000`.
We can now compare learned policy with expert policy return and std deviation.

```
python rob831/scripts/run_hw1.py \
  --expert_policy_file rob831/policies/experts/Ant.pkl \
  --env_name Ant-v2 --exp_name bc_ant_default_5000_2000 --n_iter 1 \
  --expert_data rob831/expert_data/expert_data_Ant-v2.pkl \
  --eval_batch_size 5000 --num_agent_train_steps_per_iter 2000 \
  --video_log_freq -1 --no_gpu

python rob831/scripts/run_hw1.py \
  --expert_policy_file rob831/policies/experts/Humanoid.pkl \
  --env_name Humanoid-v2 --exp_name bc_humanoid_default_5000_2000 --n_iter 1 \
  --expert_data rob831/expert_data/expert_data_Humanoid-v2.pkl \
  --eval_batch_size 5000 --num_agent_train_steps_per_iter 2000 \
  --video_log_freq -1 --no_gpu
```

## Figure 1 (Section 1, Question 4): Rollout-length ablation on Humanoid-v2

Same hyperparameters as above. Two training-data conditions, same total of
2000 transitions:

- **Fewer, longer**: Behavior Cloning using `expert_data_Humanoid-v2.pkl` with (2 traj x 1000 steps) - same as table2
- **More, shorter**: Behavior Cloning using `expert_data_Humanoid-v2_more_shorter.pkl` with (20 traj x 100 steps)

Since the code just loads expert data for training and considering eps length during data aggregation only, 
I generated a new set of data with the expert policy using shorter eps length and more rollouts:

```python
import gym
from rob831.policies.loaded_gaussian_policy import LoadedGaussianPolicy
from rob831.infrastructure import utils
import pickle

env = gym.make('Humanoid-v2')
env.reset(seed=1)
policy = LoadedGaussianPolicy('rob831/policies/experts/Humanoid.pkl')
paths = utils.sample_n_trajectories(env, policy, ntraj=20, max_path_length=100, render=False)
with open('rob831/expert_data/expert_data_Humanoid-v2_more_shorter.pkl', 'wb') as f:
    pickle.dump(paths, f)
```

Then train BC on it:
```
python rob831/scripts/run_hw1.py \
  --expert_policy_file rob831/policies/experts/Humanoid.pkl \
  --env_name Humanoid-v2 --exp_name bc_humanoid_more_shorter --n_iter 1 \
  --expert_data rob831/expert_data/expert_data_Humanoid-v2_more_shorter.pkl \
  --eval_batch_size 5000 --num_agent_train_steps_per_iter 2000 \
  --video_log_freq -1 --no_gpu
```

Bar chart plotted from the two runs' `Eval_AverageReturn` / `Eval_StdReturn`.

## Figure 2 (Section 2, Question 2): DAgger learning curves on Ant-v2 and Humanoid-v2

All default hyperparameters (2 layers, 64 units, learning rate 5e-3,
`num_agent_train_steps_per_iter=1000`, `train_batch_size=100`,
`batch_size=1000`), except `eval_batch_size=5000` (5 eval trajectories) and
`n_iter=10`, `--do_dagger`. Seed defaults to 1, so this run is reproducible
(verified: rerunning gives bit-identical `Eval_AverageReturn`/`Eval_StdReturn`
values).

```
python rob831/scripts/run_hw1.py \
  --expert_policy_file rob831/policies/experts/Ant.pkl \
  --env_name Ant-v2 --exp_name dagger_ant --n_iter 10 \
  --do_dagger --expert_data rob831/expert_data/expert_data_Ant-v2.pkl \
  --eval_batch_size 5000 --video_log_freq -1 --no_gpu

python rob831/scripts/run_hw1.py \
  --expert_policy_file rob831/policies/experts/Humanoid.pkl \
  --env_name Humanoid-v2 --exp_name dagger_humanoid --n_iter 10 \
  --do_dagger --expert_data rob831/expert_data/expert_data_Humanoid-v2.pkl \
  --eval_batch_size 5000 --video_log_freq -1 --no_gpu
```

Learning curves plot `Eval_AverageReturn` +/- `Eval_StdReturn` per DAgger
iteration (0-9) from each run's tfevents file, with the Table 2 BC/Expert
values overlaid as horizontal reference lines.

## run_logs included in this submission

- Behavior cloning Ant-v2: `run_logs/q1_bc_ant_default_5000_2000_Ant-v2_.../` 
- Behavior cloning Humanoid-v2: `run_logs/q1_bc_humanoid_default_5000_2000_Humanoid-v2_.../`
- Behavior cloning Walker2d-v2 (Table 1): `run_logs/q1_bc_walker2d_default_5000_2000_Walker2d-v2_.../`
- Behavior cloning Hopper-v2 (Table 1): `run_logs/q1_bc_hopper_default_5000_2000_Hopper-v2_.../`
- Behavior cloning HalfCheetah-v2 (Table 1): `run_logs/q1_bc_halfcheetah_default_5000_2000_HalfCheetah-v2_.../`
- DAgger (Ant-v2): `run_logs/q2_dagger_ant_Ant-v2_.../` 
- DAgger (Humanoid-v2): `run_logs/q2_dagger_humanoid_Humanoid-v2_.../`
