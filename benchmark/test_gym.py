import tianshou

from robosuite.controllers import load_controller_config
from robosuite.utils.input_utils import *
from robosuite.wrappers import GymWrapper
from robosuite.macros import SIMULATION_TIMESTEP

import numpy as np
import torch

import time
import tqdm
import csv

import warnings
warnings.filterwarnings("ignore")

import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='Door')
    parser.add_argument('--num_envs', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--control_freq', type=int, default=20)
    parser.add_argument('--headless', action='store_true')
    args = parser.parse_args()

    return args


def main(args):
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    # print welcome info
    print("Welcome to robosuite v{}!".format(suite.__version__))
    print(suite.__logo__)
    print("Num envs:", args.num_envs)
    n_envs = args.num_envs

    # create environment instance
    robosuite_env = suite.make(
        env_name=args.task, # try with other tasks like "Stack" and "Door"
        robots="Panda",  # try with other robots like "Sawyer" and "Jaco"
        has_renderer=not args.headless and n_envs == 1,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        control_freq=args.control_freq,
        horizon=200,
        controller_configs=load_controller_config(default_controller="JOINT_TORQUE"),
    )
    env = GymWrapper(robosuite_env)

    # create vectorized environment
    if n_envs == 1:
        vec_env = tianshou.env.DummyVectorEnv([lambda: env for _ in range(n_envs)])
    else:
        vec_env = tianshou.env.SubprocVectorEnv([lambda: env for _ in range(n_envs)])

    
    vec_env.seed(args.seed)
    print('Created', args.task, 'with observation_space',
          env.observation_space.shape, 'action_space',
          env.action_space.shape)
    print("action_space", env.action_space)

    # frame skip is number of simulation steps per action
    # simulation freq / control freq
    frame_skip = 1 / float(args.control_freq) / SIMULATION_TIMESTEP
    total_step = 1000

    vec_env.reset()

    # Get action limits
    low, high = robosuite_env.action_spec

    t = time.perf_counter()
    for _ in tqdm.trange(total_step):
        action = np.random.uniform(low, high, size=(n_envs, env.action_space.shape[0]))
        obs, reward, done, _ = vec_env.step(action)
        terminated_envs = np.where(done)[0]
        if len(terminated_envs) > 0:
            vec_env.reset(id=np.where(done)[0])
        if n_envs == 1 and not args.headless:
            vec_env.render()
    # FPS
    time_elapsed = time.perf_counter() - t
    fps = frame_skip * total_step * n_envs / time_elapsed

    print(f"FPS = {fps:.2f}")

    vec_env.close()

    fieldnames = ['n_envs', 'frame_skip', 'total_step', 'time_elapsed', 'fps']
    # open the file in the write mode
    with open('results.csv', 'a') as f:
        # create the csv writer
        writer = csv.writer(f)
        # write header
        # writer.writerow(fieldnames)
        # data
        row = [n_envs, frame_skip, total_step, time_elapsed, fps]
        # write a row to the csv file
        writer.writerow(row)
    


if __name__ == "__main__":
    args = parse_args()
    main(args)
