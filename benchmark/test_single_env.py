"""
Simple script to measure single-env FPS in robosuite.
"""
import time
import tqdm
import csv
import json

import warnings
warnings.filterwarnings("ignore")

import argparse

import numpy as np
import torch

import robosuite
from robosuite.controllers import load_controller_config
from robosuite.utils.input_utils import *
from robosuite.macros import SIMULATION_TIMESTEP


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='Door')
    parser.add_argument('--num_envs', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--control_freq', type=int, default=30)
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--num_trials', type=int, default=5)
    args = parser.parse_args()

    return args


def run_trial(env, headless, low, high, horizon):
    env.reset()

    # Get action limits
    low, high = env.action_spec

    # for _ in tqdm.trange(horizon):
    for _ in range(horizon):
        # action = np.random.uniform(low, high, size=(n_envs, env.action_space.shape[0]))
        action = np.random.uniform(low, high, size=(low.shape[0]))
        obs, reward, done, _ = env.step(action)
        # terminated_envs = np.where(done)[0]
        # if len(terminated_envs) > 0:
        #     env.reset(id=np.where(done)[0])
        if not headless:
            env.render()


def main(args):
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    # print welcome info
    print("Welcome to robosuite v{}!".format(suite.__version__))
    print(suite.__logo__)
    print("Num envs:", args.num_envs)
    n_envs = args.num_envs

    # create environment instance
    horizon = 1000
    env = suite.make(
        env_name=args.task, # try with other tasks like "Stack" and "Door"
        robots="Panda",  # try with other robots like "Sawyer" and "Jaco"
        has_renderer=not args.headless and n_envs == 1,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        control_freq=args.control_freq,
        # horizon=200,
        horizon=horizon,
        controller_configs=load_controller_config(default_controller="JOINT_TORQUE"),
    )

    # frame skip is number of simulation steps per action
    # simulation freq / control freq
    frame_skip = 1 / float(args.control_freq) / SIMULATION_TIMESTEP
    print("Created task env {} with control freq {}, sim timestep {}, and frame_skip {}".format(args.task, args.control_freq, SIMULATION_TIMESTEP, frame_skip))

    # Get action limits
    low, high = env.action_spec

    measurements = dict(time_elapsed=[], fps=[])
    for _ in tqdm.tqdm(range(args.num_trials)):
        t = time.perf_counter()
        run_trial(env, headless=args.headless, low=low, high=high, horizon=horizon)
        # FPS
        time_elapsed = time.perf_counter() - t
        fps = frame_skip * horizon * n_envs / time_elapsed
        measurements["time_elapsed"].append(time_elapsed)
        measurements["fps"].append(fps)
    
    avg_fps = np.mean(measurements["fps"])
    print("all FPS: {}".format(measurements["fps"]))
    print("avg FPS = {:.2f}".format(avg_fps))

    env.close()

    results = dict(
        n_envs=n_envs,
        frame_skip=frame_skip,
        total_step=horizon,
        fps=avg_fps,
    )
    print(json.dumps(results, indent=4))


if __name__ == "__main__":
    args = parse_args()
    main(args)