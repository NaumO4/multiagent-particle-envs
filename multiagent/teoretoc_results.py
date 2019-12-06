import argparse
import numpy as np
import os

import utilities
from multiagent.environment import MultiAgentEnv
from multiagent.scenarios.open_1_1_pursuit_know_vel import Scenario
from multiagent.simple_agents import StayAgent, ToPointAgent,StayForceAgent, VectorAgent
from multiagent.ddpg_agent import DDPGAgent
import baselines.common.tf_util as U
import tensorflow as tf
import evaluate_models_1_1 as evaluate_models

parser = argparse.ArgumentParser()
parser.add_argument('--actor_lr', default=1e-4, type=float)
parser.add_argument('--critic_lr', default=1e-3, type=float)

parser.add_argument('--episodes', default=10000000, type=int)
parser.add_argument('--evaluate_every_n_episodes', default=500, type=int)
parser.add_argument('--max_steps', default=200, type=int)
parser.add_argument('--batch_size', default=512, type=int)
# parser.add_argument('--nb_rollout_steps', default=32, type=int)
parser.add_argument('--nb_train_steps', default=16, type=int)
parser.add_argument('--param_noise_adaption_interval', default=8, type=int)
parser.add_argument('--experiment_prefix', default='/statistics/', type=str)
parser.add_argument('--noise_type', default='adaptive-param_0.2', type=str)
parser.add_argument('--nb_layers', default=2, type=str)
parser.add_argument('--nb_neurons', default=64, type=str)

parser.add_argument('--save_every_n_episodes', default=500, type=int)
parser.add_argument('--load_weights', default=False)
parser.add_argument('--render', default=False, action="store_true")
parser.add_argument('--memory_size', default=1e6, type=int)
args = parser.parse_args()


def train(scenario):
    train_n = 0
    world = scenario.make_world()
    env = MultiAgentEnv(world, reset_callback=scenario.reset_world, reward_callback=scenario.reward,
                        observation_callback=scenario.observation, info_callback=None,
                        done_callback=scenario.done, collision_callback=scenario.is_collision,
                        shared_viewer=True, )
    evaluator = evaluate_models.Evaluator(args, scenario, save=scenario.name + '/' + str(train_n))


    simple_agents = [StayAgent(env, 1), VectorAgent(env,0,1)] #good agent

    policies = [simple_agents[1], simple_agents[0]]
    print('agents is created')
    statistics_header = ["episode"]
    statistics_header.append("steps")
    statistics_header.extend(["reward_{}".format(i) for i in range(env.n)])
    statistics_header.extend(["q_{}".format(i) for i in range(env.n)])
    statistics = utilities.Time_Series_Statistics_Store(
        statistics_header)

    statistics.dump("{}_{}.csv".format(
        args.experiment_prefix + scenario.__module__.split('.')[-1], 0))
    evaluator.evaluate(env, policies, 0)


if __name__ == '__main__':
    scenario = Scenario()
    train(scenario)
