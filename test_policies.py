import argparse
import os

from multiagent.environment import MultiAgentEnv
from multiagent.scenarios.open_1_1_pursuit_know_vel import Scenario
from multiagent.simple_agents import StayAgent, ToPointAgent,VectorAgent
from multiagent.ddpg_agent import DDPGAgent
import baselines.common.tf_util as U
import tensorflow as tf
import evaluate_models

parser = argparse.ArgumentParser()
parser.add_argument('--actor_lr', default=0.0001, type=float)
parser.add_argument('--critic_lr', default=0.0001, type=float)

parser.add_argument('--episodes', default=10000000, type=int)
parser.add_argument('--evaluate_every_n_episodes', default=2500, type=int)
parser.add_argument('--max_steps', default=200, type=int)
parser.add_argument('--batch_size', default=256, type=int)
# parser.add_argument('--nb_rollout_steps', default=32, type=int)
parser.add_argument('--nb_train_steps', default=16, type=int)
parser.add_argument('--param_noise_adaption_interval', default=8, type=int)

parser.add_argument('--save_every_n_episodes', default=2500, type=int)
parser.add_argument('--load_weights', default=False)
parser.add_argument('--render', default=False, action="store_true")
parser.add_argument('--memory_size', default=1e6, type=int)
args = parser.parse_args()


def train(scenario):
    path_to_save = 'models/' + scenario.__module__.split('.')[-1] + '/simple'
    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)
    world = scenario.make_world()
    env = MultiAgentEnv(world, reset_callback=scenario.reset_world, reward_callback=scenario.reward,
                        observation_callback=scenario.observation, info_callback=None,
                        done_callback=scenario.done,
                        shared_viewer=True, )

    with U.single_threaded_session() as sess:
        simple_agents = [VectorAgent(env, 0, 1), StayAgent(env, 1)]  # good agent
        evaluator = evaluate_models.Evaluator(args, scenario, save=scenario.name + '/' + str(1))
        evaluator.evaluate(env, simple_agents, 0)


if __name__ == '__main__':
    scenario = Scenario()
    train(scenario)
