import argparse
import os

from multiagent.environment import MultiAgentEnv
from multiagent.scenarios.my_Scenario import MyScenario
from multiagent.scenarios.open_1_1_without_vel import Scenario
from multiagent.simple_agents import StayAgent, ToPointAgent
from multiagent.ddpg_agent import DDPGAgent
import baselines.common.tf_util as U
import tensorflow as tf
import evaluate_models
import numpy as np
from mpi4py import MPI

parser = argparse.ArgumentParser()
parser.add_argument('--learning_rate', default=0.001, type=float)
parser.add_argument('--episodes', default=10000000, type=int)
parser.add_argument('--evaluate_every_n_episodes', default=5000, type=int)
parser.add_argument('--max_steps', default=64, type=int)
parser.add_argument('--batch_size', default=128, type=int)
# parser.add_argument('--nb_rollout_steps', default=32, type=int)
parser.add_argument('--nb_train_steps', default=16, type=int)
parser.add_argument('--param_noise_adaption_interval', default=8, type=int)

parser.add_argument('--save_every_n_episodes', default=5000, type=int)
parser.add_argument('--load_weights', default=False)
parser.add_argument('--render', default=False, action="store_true")
parser.add_argument('--memory_size', default=50000, type=int)
args = parser.parse_args()


def train(scenario):
    path_to_save = 'models/' + scenario.__module__.split('.')[-1] + '/ddpg'
    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)
    world = scenario.make_world()
    env = MultiAgentEnv(world, reset_callback=scenario.reset_world, reward_callback=scenario.reward,
                        observation_callback=scenario.observation, info_callback=None,
                        done_callback=scenario.done,
                        shared_viewer=True)

    with U.single_threaded_session() as sess:
        simple_agents = [StayAgent(env, 1)]
        agents_with_nn = [DDPGAgent(env, 0, sess, batch_size=args.batch_size, memory_size=args.memory_size)]
        policies = [agents_with_nn[0], simple_agents[0]]
        print('agents is created')

        # for agent in agents_with_nn:
        #     agent.agent.initialize(sess)

        saver = tf.train.Saver()
        if args.load_weights:
            saver.restore(sess,
                          'models/' + scenario.name + '/ddpg/model')
        sess.graph.finalize()
        # for agent in agents_with_nn:
        #     agent.agent.reset()

        for episode in range(args.episodes):
            print('episode ' + str(episode))
            # reset
            for agent in policies:
                agent.reset()
            states = env.reset()

            step = 0
            while True:
                step += 1
                env_done = False
                # choose actions
                if args.render:
                    env.render()
                actions = [None for _ in range(len(world.policy_agents))]
                for agent in simple_agents:
                    actions[agent.agent_index] = (agent.action(states[agent.agent_index]))
                for agent in agents_with_nn:
                    action, q = agent.action(states[agent.agent_index], apply_noise=True, compute_Q=True)
                    actions[agent.agent_index] = action

                # step
                states_next, rewards, done, info = env.step(actions)
                # save to memory
                # print(rewards)

                if step >= args.max_steps:
                    env_done = True
                for agent in agents_with_nn:
                    if done[agent.agent_index]:
                        # rewards[agent.agent_index] -= 500
                        env_done = True

                if env_done:
                    for agent in agents_with_nn:
                        agent.agent.store_transition(states[agent.agent_index], actions[agent.agent_index],
                                                     rewards[agent.agent_index], states_next[agent.agent_index],
                                                     done[agent.agent_index])
                    break
                else:
                    states = states_next
            # learn
            # Adapt param noise, if necessary.
            for t_train in range(args.nb_train_steps):
                for agent in agents_with_nn:
                    if agent.agent.memory.nb_entries >= args.batch_size:
                        if episode % args.param_noise_adaption_interval == 0:
                            distance = agent.agent.adapt_param_noise()
                        # print('train')
                        cl, al = agent.agent.train()
                        agent.agent.update_target_net()

            if episode % args.save_every_n_episodes == 0:
                saver.save(sess, 'models/' + scenario.__module__.split('.')[-1] + '/ddpg/model')

            if args.evaluate_every_n_episodes != 0 and episode % args.evaluate_every_n_episodes == 0:
                evaluate_models.evaluate(scenario, env, policies,
                                         save='saved_trajectories/' + scenario.name + '_ddpg_and_simple_' + str(
                                             episode))

        saver.save(sess, 'models/' + scenario.__module__.split('.')[-1] + '/ddpg/model')


if __name__ == '__main__':
    scenario = Scenario()
    train(scenario)
