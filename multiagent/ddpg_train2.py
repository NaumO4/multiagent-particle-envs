import argparse
import os
import numpy as np

import utilities
from multiagent.environment import MultiAgentEnv
from multiagent.scenarios.open_1_2_pursuit_know_vel import Scenario
from multiagent.simple_agents import StayAgent, ToPointAgent, ToPointsAgent
from multiagent.ddpg_agent import DDPGAgent
import baselines.common.tf_util as U
import tensorflow as tf
import evaluate_models
from evaluate_models import Evaluator

parser = argparse.ArgumentParser()
parser.add_argument('--actor_lr', default=1e-4, type=float)
parser.add_argument('--critic_lr', default=1e-3, type=float)

parser.add_argument('--episodes', default=200000, type=int)
parser.add_argument('--evaluate_every_n_episodes', default=2500, type=int)
parser.add_argument('--max_steps', default=500, type=int)
parser.add_argument('--batch_size', default=512, type=int)
# parser.add_argument('--nb_rollout_steps', default=32, type=int)
parser.add_argument('--nb_train_steps', default=16, type=int)
parser.add_argument('--param_noise_adaption_interval', default=8, type=int)
parser.add_argument('--experiment_prefix', default='/statistics/', type=str)
parser.add_argument('--noise_type', default='adaptive-param_0.2', type=str)
parser.add_argument('--nb_layers', default=2, type=int)
parser.add_argument('--nb_neurons', default=128, type=int)
parser.add_argument('--activation', default='tanh', type=str)


parser.add_argument('--save_every_n_episodes', default=2500, type=int)
parser.add_argument('--load_weights', default=False)
parser.add_argument('--render', default=False, action="store_true")
parser.add_argument('--memory_size', default=1e6, type=int)
args = parser.parse_args()


def train(scenario):
    path_to_save = 'models/' + scenario.__module__.split('.')[-1] + '/ddpg'
    train_n = 9
    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)
    world = scenario.make_world()
    env = MultiAgentEnv(world, reset_callback=scenario.reset_world, reward_callback=scenario.reward,
                        observation_callback=scenario.observation, info_callback=None,
                        done_callback=scenario.done, collision_callback=scenario.is_collision,
                        shared_viewer=True, )
    evaluator = Evaluator(args, scenario, save=scenario.name + '/' + str(train_n),max_steps=args.max_steps)

    with U.single_threaded_session() as sess:
        simple_agents = [ToPointsAgent(env, 0, [1, 2])]  # adversary
        from copy import deepcopy
        ddpg_agent = DDPGAgent(env, 1, sess, batch_size=args.batch_size, memory_size=args.memory_size,
                               noise_type=args.noise_type,
                               # good agent
                               actor_lr=args.actor_lr, critic_lr=args.critic_lr, layer_norm=True,
                               nb_layers=args.nb_layers, nb_neurons=args.nb_neurons, activation=args.activation)
        ddpg_agent2 = DDPGAgent(env, 2, sess, batch_size=args.batch_size, memory_size=args.memory_size,
                                noise_type=args.noise_type,
                                # good agent
                                actor_lr=args.actor_lr, critic_lr=args.critic_lr, layer_norm=True,
                                nb_layers=args.nb_layers, nb_neurons=args.nb_neurons, activation=args.activation)
        # ddpg_agent3 = DDPGAgent(env, 3, sess, batch_size=args.batch_size, memory_size=args.memory_size,
        #                         noise_type=args.noise_type,
        #                         # good agent
        #                         actor_lr=args.actor_lr, critic_lr=args.critic_lr, layer_norm=True,
        #                         nb_layers=args.nb_layers, nb_neurons=args.nb_neurons)
        # ddpg_agent2.agent = ddpg_agent.agent

        agents_with_nn = [
            ddpg_agent, ddpg_agent2]
        policies = simple_agents + agents_with_nn
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

        statistics_header = ["episode"]
        statistics_header.append("steps")
        statistics_header.extend(["reward_{}".format(i) for i in range(env.n)])
        statistics_header.extend(["q_{}".format(i) for i in range(env.n)])
        statistics = utilities.Time_Series_Statistics_Store(
            statistics_header)

        for episode in range(args.episodes):
            if episode % 500 == 0:
                print('episode ' + str(episode))
            # reset
            for agent in policies:
                agent.reset()
            states = env.reset()
            agent_is_done = np.full((len(policies),), False)

            step = 0
            while True:
                episode_q = np.zeros(env.n)
                episode_rewards = np.zeros(env.n)

                step += 1
                env_done = False
                # choose actions
                if args.render:
                    env.render()
                actions = [None for _ in range(len(world.policy_agents))]
                for agent in simple_agents:
                    actions[agent.agent_index] = (agent.action(states[agent.agent_index]))
                    episode_q[0] += 0
                for agent in agents_with_nn:
                    # if agent_is_done[agent.agent_index]:
                    #     actions[agent.agent_index] = np.zeros((4,))
                    #     continue
                    action, q = agent.action(states[agent.agent_index], apply_noise=True, compute_Q=True)
                    actions[agent.agent_index] = action
                    episode_q[agent.agent_index] += q
                    # print(agent.agent_index)
                    # print(action)

                # step
                states_next, rewards, done, info = env.step(actions)
                # save to memory
                # print(rewards)
                episode_rewards += rewards

                for agent in agents_with_nn:
                    if agent_is_done[agent.agent_index]:
                        continue
                    agent.agent.store_transition(states[agent.agent_index],
                                                 actions[agent.agent_index],
                                                 rewards[agent.agent_index],
                                                 states_next[agent.agent_index],
                                                 done[agent.agent_index])

                for agent in agents_with_nn:
                    if done[agent.agent_index]:
                        # rewards[agent.agent_index] -= 500
                        agent_is_done[agent.agent_index] = True

                env_done = done[0]
                if step >= args.max_steps:
                    env_done = True

                states = states_next
                if env_done:
                    episode_rewards = episode_rewards / step
                    episode_losses = episode_q / step
                    statistic = [episode]
                    statistic.append(step)
                    statistic.extend([episode_rewards[i] for i in range(env.n)])
                    statistic.extend([episode_q[i] for i in range(env.n)])
                    statistics.add_statistics(statistic)
                    break
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
                statistics.dump("{}_{}.csv".format(
                    args.experiment_prefix + scenario.__module__.split('.')[-1], episode))
                evaluator.evaluate(env, policies, episode)

        saver.save(sess, 'models/' + scenario.__module__.split('.')[-1] + '/ddpg/model')


if __name__ == '__main__':
    scenario = Scenario()
    train(scenario)
