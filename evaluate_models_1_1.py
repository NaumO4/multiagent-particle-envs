import os
from os.path import join

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import utilities


class Evaluator:

    def __init__(self, args, scenario, max_steps=200, save=None, save_statistic=None):
        if save is not None:
            self.path_to_save = 'statistic/' + save
            self.path_to_save_trajectories = self.path_to_save + '/trajectories'
            self.path_to_save_statistic = self.path_to_save + '/statistic.csv'
            self.path_to_save_args = self.path_to_save + '/args.csv'
            if not os.path.exists(self.path_to_save):
                os.makedirs(self.path_to_save)
                os.makedirs(self.path_to_save_trajectories)
            utilities.save_dict(vars(args), self.path_to_save_args)
        columns = ['episode']
        for i in range(len(scenario.evaluate_corners)):
            columns.append('steps_' + str(scenario.evaluate_corners[i]))
            columns.append('rewards_' + str(scenario.evaluate_corners[i]))

        self.statistic = pd.DataFrame(columns=columns)
        self.save = save
        self.max_steps = max_steps
        self.scenario = scenario

    def evaluate(self, env, policies, episode):
        self.scenario.evaluate = True
        evaluate_episodes = len(self.scenario.evaluate_corners)
        sorted(policies, key=(lambda x: x.agent_index))
        statistic_of_episode = {'episode': episode}
        for ev_episode in range(evaluate_episodes):
            self.scenario.evaluate_episode = ev_episode
            step = 0
            observations = list()
            for agent in policies:
                agent.reset()
            obs = env.reset()

            # add observation
            positions_in_step = list()
            for i in range(len(env.world.agents)):
                positions_in_step.append(env.world.agents[i].state.p_pos[0])
                positions_in_step.append(env.world.agents[i].state.p_pos[1])
            observations.append(positions_in_step)

            episode_reward = np.zeros(env.n)

            while True:
                # query for action from each agent's policy
                step += 1
                act_n = []
                for i, policy in enumerate(policies):
                    act_n.append(policy.action(obs[i]))
                if step == 1 and episode == 0:
                    print('actions: ')
                    print(act_n)
                # step environment
                obs, reward, done, _ = env.step(act_n)

                episode_reward += np.array(reward)

                # add observation
                positions_in_step = list()
                for i in range(len(env.world.agents)):
                    positions_in_step.append(env.world.agents[i].state.p_pos[0])
                    positions_in_step.append(env.world.agents[i].state.p_pos[1])

                observations.append(positions_in_step)

                env_done = False
                if step >= self.max_steps:
                    env_done = True
                if all(done):
                    env_done = True
                if env_done:
                    episode_reward = episode_reward / step
                    print('episode_reward', episode_reward)
                    print('step', step)
                    break

            if self.save is not None:
                print(self.save.split('/'))
                path_to_save = self.path_to_save_trajectories + '/' + str(episode) + '/'
                if not os.path.exists(path_to_save):
                    os.makedirs(path_to_save)
                observations = np.array(observations)
                observations = np.reshape(observations, (observations.shape[0], -1))
                print('obs shape: ' + str(observations.shape))
                np.savetxt((path_to_save + str(self.scenario.evaluate_corners[ev_episode])), observations,
                           delimiter=",")
                statistic_of_episode['steps_' + str(self.scenario.evaluate_corners[ev_episode])] = step
                statistic_of_episode['rewards_' + str(self.scenario.evaluate_corners[ev_episode])] = str(episode_reward)

        if self.save is not None:
            self.statistic = self.statistic.append(
                statistic_of_episode, ignore_index=True)
            self.statistic.to_csv(self.path_to_save_statistic)
        self.scenario.evaluate = False


def show_all_on_folder(sc, folder, corner='6.283185307179586'):
    world = sc.make_world()
    # folder = 'multiagent/saved_trajectories/' + sc.name + '/ddpg_and_simple'
    trjs = os.listdir(folder)
    trjs = map(lambda x: int(x), trjs)
    trjs = sorted(trjs)
    for trj in trjs:
        data = np.loadtxt(join(folder, str(trj), corner), delimiter=',')
        print(trj)
        print(data.shape)
        ax = plt.gca()
        for i, agent in enumerate(world.agents):
            x = data[:, i * 2]
            y = data[:, i * 2 + 1]
            plt.plot(x, y, 'b-', color=agent.color)
            circle = plt.Circle((x[-1], y[-1]), radius=agent.size, color=agent.color)
            ax.add_patch(circle)
        plt.title(str(trj) + ' iteration')
        plt.show()


def show_time_of_each_corner(sc, path_to_statistic, path_to_teoretic_results):
    world = sc.make_world()
    statistic = pd.read_csv(path_to_statistic)
    tr = pd.read_csv(path_to_teoretic_results)

    columns = list(statistic.columns.values.tolist())
    columns = list(filter(lambda x: x.startswith('steps_'), columns))
    corners = np.array(list(map(lambda x: float(x[6:]), columns)))
    statistic = statistic.iloc[6]
    tr = tr.iloc[0]
    plt.title('Залежність часу від напрямку в якому рухається втікач')
    plt.ylabel('Час, t')
    plt.xlabel('Кут, рад')

    #corners -= np.pi / 2
    print(len(columns))
    # corners = corners[6:-6]
    # columns = columns[6:-6]
    plt.plot(corners,statistic[columns],corners,tr[columns],  'b-')
    plt.legend(('DDPG', 'Теоретичні результати'),
               loc='upper right')
    plt.show()

if __name__ == '__main__':
    from multiagent.scenarios.open_1_1_pursuit_know_vel import Scenario

    show_time_of_each_corner(Scenario(), 'multiagent/statistic/open_1_1_pursuit_know_vel_REWARD_FOR_COLISION_500/1/statistic.csv', 'multiagent/statistic/open_1_1_pursuit_know_vel_REWARD_FOR_COLISION_500/0/statistic.csv')

    folder = 'multiagent/statistic/open_1_1_pursuit_know_vel_REWARD_FOR_COLISION_500/1/trajectories/'
    sc = Scenario()
    #show_all_on_folder(sc, folder)
