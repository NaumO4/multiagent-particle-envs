import numpy as np
import matplotlib.pyplot as plt


def evaluate(scenario, env, policies, max_steps=64, save=None):
    scenario.evaluate = True
    sorted(policies, key=(lambda x: x.agent_index))

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

    while True:
        # query for action from each agent's policy
        step += 1
        act_n = []
        for i, policy in enumerate(policies):
            act_n.append(policy.action(obs[i]))
        # step environment
        obs, reward, done, _ = env.step(act_n)

        # add observation
        positions_in_step = list()
        for i in range(len(env.world.agents)):
            positions_in_step.append(env.world.agents[i].state.p_pos[0])
            positions_in_step.append(env.world.agents[i].state.p_pos[1])

        observations.append(positions_in_step)

        env_done = False
        if step >= max_steps:
            env_done = True
        for agent in policies:
            if done[agent.agent_index]:
                env_done = True
        if env_done:
            break

    if save is not None:
        observations = np.array(observations)
        observations = np.reshape(observations, (observations.shape[0], -1))
        print('obs shape: ' + str(observations.shape))
        np.savetxt(save, observations, delimiter=",")
    scenario.evaluate = False


if __name__ == '__main__':
    data = np.loadtxt('multiagent/saved_trajectories/open_1_1_without_vel_ddpg_and_simple_0', delimiter=',')
    print(data.shape)
    x = data[:, 0]
    y = data[:, 1]
    x2 = data[:, 2]
    y2 = data[:, 3]
    print(x.shape)

    plt.plot(x, y, 'b-')
    plt.plot(x2, y2, 'r-')

    plt.show()
    #fig, ax = plt.subplots(figsize=(10, 10))
    # data.plot(color='blue',
    #           ax=ax,
    #           label='ddpg')
