import numpy as np
import random

from multiagent.policy import Policy


class StayAgent(Policy):
    def __init__(self, env, agent_index):
        super(StayAgent, self).__init__(agent_index)
        self.env = env
        if self.env.discrete_action_input:
            self.u = random.randint(1, 4)
        else:
            self.u = np.zeros(5)  # 5-d because of no-move action
            self.u[1] = random.random() * 2 - 1
            self.u[2] = 0.0
            self.u[3] = random.random() * 2 - 1
            self.u[4] = 0.0

    def action(self, obs):
        # ignore observation and just act based on keyboard events
        # if self.env.discrete_action_input:
        #     u = 1
        # else:
        #     u = np.zeros(5)  # 5-d because of no-move action
        #     u[1] = 1
        #     u[2] = 0.0
        #     u[3] = 0.0
        #     u[4] = 0.0
        return np.concatenate([self.u, np.zeros(self.env.world.dim_c)])

    def reset(self):
        if self.env.discrete_action_input:
            self.u = random.randint(1, 4)
        else:
            self.u = np.zeros(5)  # 5-d because of no-move action
            self.u[1] = random.random() * 2 - 1
            self.u[2] = 0.0
            self.u[3] = random.random() * 2 - 1
            self.u[4] = 0.0


class VectorAgent:
    pass


class ToPointAgent(Policy):
    def __init__(self, env, agent_index, advertisal_agent_index):
        super(ToPointAgent, self).__init__(agent_index)
        self.advertisal_agent_index = advertisal_agent_index
        self.env = env
        self.agent_index = agent_index

    def action(self, obs):
        # ignore observation and just act based on keyboard events
        u = np.zeros(5)
        pos1 = self.env.agents[self.agent_index].state.p_pos
        pos2 = self.env.agents[self.advertisal_agent_index].state.p_pos
        xd = pos2[1] - pos1[1]
        yd = pos2[0] - pos1[0]
        k = np.sqrt(xd * xd + yd * yd)
        xd = k * xd
        yd = k * yd
        u[3] = xd
        u[1] = yd

        return np.concatenate([u, np.zeros(self.env.world.dim_c)])
