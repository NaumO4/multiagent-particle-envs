import numpy as np
import random

from multiagent.policy import Policy


class StayAgent(Policy):
    def __init__(self, env, agent_index):
        super(StayAgent, self).__init__(agent_index)
        self.env = env
        self.reset()

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
            self.u[0] = random.random() * 2 - 1
            self.u[1] = 0.0
            self.u[2] = random.random() * 2 - 1
            self.u[3] = 0.0


class StayForceAgent(Policy):
    # action return force always in same direction
    def __init__(self, env, agent_index):
        super(StayForceAgent, self).__init__(agent_index)
        self.env = env
        self.reset()

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
            self.u[0] = random.random() * 2 - 1
            self.u[1] = 0.0
            self.u[2] = random.random() * 2 - 1
            self.u[3] = 0.0
            self.u[0] = self.u[0] / (np.sqrt(np.square(self.u[0]) + np.square(self.u[2])))
            self.u[2] = self.u[2] / (np.sqrt(np.square(self.u[0]) + np.square(self.u[2])))


class VectorAgent(Policy):
    def __init__(self, env, agent_index, advertisal_agent_index):
        super(VectorAgent, self).__init__(agent_index)
        self.advertisal_agent_index = advertisal_agent_index
        self.env = env
        self.agent_index = agent_index

    "action return speed the fastest to get to the point if we know velocity"
    def action(self, obs):
        # ignore observation and just act based on keyboard events
        u = np.zeros(4)
        pos1 = self.env.agents[self.agent_index].state.p_pos
        pos2 = self.env.agents[self.advertisal_agent_index].state.p_pos
        xd = pos2[1] - pos1[1]
        yd = pos2[0] - pos1[0]
        distance = np.sqrt(xd * xd + yd * yd)
        if distance == 0:
            return u
        v1 = self.env.agents[self.advertisal_agent_index].max_speed
        v2 = self.env.agents[self.agent_index].max_speed
        scalar = xd * self.env.agents[self.advertisal_agent_index].state.p_vel[1] + yd * \
                 self.env.agents[self.advertisal_agent_index].state.p_vel[0]
        print(((np.sqrt(v1 * v1 + v2 * v2 - 2 * v2 * scalar / distance))))
        # if (np.sqrt(v1 * v1 + v2 * v2 - 2 * v1 * v2 * scalar / distance)) == 0:
        #     return u
        t = distance / (np.sqrt(v1 * v1 + v2 * v2 - 2 * v2 * scalar / distance))
        pointX = pos2[1] + t * self.env.agents[self.advertisal_agent_index].state.p_vel[1]
        pointY = pos2[0] + t * self.env.agents[self.advertisal_agent_index].state.p_vel[0]

        pointXD = pointX - pos1[1]
        pointYD = pointY - pos1[0]
        distanceToPoint = np.sqrt(pointXD * pointXD + pointYD * pointYD)
        xd = pointXD / distanceToPoint
        yd = pointYD / distanceToPoint
        u[2] = xd
        u[0] = yd


        return np.concatenate([u, np.zeros(self.env.world.dim_c)])


class ToPointAgent(Policy):
    def __init__(self, env, agent_index, advertisal_agent_index):
        super(ToPointAgent, self).__init__(agent_index)
        self.advertisal_agent_index = advertisal_agent_index
        self.env = env
        self.agent_index = agent_index

    "action return speed the fastest to get to the point if we don't know velocity"
    def action(self, obs):
        # ignore observation and just act based on keyboard events
        u = np.zeros(4)
        pos1 = self.env.agents[self.agent_index].state.p_pos
        pos2 = self.env.agents[self.advertisal_agent_index].state.p_pos
        xd = pos2[1] - pos1[1]
        yd = pos2[0] - pos1[0]
        k = np.sqrt(xd * xd + yd * yd)
        xd = xd / k
        yd = yd / k
        u[2] = xd
        u[0] = yd

        return np.concatenate([u, np.zeros(self.env.world.dim_c)])


class ToPointForceAgent(Policy):
    def __init__(self, env, agent_index, advertisal_agent_index):
        super(ToPointAgent, self).__init__(agent_index)
        self.advertisal_agent_index = advertisal_agent_index
        self.env = env
        self.agent_index = agent_index

    def action(self, obs):
        # ignore observation and just act based on keyboard events
        u = np.zeros(4)
        pos1 = self.env.agents[self.agent_index].state.p_pos
        pos2 = self.env.agents[self.advertisal_agent_index].state.p_pos
        xd = pos2[1] - pos1[1]
        yd = pos2[0] - pos1[0]
        k = np.sqrt(xd * xd + yd * yd)
        xd = xd / k
        yd = yd / k
        u[2] = xd
        u[0] = yd

        return np.concatenate([u, np.zeros(self.env.world.dim_c)])


class ToPointsAgent(Policy):
    def __init__(self, env, agent_index, advertisal_agent_indexex):
        super(ToPointsAgent, self).__init__(agent_index)
        self.advertisal_agent_indexex = advertisal_agent_indexex
        self.env = env
        self.caught_agents = set()

    def action(self, obs):
        # ignore observation and just act based on keyboard events
        agent = self.env.agents[self.agent_index]
        # check if catch
        for advertisal_agent_index in self.advertisal_agent_indexex:
            if advertisal_agent_index in self.caught_agents:
                continue
            if self.env._is_collision(self.env.agents[self.agent_index], self.env.agents[advertisal_agent_index]):
                self.caught_agents.add(advertisal_agent_index)
        min_distance = 1e10
        closest_agent = self.advertisal_agent_indexex[0]
        for advertisal_agent_index in self.advertisal_agent_indexex:
            if advertisal_agent_index in self.caught_agents:
                continue
            if self.env._is_collision(self.env.agents[self.agent_index], self.env.agents[advertisal_agent_index]):
                self.caught_agents.add(advertisal_agent_index)

            distance = np.sqrt(
                np.sum(np.square(agent.state.p_pos - self.env.agents[advertisal_agent_index].state.p_pos)))
            if distance < min_distance:
                min_distance = distance
                closest_agent = advertisal_agent_index

        u = np.zeros(4)
        pos1 = agent.state.p_pos
        pos2 = self.env.agents[closest_agent].state.p_pos
        xd = pos2[1] - pos1[1]
        yd = pos2[0] - pos1[0]
        k = np.sqrt(xd * xd + yd * yd)
        xd = xd / k
        yd = yd / k
        u[2] = xd
        u[0] = yd

        return np.concatenate([u, np.zeros(self.env.world.dim_c)])

    def reset(self):
        self.caught_agents = set()


class ToPointsAgentWithVelocity(Policy):
    def __init__(self, env, agent_index, advertisal_agent_indexex):
        super(ToPointsAgent, self).__init__(agent_index)
        self.advertisal_agent_indexex = advertisal_agent_indexex
        self.env = env
        self.caught_agents = set()

    def action(self, obs):
        # ignore observation and just act based on keyboard events
        agent = self.env.agents[self.agent_index]
        # check if catch
        for advertisal_agent_index in self.advertisal_agent_indexex:
            if advertisal_agent_index in self.caught_agents:
                continue
            if self.env._is_collision(self.env.agents[self.agent_index], self.env.agents[advertisal_agent_index]):
                self.caught_agents.add(advertisal_agent_index)
        min_distance = 1e10
        closest_agent = self.advertisal_agent_indexex[0]
        for advertisal_agent_index in self.advertisal_agent_indexex:
            if advertisal_agent_index in self.caught_agents:
                continue
            if self.env._is_collision(self.env.agents[self.agent_index], self.env.agents[advertisal_agent_index]):
                self.caught_agents.add(advertisal_agent_index)

            distance = np.sqrt(
                np.sum(np.square(agent.state.p_pos - self.env.agents[advertisal_agent_index].state.p_pos)))
            if distance < min_distance:
                min_distance = distance
                closest_agent = advertisal_agent_index

        u = np.zeros(5)
        pos1 = agent.state.p_pos
        pos2 = self.env.agents[closest_agent].state.p_pos
        xd = pos2[1] - pos1[1]
        yd = pos2[0] - pos1[0]
        k = np.sqrt(xd * xd + yd * yd)
        xd = xd / k
        yd = yd / k
        u[3] = xd
        u[1] = yd

        return np.concatenate([u, np.zeros(self.env.world.dim_c)])
