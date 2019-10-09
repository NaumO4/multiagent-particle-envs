from multiagent.scenarios.my_Scenario import MyScenario
import numpy as np

from multiagent.simple_agents import StayAgent


class Scenario(MyScenario):

    def make_world(self):
        name = 'open_1_1_without_vel'
        world = super(Scenario, self).make_world(name, 1, 1, is_random_states_for_new_agent=False, bounds=False)
        world.step = self.step_without_velocity(world)
        return world

    def step_without_velocity(self, world):
        def step_without_velocity():
            if self.evaluate:
                world.agents[0].u = np.array([0, 1.])
            for agent in world.scripted_agents:
                agent.action = agent.action_callback(agent, self)
                # gather forces applied to entities
            for agent in world.agents:
                speed = agent.action.u
                speed = speed / np.sqrt(np.square(speed[0]) + np.square(speed[1])) * agent.max_speed
                agent.state.p_pos += speed * world.dt
            for agent in world.agents:
                world.update_agent_state(agent)

        return step_without_velocity

    def reset_world(self, world):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.85, 0.35]) if not agent.adversary else np.array([0.85, 0.35, 0.35])
            # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])
        # set random initial states
        if self.is_random_states_for_new_agent and not self.evaluate:
            for agent in world.agents:
                agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
                agent.state.p_vel = np.zeros(world.dim_p)
                agent.state.c = np.zeros(world.dim_c)
        else:
            self.set_states_for_good_agent_and_adversary(world)
        for i, landmark in enumerate(world.landmarks):
            if not landmark.boundary:
                landmark.state.p_pos = np.random.uniform(-0.9, +0.9, world.dim_p)
                landmark.state.p_vel = np.zeros(world.dim_p)

    def set_states_for_good_agent_and_adversary(self, world):
        if len(world.agents) != 2: return
        world.agents[0].state.p_pos = np.array([-0.5, -0.5])
        world.agents[1].state.p_pos = np.array([0.5, 0.5])
        for agent in world.agents:
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:
            if not entity.boundary:
                entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # communication of all other agents
        comm = []
        other_pos = []
        other_vel = []
        for other in world.agents:
            if other is agent: continue
            comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)
            if not other.adversary:
                other_vel.append(other.state.p_vel)
        return np.concatenate([agent.state.p_vel] + entity_pos + other_pos)
