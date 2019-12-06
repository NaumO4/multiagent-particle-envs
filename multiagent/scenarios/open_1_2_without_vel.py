import os

from multiagent.scenarios.my_Scenario import MyScenario
import numpy as np
import matplotlib.pyplot as plt
from os.path import join

from multiagent.simple_agents import StayAgent


class Scenario(MyScenario):

    def make_world(self):
        self.REWARD_FOR_COLISION = 505
        name = 'open_1_2_without_vel_REWARD_FOR_COLISION_' + str(self.REWARD_FOR_COLISION)
        world = super(Scenario, self).make_world(name, 2, 1, is_random_states_for_new_agent=False, bounds=False, REWARD_FOR_COLISION = self.REWARD_FOR_COLISION)
        world.step = self.step_without_velocity(world)
        self.caught_agents = set()
        return world

    def step_without_velocity(self, world):
        def step_without_velocity():
            for agent in world.scripted_agents:
                agent.action = agent.action_callback(agent, self)
                # gather forces applied to entities
            for agent in world.agents:
                speed = agent.action.u
                if np.sqrt(np.square(speed[0]) + np.square(speed[1])) != 0:
                    speed = speed / np.sqrt(np.square(speed[0]) + np.square(speed[1])) * agent.max_speed
                else:
                    print(('Speed equal: ' + str(speed[0]) + ' and ' + str(speed[1])))
                agent.state.p_pos += speed * world.dt
                agent.state.p_vel = speed
            for agent in world.agents:
                world.update_agent_state(agent)

            # for good_agent in self.good_agents(world):
            #     for advertisal_agent in self.adversaries(world):
            #         if good_agent in self.caught_agents:
            #             continue
                    # if self.is_collision(advertisal_agent, good_agent):
                    #     self.caught_agents.add(good_agent)
        return step_without_velocity

    def reset_world(self, world):
        # random properties for agents
        self.caught_agents = set()
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

    def agent_reward(self, agent, world):
        # Agents are negatively rewarded if caught by adversaries
        if agent in self.caught_agents:
            return 0
        rew = 0
        shape = True
        adversaries = self.adversaries(world)
        if shape:  # reward can optionally be shaped (increased reward for increased distance from adversary)
            for adv in adversaries:
                rew += 0.1 * np.sqrt(np.sum(np.square(agent.state.p_pos - adv.state.p_pos)))
        if agent.collide:
            for a in adversaries:
                if self.is_collision(a, agent):
                    rew -= self.REWARD_FOR_COLISION
                    break

        # agents are penalized for exiting the screen, so that they can be caught by the adversaries
        def bound(x):
            if x < 0.9:
                return 0
            if x < 1.0:
                return (x - 0.9) * 10
            return min(np.exp(2 * x - 2), 10)

        if self.bounds:
            for p in range(world.dim_p):
                x = abs(agent.state.p_pos[p])
                rew -= bound(x)

        return rew

    def adversary_reward(self, agent, world):
        # Adversaries are rewarded for collisions with agents
        rew = 0
        shape = True
        good_agents = self.good_agents(world)
        agents = []
        for ag in good_agents:
            if ag not in self.caught_agents:
                agents.append(ag)
        if len(agents) == 0:
            return 0
        adversaries = self.adversaries(world)
        if shape:  # reward can optionally be shaped (decreased reward for increased distance from agents)
            for adv in adversaries:
                rew -= 0.1 * min([np.sqrt(np.sum(np.square(a.state.p_pos - adv.state.p_pos))) for a in agents])
        if agent.collide:
            for ag in agents:
                for adv in adversaries:
                    delta_pos = ag.state.p_pos - adv.state.p_pos
                    dist = np.sqrt(np.sum(np.square(delta_pos)))
                    if self.is_collision(ag, adv):
                        rew += self.REWARD_FOR_COLISION
                        break
        return rew

    def set_states_for_good_agent_and_adversary(self, world):
        if len(world.agents) != 3: return
        world.agents[0].state.p_pos = np.array([-0.5, -0.5])
        world.agents[1].state.p_pos = np.array([0.5, 0.5])
        world.agents[2].state.p_pos = np.array([-0.3, 0.5])

        for agent in world.agents:
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)

    def done(self, agent, world):
        return self.adversary_done(agent, world) if agent.adversary else self.good_done(agent, world)

    def good_done(self, agent, world):
        if agent in self.caught_agents:
            return True
        for agent2 in world.agents:
            if agent2.adversary:
                if self.is_collision(agent, agent2):
                    self.caught_agents.add(agent)
                    return True
        return False

    def adversary_done(self, agent, world):
        return len(self.good_agents(world)) == len(self.caught_agents)

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
        #normalize
        other_pos = np.array(other_pos)
        # if np.sqrt(np.sum(np.square(other_pos))) != 0:
        #     other_pos = other_pos / np.sqrt(np.sum(np.square(other_pos)))
        #other_pos = other_pos.tolist()
        is_caught = []
        for good_agent in self.good_agents(world):
            if good_agent in self.caught_agents:
                is_caught.append([1.])
            else:
                is_caught.append([0.])
        entity_pos = np.array(entity_pos).flatten()
        other_pos = other_pos.flatten()
        # if np.sqrt(np.sum(np.square(other_pos))) != 0:
        #     other_pos = other_pos / np.sqrt(np.sum(np.square(other_pos)))
        is_caught = np.array(is_caught).flatten()
        return np.concatenate([other_pos, is_caught])