import numpy as np


# defines scenario upon which the world is built
class BaseScenario(object):
    # create elements of the world
    def make_world(self, num_good_agents, num_adversaries, is_set_fixed_states):
        raise NotImplementedError()

    # create initial conditions of the world
    def reset_world(self, world):
        raise NotImplementedError()
