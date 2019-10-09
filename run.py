from multiagent import ddpg_train
from multiagent.scenarios.open_1_1_without_vel import Scenario

if __name__ == '__main__':
    scenario = Scenario()
    ddpg_train.train(scenario)