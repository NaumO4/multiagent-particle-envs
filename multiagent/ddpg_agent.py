import numpy as np
from baselines.ddpg.memory import Memory
from baselines.ddpg.models import Actor, Critic
from baselines.ddpg.ddpg import DDPG
from baselines.ddpg.noise import AdaptiveParamNoiseSpec, NormalActionNoise, OrnsteinUhlenbeckActionNoise
from baselines import logger

# from ddpg_model import Actor
from multiagent.policy import Policy


class DDPGAgent(Policy):
    def __init__(self, env, agent_index, sess, action_range=(-1., 1.), reward_scale=1.0, critic_l2_reg=1e-2,
                 actor_lr=1e-4, critic_lr=1e-3, popart=False, gamma=0.99, clip_norm=None, batch_size=64, memory_size=1e6, tau=0.01,
                 normalize_returns=False, normalize_observations=True,
                 noise_type="adaptive-param_0.2", **network_kwargs):
        super(DDPGAgent, self).__init__(agent_index)
        self.sess = sess
        self.nb_actions = env.action_space[agent_index].n
        print('agent action_space ' + str(env.action_space[agent_index].n))
        self.state_size = env.observation_space[agent_index].shape
        self.action_range = action_range
        critic = Critic(**network_kwargs)
        actor = Actor(self.nb_actions, **network_kwargs)
        memory = Memory(limit=int(memory_size), action_shape=(self.nb_actions,),
                        observation_shape=self.state_size)
        action_noise = None
        param_noise = None
        if noise_type is not None:
            for current_noise_type in noise_type.split(','):
                current_noise_type = current_noise_type.strip()
                if current_noise_type == 'none':
                    pass
                elif 'adaptive-param' in current_noise_type:
                    _, stddev = current_noise_type.split('_')
                    param_noise = AdaptiveParamNoiseSpec(initial_stddev=float(stddev),
                                                         desired_action_stddev=float(stddev))
                elif 'normal' in current_noise_type:
                    _, stddev = current_noise_type.split('_')
                    action_noise = NormalActionNoise(mu=np.zeros(self.nb_actions),
                                                     sigma=float(stddev) * np.ones(self.nb_actions))
                elif 'ou' in current_noise_type:
                    _, stddev = current_noise_type.split('_')
                    action_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(self.nb_actions),
                                                                sigma=float(stddev) * np.ones(self.nb_actions))
                else:
                    raise RuntimeError('unknown noise type "{}"'.format(current_noise_type))

        self.agent = DDPG(actor, critic, memory, self.state_size, (self.nb_actions,), action_range=self.action_range,
                          gamma=gamma, tau=tau, normalize_returns=normalize_returns,
                          normalize_observations=normalize_observations,
                          batch_size=batch_size, action_noise=action_noise, param_noise=param_noise,
                          critic_l2_reg=critic_l2_reg,
                          actor_lr=actor_lr, critic_lr=critic_lr, enable_popart=popart, clip_norm=clip_norm,
                          reward_scale=reward_scale)

        logger.info('Using agent with the following configuration:')
        logger.info(str(self.agent.__dict__.items()))

        self.agent.initialize(self.sess)
        self.agent.reset()

    def action(self, obs, apply_noise=False, compute_Q=False):
        if compute_Q:
            return self.agent.pi(obs, apply_noise=apply_noise, compute_Q=compute_Q)
        else:
            return self.agent.pi(obs, apply_noise=apply_noise, compute_Q=compute_Q)[0]

    def reset(self):
        return self.agent.reset()