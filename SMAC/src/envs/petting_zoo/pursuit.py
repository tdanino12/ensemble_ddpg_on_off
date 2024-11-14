from envs.multiagentenv import MultiAgentEnv
from utils.dict2namedtuple import convert
import numpy as np
from gym.spaces import Box
from pettingzoo.sisl import pursuit_v4

class Pursuit(MultiAgentEnv):
    def __init__(self, batch_size=None, **kwargs):

        # Define the agents and actions
        self.n_agents = 8
        self.n_actions = 5
        self.episode_limit = 1
        self.env = pursuit_v4.parallel_env(
                                      max_cycles=500,
                                      x_size=16,
                                      y_size=16,
                                      shared_reward=True,
                                      n_evaders=30,
                                      n_pursuers=8,
                                      obs_range=7,
                                      n_catch=2,
                                      freeze_evaders=False,
                                      tag_reward=0.01,
                                      catch_reward=5.0,
                                      urgency_reward=-0.1,
                                      surround=True,
                                      constraint_window=1.0,
        )
        self.state = self.env.observation_spaces['pursuer_0']
        self.action_space = self.env.action_spaces['pursuer_0']
    def reset(self):
        self.observations, infos = self.env.reset()
        return self.observations, self.state

    def step(self, actions):
        self.observations, rewards, terminations, truncations, infos = self.env.step(actions)
        return rewards, terminations, infos

    def get_obs(self):
        return [self.state for _ in range(self.n_agents)]

    def get_obs_agent(self, agent_id):
        return self.observations['pursuer_{}'.format(agent_id)]

    def get_obs_size(self):
        """ Returns the shape of the observation """
        return len(self.observations['pursuer_{}'.format(agent_id)])

    def get_state(self):
        return self.state

    def get_state_size(self):
        return len(self.state)

    def get_avail_actions(self):
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_agent = self.get_avail_agent_actions(agent_id)
            avail_actions.append(avail_agent)
        return avail_actions

    def get_avail_agent_actions(self, agent_id):
        """ Returns the available actions for agent_id """
        return np.ones(self.n_actions)

    def get_total_actions(self):
        """ Returns the total number of actions an agent could ever take """
        return self.n_actions

    def get_stats(self):
        raise NotImplementedError

    def get_env_info(self):
        env_info = {"state_shape": self.get_state_size(),
                    "obs_shape": self.get_obs_size(),
                    "n_actions": 5,
                    "n_agents": 8,
                    "episode_limit": self.episode_limit,
                    "action_spaces": self.action_space,
                    "actions_dtype": np.float32,
                    "normalise_actions": False
                    }
        return env_info
