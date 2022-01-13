from abc import ABC, abstractmethod
from collections import defaultdict
import random
from typing import List, Dict, DefaultDict
from gym.spaces import Space
from gym.spaces.utils import flatdim


class Agent(ABC):
    """Base class for RL agent

    **ONLY CHANGE THE BODY OF THE act() FUNCTION**

    """

    def __init__(
        self,
        num_actions: int,
        gamma: float,
        epsilon: float,
        **kwargs
    ):
        """Constructor of base agent for Q-Learning

        Initializes basic variables of the RL agent
        namely the epsilon, learning rate and discount rate.

        :param num_actions (int): number of actions available to agent
        :param gamma (float): discount factor (gamma)
        :param epsilon (float): epsilon for epsilon-greedy action selection

        :attr n_acts (int): number of actions
        :attr q_table (DefaultDict): table for Q-values mapping (OBS, ACT) pairs of observations
            and actions to respective Q-values
        """
        self.n_acts: int = num_actions
        self.epsilon: float = epsilon
        self.gamma: float = gamma
        self.q_table: DefaultDict = defaultdict(lambda: 0)

    def act(self, obs: int) -> int:
        """Implement the epsilon-greedy action selection here

        **YOU MUST IMPLEMENT THIS FUNCTION FOR SARSA**

        :param obs (int): received observation representing the current environmental state
        :return (int): index of selected action
        """
        ### SOLUTION BELOW ###
        act_vals = [self.q_table[(obs, act)] for act in range(self.n_acts)]
        max_val = max(act_vals)
        max_acts = [idx for idx, act_val in enumerate(act_vals) if act_val == max_val]

        if random.random() < self.epsilon:
            return random.randint(0, self.n_acts - 1)
        else:
            return random.choice(max_acts)

    @abstractmethod
    def schedule_hyperparameters(self, timestep: int, max_timestep: int):
        """Updates the hyperparameters

        This function is called before every episode and allows you to schedule your
        hyperparameters.

        :param timestep (int): current timestep at the beginning of the episode
        :param max_timestep (int): maximum timesteps that the training loop will run for
        """
        ...

    @abstractmethod
    def learn(self):
        ...


class SARSA(Agent):
    """Class for SARSA agent

    :attr n_acts (int): number of actions
    :attr gamma (float): discount factor gamma
    :attr epsilon (float): epsilon hyperparameter for epsilon-greedy policy
    :attr alpha (float): learning rate alpha for updates
    :attr q_table (DefaultDict): table for Q-values mapping (OBS, ACT) pairs of observations
        and actions to respective Q-values
    """

    def __init__(self, alpha: float, **kwargs):
        """Constructor for SARSA agent

        Initializes basic variables of the agent namely the epsilon, learning rate and discount

        :param alpha (float): learning rate alpha for SARSA updates
        """

        super().__init__(**kwargs)
        self.alpha: float = alpha

    def learn(
            self,
            obs: int,
            action: int,
            reward: float,
            n_obs: int,
            n_action: int,
            done: bool
        ) -> float:
        """Updates the SARSA Q-table based on agent experience

        **YOU MUST IMPLEMENT THIS FUNCTION FOR SARSA**

        :param obs (int): received observation representing the current environmental state
        :param action (int): index of applied action
        :param reward (float): received reward
        :param n_obs (int): received observation representing the next environmental state
        :param n_action (int): index of applied action in next state
        :param done (bool): flag indicating whether a terminal state has been reached
        :return (float): updated Q-value for current observation-action pair
        """
        ### SOLUTION BELOW ###
        target_value = reward + self.gamma * (1 - done) * self.q_table[(n_obs, n_action)]
        self.q_table[(obs, action)] += self.alpha * (
            target_value - self.q_table[(obs, action)]
        )
        return self.q_table[(obs, action)]


    def schedule_hyperparameters(self, timestep: int, max_timestep: int):
        """Updates the hyperparameters

        This function is called before every episode and allows you to schedule your
        hyperparameters.

        :param timestep (int): current timestep at the beginning of the episode
        :param max_timestep (int): maximum timesteps that the training loop will run for
        """
        self.epsilon = 1.0-(min(1.0, timestep/(0.07*max_timestep)))*0.95
