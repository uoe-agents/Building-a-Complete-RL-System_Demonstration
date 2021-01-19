from abc import ABC
from collections import defaultdict
import numpy as np
import random
from typing import DefaultDict


class SARSA(ABC):
    """Base class for SARSA agent

    :attr n_acts (int): number of actions
    :attr gamma (float): discount factor gamma
    :attr epsilon (float): epsilon hyperparameter for epsilon-greedy policy
    :attr alpha (float): learning rate alpha for updates
    :attr q_table (DefaultDict): table for Q-values mapping (OBS, ACT) pairs of observations
        and actions to respective Q-values
    """

    def __init__(
            self,
            num_acts: int,
            gamma: float,
            epsilon: float = 0.9,
            alpha: float = 0.1
        ):
        """Constructor for SARSA agent

        Initializes basic variables of the agent namely the epsilon, learning rate and discount
        rate.

        :param num_acts (int): number of possible actions
        :param gamma (float): discount factor (gamma)
        :param epsilon (float): initial epsilon for epsilon-greedy action selection
        :param alpha (float): learning rate alpha
        """
        self.n_acts: int = num_acts
        self.gamma: float = gamma
        self.epsilon: float = epsilon
        self.alpha: float = alpha

        self.q_table: DefaultDict = defaultdict(lambda: 0)

    def act(self, obs: np.ndarray) -> int:
        """Epsilon-greedy action selection 

        :param obs (np.ndarray of float with dim (observation size)):
            received observation representing the current environmental state
        :return (int): index of selected action
        """
        act_vals = [self.q_table[(obs, act)] for act in range(self.n_acts)]
        max_val = max(act_vals)
        max_acts = [idx for idx, act_val in enumerate(act_vals) if act_val == max_val]

        if random.random() < self.epsilon:
            return random.randint(0, self.n_acts - 1)
        else:
            return random.choice(max_acts)

    def learn(
            self,
            obs: np.ndarray,
            action: int,
            reward: float,
            n_obs: np.ndarray,
            n_action: int,
            done: bool
        ) -> float:
        """Updates the Q-table based on agent experience

        :param obs (np.ndarray of float with dim (observation size)):
            received observation representing the current environmental state
        :param action (int): index of applied action
        :param reward (float): received reward
        :param n_obs (np.ndarray of float with dim (observation size)):
            received observation representing the next environmental state
        :param done (bool): flag indicating whether a terminal state has been reached
        :return (float): updated Q-value for current observation-action pair
        """
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
