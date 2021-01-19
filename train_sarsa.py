import gym
import matplotlib.pyplot as plt
import numpy as np
import random
from time import sleep

from sarsa import SARSA
from utils import visualise_q_table
from plot_utils import plot_timesteps

CONFIG = {
    "total_eps": 10000,
    "max_episode_steps": 100,
    "eval_episodes": 100,
    "eval_freq": 500,
    "gamma": 0.99,
    "alpha": 1e-1,
    "epsilon": 0.9,
}

RENDER = False
SEEDS = 10


def evaluate(env, config, q_table, render=False):
    """
    Evaluate configuration of SARSA on given environment initialised with given Q-table

    :param env (gym.Env): environment to execute evaluation on
    :param config (Dict[str, float]): configuration dictionary containing hyperparameters
    :param q_table (Dict[(Obs, Act), float]): Q-table mapping observation-action to Q-values
    :param render (bool): flag whether evaluation runs should be rendered
    :return (float, float, int): mean and standard deviation of return received over episodes, number
        of negative returns
    """
    eval_agent = SARSA(
            num_acts=env.action_space.n,
            gamma=config["gamma"],
            epsilon=0.0, 
            alpha=config["alpha"],
    )
    eval_agent.q_table = q_table
    episodic_returns = []
    for eps_num in range(config["eval_episodes"]):
        obs = env.reset()
        if render:
            env.render()
            sleep(1)
        episodic_return = 0
        done = False

        steps = 0
        while not done and steps <= config["max_episode_steps"]:
            steps += 1
            act = eval_agent.act(obs)
            n_obs, reward, done, info = env.step(act)
            if render:
                env.render()
                sleep(1)

            episodic_return += reward

            obs = n_obs

        episodic_returns.append(episodic_return)

    mean_return = np.mean(episodic_returns)
    std_return = np.std(episodic_returns)
    negative_returns = sum([ret < 0 for ret in episodic_returns])
    return mean_return, std_return, negative_returns


def train(env, config, output=True):
    """
    Train and evaluate SARSA on given environment with provided hyperparameters

    :param env (gym.Env): environment to execute evaluation on
    :param config (Dict[str, float]): configuration dictionary containing hyperparameters
    :param output (bool): flag whether mean evaluation performance should be printed
    :return (List[float], List[float], List[float], Dict[(Obs, Act)]):
        list of means and standard deviations of evaluation returns, list of epislons, final Q-table
    """
    agent = SARSA(
            num_acts=env.action_space.n,
            gamma=config["gamma"],
            epsilon=config["epsilon"],
            alpha=config["alpha"],
    )

    step_counter = 0
    # 100 as estimate of max steps to take in an episode
    max_steps = config["total_eps"] * config["max_episode_steps"]
    
    evaluation_return_means = []
    evaluation_return_stds = []
    evaluation_epsilons = []

    for eps_num in range(config["total_eps"]):
        obs = env.reset()
        episodic_return = 0
        steps = 0
        done = False

        # take first action
        act = agent.act(obs)

        while not done and steps < config["max_episode_steps"]:
            n_obs, reward, done, info = env.step(act)
            step_counter += 1
            episodic_return += reward

            agent.schedule_hyperparameters(step_counter, max_steps)
            n_act = agent.act(n_obs)
            agent.learn(obs, act, reward, n_obs, n_act, done)

            obs = n_obs
            act = n_act

        if eps_num % config["eval_freq"] == 0:
            mean_return, std_return, negative_returns = evaluate(
                    env,
                    config,
                    agent.q_table,
                    render=RENDER,
            )
            if output:
                print(f"EVALUATION: EP {eps_num} - MEAN RETURN {mean_return} +/- {std_return} ({negative_returns}/{config['eval_episodes']} failed episodes)")
            evaluation_return_means.append(mean_return)
            evaluation_return_stds.append(std_return)
            evaluation_epsilons.append(agent.epsilon)

    return evaluation_return_means, evaluation_return_stds, evaluation_epsilons, agent.q_table


if __name__ == "__main__":
    # execute training and evaluation to generate return plots
    plt.figure(figsize=(8, 8))
    axes = plt.gca()
    plt.title(f"Average Returns on Taxi-v3")

    # draw goal line
    x_min = 0
    x_max = CONFIG["total_eps"]
    plt.hlines(y=8, xmin=x_min, xmax=x_max, colors='k', linestyles='dotted', label="Taxi threshold = 8")
    axes.set_ylim([-100,20])

    env = gym.make("Taxi-v3")

    num_returns = CONFIG["total_eps"] // CONFIG["eval_freq"]

    eval_returns = np.zeros((SEEDS, num_returns))
    for i in range(SEEDS):
        print(f"Executing training for SARSA - run {i + 1}")
        env.seed(i * 100)
        returns, _, epsilons, q_table = train(env, CONFIG, output=False)
        returns = np.array(returns)
        eval_returns[i, :] = returns
    returns_total = eval_returns.mean(axis=0)
    returns_std = eval_returns.std(axis=0)
    plot_timesteps(CONFIG["eval_freq"], returns_total, returns_std, "Episodes", "Mean Eval Returns", "Sarsa")

    plt.show()
