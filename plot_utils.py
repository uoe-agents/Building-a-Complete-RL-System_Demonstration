import matplotlib.pyplot as plt
import numpy as np

plt.style.use("seaborn-darkgrid")
plt.rcParams.update({"font.size": 15})


def plot_timesteps(eval_freq: int, values: np.ndarray, stds: np.ndarray, xlabel: str, ylabel: str, legend_name: str):
    """
    Plot values with respect to timesteps
    
    :param eval_freq (int): evaluation frequency
    :param values (np.ndarray): numpy array of values to plot as y values
    :param std (np.ndarray): numpy array of stds of y values to plot as shading
    :param xlabel (str): label of x-axis
    :param ylabel (str): label of y-axis
    :param legend_name (str): name of algorithm
    """
    x_values = eval_freq + np.arange(len(values)) * eval_freq
    plt.plot(x_values, values, "-", alpha=0.7, label=f"{legend_name}")
    plt.fill_between(
        x_values,
        values - stds,
        values + stds,
        alpha=0.2,
        antialiased=True,
    )
    plt.legend(loc="best")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout(pad=0.3)
