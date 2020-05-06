import numpy as np
from util.agent import History


def run_A2C_GTP(env,
            objective,
            gradient_threshold=None,
            uncertain_model=False,
            max_steps=100000,
            verbose=True):

    """
    Receives a continuing gym environment and runs the online A2C-GTP algorithm on it

    :param env: access to gym environment
    :param reward_type: determines objective function g
    :param gradient_threshold: gradient threshold limit
    :param uncertain_model: if the algorithm has access to the probability kernel or rewards
    :param max_steps: maximum no. of simulation time steps
    :param verbose: if print statements should be printed for debugging
    :return: to be determined
    """

    """
    Initialise history object to store simulation history
    """
    sim_history = History()

    """
    Ensure gradient_threshold is well defined
    """
    if gradient_threshold is None:
        gradient_threshold = 1 / np.sqrt(objective.num_dim)

    """
    Initialise parameters for reward vector estimator
    """
