import numpy as np
from agent.agent import History
from torch_models.reward_vector_estimator import RewardVectorEstimator
from torch_models.a2c import ActorCritic


def run_A2C_GTP(env,
            agent,
            gradient_threshold=None,
            uncertain_model=False,
            max_steps=100000,
            verbose=True):

    """
    Receives a continuing gym environment and runs the online A2C-GTP algorithm on it

    :param env: access to gym environment
    :param agent: access to the Agent object which stores its objective function and the simulation history
    :param gradient_threshold: gradient threshold limit
    :param uncertain_model: if the algorithm has access to the probability kernel or rewards
    :param max_steps: maximum no. of simulation time steps
    :param verbose: if print statements should be printed for debugging
    :return: to be determined
    """

    """
    Retrieve essential parameters from the environment
    """
    num_dim_features = env.observation_space.shape[0]
    #num_dim_rewards = env.reward_range[0].size  # TODO: Find a more robust way to get gym environments to return this value
    num_dim_rewards = 3  # arbitrary

    """
    Ensure gradient_threshold is well defined
    """
    if gradient_threshold is None:
        gradient_threshold = 1 / np.sqrt(agent.objective_function.num_dim)

    """
    Initialise parameters for reward vector estimator
    """
    reward_vector_estimator = RewardVectorEstimator(num_dim_features, num_dim_rewards, hidden_size=10)
    scalarised_reward_estimator = RewardVectorEstimator(num_dim_features, num_dim_rewards, hidden_size=10)
    actor_critic = ActorCritic(num_dim_features, num_dim_rewards, hidden_size=10)

    """
    Initialise first step
    """
    current_time = 0
    current_epoch = -1  # to be incremented upon entering the loop below
    current_features = env.reset()

    """
    Actual algorithm 
    """
    while current_time < max_steps:

        # Increment epoch counter
        current_epoch += 1

        if verbose:
            print("Epoch {} has begun at time step {}".format(current_epoch, current_time))

