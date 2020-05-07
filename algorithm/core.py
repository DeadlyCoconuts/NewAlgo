import numpy as np
import torch
import torch.optim as optim
from torch_models.reward_vector_estimator import RewardVectorEstimator
from torch_models.scalar_reward_estimator import ScalarRewardEstimator
from torch_models.advantage_actor_critic import ActorCritic


def run_A2C_GTP(env,
                agent,
                gradient_threshold=None,
                max_time=10,
                verbose=True):

    """
    Receives a continuing gym environment and runs the online A2C-GTP algorithm on it

    :param env: access to gym environment
    :param agent: access to the Agent object which stores its objective function and the simulation history
    :param gradient_threshold: gradient threshold limit
    :param uncertain_model: if the algorithm has access to the probability kernel or rewards
    :param max_time: maximum no. of simulation time steps
    :param verbose: if print statements should be printed for debugging
    :return: to be determined
    """

    """
    Retrieve essential parameters from the environment
    """
    num_dim_features = env.observation_space.shape[0]
    #num_dim_rewards = env.reward_range[0].size  # TODO: Find a more robust way to get gym environments to return this value
    num_dim_rewards = 3  # arbitrary
    num_actions = env.action_space.n

    """
    Ensure gradient_threshold is well defined
    """
    if gradient_threshold is None:
        gradient_threshold = 1 / np.sqrt(agent.objective_function.num_dim)

    """
    Initialise parameters for the various parameterised variables
    """
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # Train estimator again
    reward_vector_estimator = RewardVectorEstimator(num_dim_features, num_dim_rewards, hidden_size=10).to(device)
    #reward_vector_estimator_optimizer = optim.Adam(reward_vector_estimator.parameters())

    scalar_reward_estimator = ScalarRewardEstimator(num_dim_features, num_dim_rewards, hidden_size=10).to(device)
    scalar_reward_estimator_optimizer = optim.Adam(scalar_reward_estimator.parameters())

    actor_critic = ActorCritic(num_dim_features, num_actions, hidden_size=10).to(device)

    """
    Initialise first step
    """
    current_time = 0
    current_epoch = -1  # to be incremented upon entering the loop below
    current_features = env.reset()

    """
    Actual algorithm 
    """
    while current_time < max_time:

        # Increment epoch counter
        current_epoch += 1

        if verbose:
            print("Epoch {} has begun at time step {}".format(current_epoch, current_time))

        # Calculate gradient of objective function and store it as ref_grad
        _, ref_grad = agent.get_objectives()

        # Reinitialise the scalar reward approximator
        reward_vector_estimator_params = reward_vector_estimator.state_dict()

        scalar_reward_estimator.add_grad_objective_weights(ref_grad, reward_vector_estimator_params)
        scalar_reward_estimator.eval()  # evaluation mode; weights do not changes

        # Reinitialise the actor-critic (policy & value function) approximators
        actor_critic.reset_model()

        # Reinitialise sum of grad_objective psi
        psi = 0

        while psi <= gradient_threshold and current_time <= max_time:
            pass
            # Take one step forward
            policy_dist, value = actor_critic(current_features)
            policy_dist = policy_dist.detach().numpy()

            current_action = np.random.choice(num_actions, p=policy_dist)
            new_features, reward_vector, done, _ = env.step(current_action)  # TODO: Clean up

            # Update parameters of reward vector estimator
            reward_vector_estimate = reward_vector_estimator(reward_vector)
            reward_vector_estimator_loss = (reward_vector_estimate - reward_vector).pow(2).sum()
            reward_vector_estimator_optimizer.zero_grad()
            reward_vector_estimator_loss.backward()
            reward_vector_estimator_optimizer.step()






