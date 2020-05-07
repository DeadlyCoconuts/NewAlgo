import numpy as np
import torch
import torch.optim as optim

from agent.agent import Transition
from torch_models.reward_vector_estimator import RewardVectorEstimator
from torch_models.scalar_reward_estimator import ScalarRewardEstimator
from torch_models.advantage_actor_critic import ActorCritic


def run_A2C_GTP(env,
                agent,
                gamma=1,
                gradient_threshold=None,
                max_time=10000,
                verbose=True):

    """
    Receives a continuing gym environment and runs the online A2C-GTP algorithm on it

    :param env: access to gym environment
    :param agent: access to the Agent object which stores its objective function and the simulation history
    :param gamma: discount factor for rewards
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

    #reward_vector_estimator = RewardVectorEstimator(num_dim_features, num_dim_rewards, hidden_size=10).to(device)
    #reward_vector_estimator_optimizer = optim.Adam(reward_vector_estimator.parameters())

    scalar_reward_estimator = ScalarRewardEstimator(num_dim_features, num_dim_rewards, hidden_size=10).to(device)

    actor_critic = ActorCritic(num_dim_features, num_actions, hidden_size=10).to(device)

    """
    Initialise first step
    """
    current_time = 0
    current_epoch = -1  # to be incremented upon entering the loop below
    current_features = env.reset()
    agent.feature_list.append(current_features)  # add first set of features

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

        # Reinitialise and train the scalar_reward estimator with updated ref_grad
        scalar_reward_estimator.reset_model()
        scalar_reward_estimator_optimizer = optim.Adam(scalar_reward_estimator.parameters())

        batch = Transition(*zip(*agent.transition_list))
        features_batch = torch.cat(batch.features).view(-1, num_dim_rewards)
        action_batch = torch.cat(batch.action)

        for i in range(agent.max_memory_capacity - 1):
            scalar_reward_estimate = scalar_reward_estimator(features_batch[i].gather(0, action_batch[i]))
            scalar_reward_actual = np.dot(features_batch[i].clone().detach().numpy(), ref_grad)
            loss = (scalar_reward_estimate - torch.tensor(scalar_reward_actual).float().unsqueeze(0).detach()).pow(2)
            scalar_reward_estimator_optimizer.zero_grad()
            loss.backward()
            scalar_reward_estimator_optimizer.step()

        scalar_reward_estimator.eval()

        # Reinitialise the actor-critic (policy & value function) approximators
        actor_critic.reset_model()
        actor_critic_optimizer = optim.Adam(actor_critic.parameters())

        # Reinitialise sum of grad_objective psi
        psi = 0

        while psi <= gradient_threshold and current_time <= max_time:
            # Take one step forward
            policy_dist, value = actor_critic(current_features)

            current_action = np.random.choice(num_actions, p=policy_dist.clone().detach().numpy())
            next_features, reward_vector, done, _ = env.step(current_action)  # TODO: Clean up

            # Store new transition in history  # TODO: Add other history storing steps
            agent.update_history(next_features, current_action, reward_vector)
            agent.add_transition(torch.tensor(current_features).float(), torch.tensor([[current_action]]),
                                 torch.tensor(next_features).float(), torch.tensor(reward_vector).float().unsqueeze())

            # Update actor critic parameters
            log_prob = torch.log(policy_dist[current_action])
            with torch.no_grad():
                _, new_value = actor_critic(next_features)
                advantage = reward_vector + gamma * new_value - value
            actor_loss = (-log_prob * advantage)
            critic_loss = advantage.pow(2)
            actor_critic_loss = actor_critic + critic_loss  # TODO: Add entropy term?

            actor_critic_optimizer.zero_grad()
            actor_critic_loss.backward()
            actor_critic_optimizer.step()

            # Update gradient counter
            _, current_grad = agent.get_objectives()
            psi += np.linalg.norm(current_grad - ref_grad)

            # Update current step
            current_features = next_features
            current_time += 1









