import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn

from agent.agent import Transition
from algorithm.actor_critic_policy import ActorCriticAlgorithm
from torch import autograd
from torch_models.reward_vector_estimator import RewardVectorEstimator
from torch_models.scalar_reward_estimator import ScalarRewardEstimator
from torch_models.advantage_actor_critic import ActorCritic


def run_a2c_gtp(env,
                env_code,
                agent,
                gradient_threshold=None,
                max_time=1000,
                verbose=True):

    """
    Receives a continuing gym environment and runs the online A2C-GTP algorithm on it

    :param env: access to gym environment
    :param agent: access to the Agent object which stores its objective function and the simulation historyg
    :param gradient_threshold: gradient threshold limit
    :param uncertain_model: if the algorithm has access to the probability kernel or rewards
    :param max_time: maximum no. of simulation time steps
    :param verbose: if print statements should be printed for debugging
    :return: to be determined
    """

    def get_one_hot(coordinates):
        new_cood = tuple(coordinates)
        one_hot = {
            (0.0, 0.0): [1, 0, 0, 0, 0, 0, 0, 0, 0],
            (1.0, 0.0): [0, 1, 0, 0, 0, 0, 0, 0, 0],
            (2.0, 0.0): [0, 0, 1, 0, 0, 0, 0, 0, 0],
            (0.0, 1.0): [0, 0, 0, 1, 0, 0, 0, 0, 0],
            (1.0, 1.0): [0, 0, 0, 0, 1, 0, 0, 0, 0],
            (2.0, 1.0): [0, 0, 0, 0, 0, 1, 0, 0, 0],
            (0.0, 2.0): [0, 0, 0, 0, 0, 0, 1, 0, 0],
            (1.0, 2.0): [0, 0, 0, 0, 0, 0, 0, 1, 0],
            (2.0, 2.0): [0, 0, 0, 0, 0, 0, 0, 0, 1],
        }
        return one_hot[new_cood]

    """
    Retrieve essential parameters from the environment
    """
    num_dim_features = 9 #env.observation_space.shape[0]
    num_dim_rewards = env.unwrapped.reward_dim  # compatibility will break with other envs (must define variable)
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

    scalar_reward_estimator = ScalarRewardEstimator(num_dim_features, num_dim_rewards, hidden_size=30).to(device)

    policy_generator = ActorCriticAlgorithm(env_code, num_dim_features, device)

    """
    Initialise first step
    """
    current_time = 0
    current_epoch = -1  # to be incremented upon entering the loop below
    current_features = get_one_hot(env.reset())

    agent.feature_list.append(current_features)  # add first set of features

    current_features = torch.tensor(current_features).float()

    """
    Actual algorithm 
    """
    while current_time < max_time:

        # Increment epoch counter
        current_epoch += 1

        if verbose:
            print("Epoch {} has begun at time step {}".format(current_epoch, current_time))

        # Calculate gradient of objective function and store it as ref_grad
        ab, ref_grad = agent.get_objectives()

        if current_time > 0:
            print(agent.avg_reward_list[-1])

        """
        # Reinitialise and train the scalar_reward estimator with updated ref_grad  # TODO: Dump this into another module
        scalar_reward_estimator.train()
        scalar_reward_estimator.reset_model()
        scalar_reward_estimator_optimizer = optim.Adam(scalar_reward_estimator.parameters(), lr=0.01)

        if current_time > 0:
            transition_history = Transition(*zip(*agent.transition_list))
            features = torch.cat(transition_history.features).view(-1, num_dim_features)
            actions = torch.cat(transition_history.action)
            rewards = torch.cat(transition_history.reward)

            # Tunes the scalar reward estimator
            for i in range(len(agent.transition_list)):
                scalar_reward_estimate = scalar_reward_estimator(features[i]).gather(0, actions[i])
                scalar_reward_actual = np.dot(rewards[i].clone().detach().numpy(), ref_grad)
                #print(scalar_reward_actual)
                loss = (scalar_reward_estimate - torch.tensor(scalar_reward_actual).float().unsqueeze(0).detach()).pow(10)
                scalar_reward_estimator_optimizer.zero_grad()
                loss.backward()
                scalar_reward_estimator_optimizer.step()
        """

        # Initialises the scalar reward dictionary  # TODO: Put this chunk somewhere more pretty
        scalar_reward_estimator.eval()
        est_reward_dict = {}
        real_reward_dict = {}
        for i in range(3):
            for j in range(3):  # TODO: Remove this line?
                i = i * 1.0
                j = j * 1.0
                est_reward_dict[tuple(get_one_hot((i, j)))] = scalar_reward_estimator(torch.tensor(get_one_hot(np.array([i, j]))).float()).detach()
                reward_list = []
                for action in range(4):
                    outcome = np.dot(env.unwrapped.v_mean[((i, j), action)], ref_grad)
                    reward_list.append(outcome)
                reward_list = np.array(reward_list)
                real_reward_dict[(i, j)] = torch.tensor(reward_list).float().detach()

        print(ref_grad)
        # Use the policy generator to generate a policy given the current ref_grad
        policy = policy_generator.generate_policy(real_reward_dict)

        # Reinitialise sum of grad_objective psi
        psi = 0

        current_epoch_start = current_time
        while psi <= gradient_threshold and current_time < max_time:

            # Take one step forward
            policy_dist, _ = policy(current_features)

            #print(policy_dist)
            m = nn.Softmax(dim=0)
            policy_dist = m(policy_dist)
            current_action = np.random.choice(num_actions, p=policy_dist.clone().detach().numpy())

            next_features, _, _, _ = env.step(current_action)  # TODO: Clean up the step function somehow
            next_features = get_one_hot(next_features)
            reward_vector = env.algo_step(current_action)

            if current_time >= (max_time - 6000):
                env.render()

            # Store new transition in history
            agent.update_history(next_features, current_action, reward_vector)

            next_features = torch.tensor(next_features).float()

            #agent.add_transition(current_features, torch.tensor([[current_action]]),
            #                     next_features, torch.tensor(reward_vector).float().unsqueeze(0))

            # Update gradient counter
            _, current_grad = agent.get_objectives()
            psi += np.linalg.norm(current_grad - ref_grad)

            # Update current step
            current_features = next_features
            current_time += 1








