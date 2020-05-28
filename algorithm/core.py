import numpy as np
import torch
import torch.optim as optim

from agent.agent import Transition
from torch_models.reward_vector_estimator import RewardVectorEstimator
from torch_models.scalar_reward_estimator import ScalarRewardEstimator
from torch_models.advantage_actor_critic import ActorCritic


def run_a2c_gtp(env,
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
    num_dim_rewards = env.unwrapped.reward_dim  # compatibility will break with other envs (must define variable)
    num_actions = env.action_space.n

    """
    Ensure gradient_threshold is well defined
    """
    if gradient_threshold is None:
        gradient_threshold = 1 / np.sqrt(agent.objective_function.num_dim)

    """
    Initialise features-action counter
    """
    features_action_counter_dict = dict()

    """
    Initialise parameters for the various parameterised variables
    """
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    #reward_vector_estimator = RewardVectorEstimator(num_dim_features, num_dim_rewards, hidden_size=10).to(device)
    #reward_vector_estimator_optimizer = optim.Adam(reward_vector_estimator.parameters())

    scalar_reward_estimator = ScalarRewardEstimator(num_dim_features, num_dim_rewards, hidden_size=30).to(device)

    actor_critic = ActorCritic(num_dim_features, num_actions, hidden_size=10).to(device)

    """
    Initialise first step
    """
    current_time = 0
    current_epoch = -1  # to be incremented upon entering the loop below
    current_features = env.reset()

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
        _, ref_grad = agent.get_objectives()

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

        # Initialises the scalar reward dictionary  # TODO: Put this chunk somewhere more pretty
        scalar_reward_estimator.eval()
        est_reward_dict = {}
        real_reward_dict = {}
        for i in range(3):
            for j in range(3):  # TODO: Remove this line?
                est_reward_dict[(i, j)] = scalar_reward_estimator(torch.tensor(np.array([i, j])).float()).detach()
                reward_list = []
                for action in range(4):
                    outcome = np.dot(env.unwrapped.v_mean[((i, j), action)], ref_grad)
                    reward_list.append(outcome)
                reward_list = np.array(reward_list)
                real_reward_dict[(i, j)] = torch.tensor(reward_list).float().detach()
        #print(real_reward_dict)

        # Reinitialise the actor-critic (policy & value function) approximators
        actor_critic.reset_model()
        actor_critic_optimizer = optim.Adam(actor_critic.parameters())

        # Trains the actor critic model
        batch_size = 2
        if current_time > 0:
            print("hiya")
            print(ref_grad)
            for i in range(0, len(agent.transition_list), batch_size):
                print(i)
                features_batch = features[i:i+batch_size].detach()
                action_batch = actions[i:i+batch_size].detach()
                scalar_reward_batch = torch.matmul(rewards[i:i+batch_size].detach(), torch.tensor(ref_grad).float())

                current_batch_size = scalar_reward_batch.size()[0]
                policy_dists, state_values = actor_critic.forward(features_batch)

                log_probs = torch.log(policy_dists.gather(1, action_batch))

                returns = np.zeros(current_batch_size)
                td_return = state_values[-1].detach().numpy()
                for t in reversed(range(current_batch_size)):   # phantom addition?
                    td_return = scalar_reward_batch[t] + gamma * td_return
                    returns[t] = td_return
                print(features_batch)
                print(action_batch)
                print(state_values)
                print(returns)
                advantage = torch.tensor(returns).float().unsqueeze(1) - state_values

                print(advantage)

                actor_loss = (-log_probs * advantage).mean()
                critic_loss = advantage.pow(2).mean()
                actor_critic_loss = actor_loss + critic_loss

                actor_critic_optimizer.zero_grad()
                actor_critic_loss.backward()
                actor_critic_optimizer.step()

        # Reinitialise sum of grad_objective psi
        psi = 0

        current_epoch_start = current_time
        while psi <= gradient_threshold and current_time <= max_time:

            # Take one step forward
            policy_dist, value = actor_critic(current_features)

            current_action = np.random.choice(num_actions, p=policy_dist.clone().detach().numpy())

            next_features, _, _, _ = env.step(current_action)  # TODO: Clean up the step function somehow
            reward_vector = env.algo_step(current_action)

            if current_time >= (max_time - 6000):
                env.render()

            # Store new transition in history
            agent.update_history(next_features, current_action, reward_vector)

            next_features = torch.tensor(next_features).float()

            agent.add_transition(current_features, torch.tensor([[current_action]]),
                                 next_features, torch.tensor(reward_vector).float().unsqueeze(0))

            """
            # Update actor critic parameters
            list_log_prob.append(torch.log(policy_dist[current_action]))
            list_values.append(value)
            with torch.no_grad():
                list_scalar_rewards.append(scalar_reward_estimator(current_features).gather(0, torch.tensor([current_action])))
                
            actor_critic.eval()
            #entropy = -np.sum(np.mean(policy_dist.clone().detach().numpy()) * np.log(policy_dist.clone().detach().numpy()))
            log_prob = torch.log(policy_dist[current_action])
            with torch.no_grad():
                _, new_value = actor_critic(next_features)
                #print("NEW")
                #print(current_time)
                #print(current_features)
                #print(new_value)
                #print(value)
                #print(tuple(current_features.numpy().astype(int)))
                scalar_reward_estimate = real_reward_dict[tuple(current_features.numpy().astype(int))].gather(0, torch.tensor([current_action]))
                #print(scalar_reward_estimate)
                #print(policy_dist)
                #scalar_reward_estimate = scalar_reward_estimator(current_features).gather(0, torch.tensor([current_action])).detach()
                #print(scalar_reward_estimate)

            advantage = scalar_reward_estimate + gamma * new_value - value  # value explodes
            #print(advantage)
            #print(entropy)
            actor_loss = (-log_prob * advantage.clone().detach())
            critic_loss = advantage.pow(2)
            actor_critic_loss = actor_loss + critic_loss  # + entropy  # TODO: Add entropy term?
            #print("Actor Loss")
            #print(actor_loss)
            #print("Critic Loss")
            #print(critic_loss)
            #print(actor_critic_loss)

            actor_critic_optimizer.zero_grad()
            #print(actor_critic.critic[0].weight.grad)
            actor_critic_loss.backward()  # to change
            #print(actor_critic.critic[0].weight.grad)
            actor_critic_optimizer.step()
            """
            # Update gradient counter
            _, current_grad = agent.get_objectives()
            psi += np.linalg.norm(current_grad - ref_grad)

            # Update current step
            current_features = next_features
            current_time += 1








