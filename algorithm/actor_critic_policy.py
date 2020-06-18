import gym
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn

from torch import autograd
from torch_models.reward_vector_estimator import RewardVectorEstimator
from torch_models.scalar_reward_estimator import ScalarRewardEstimator
from torch_models.advantage_actor_critic import ActorCritic

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

class ActorCriticAlgorithm(object):
    def __init__(self, env_code, num_dim_features, device, num_steps=100, max_episodes=350, gamma=0.99):
        self.rewards = None
        self.env = gym.make(env_code).env

        self.num_actions = self.env.action_space.n
        # to retrieve num_dim_features directly ?

        self.num_steps = num_steps
        self.max_episodes = max_episodes
        self.gamma = gamma

        self.actor_critic = ActorCritic(num_dim_features, self.num_actions, hidden_size=50).to(device)
        self.actor_critic_optimizer = optim.Adam(self.actor_critic.parameters())

    def generate_policy(self, reward_dict):  # Returns neural net?
        self._reset_parameters()

        list_episodic_rewards = []
        entropy_term = 0

        for episode in range(self.max_episodes):
            list_log_probs = []
            list_state_values = []
            list_rewards = []

            q_value = 0

            current_state = torch.tensor(get_one_hot(self.env.reset())).float()
            for step in range(self.num_steps):
                policy_dist, state_value = self.actor_critic(current_state)

                list_state_values.append(state_value)
                state_value = state_value.detach().numpy()

                soft_max_transform = nn.Softmax(dim=0)
                dist = soft_max_transform(policy_dist).detach().numpy()

                # check dist
                action = np.random.choice(self.num_actions, p=np.squeeze(dist))

                log_soft_max_transform = nn.LogSoftmax(dim=0)
                log_prob = log_soft_max_transform(policy_dist)[action]

                entropy = -np.sum(np.mean(dist) * np.log(dist))
                new_state, _, _, _ = self.env.step(action)

                list_rewards.append(reward_dict[tuple(new_state)][action])
                #list_state_values.append(state_value)
                list_log_probs.append(log_prob)
                entropy_term += entropy

                current_state = torch.tensor(get_one_hot(new_state)).float()

                if step == self.num_steps - 1:
                    _, final_state_value = self.actor_critic(current_state)
                    q_value = final_state_value.detach().numpy()  # to be used in computation of q_values via
                    # bootstrapping below

                    list_episodic_rewards.append(np.sum(list_rewards))
                    if episode % 10 == 0:
                        print("Episode: {}, Total Reward for this Episode: {}".format(episode, list_episodic_rewards[-1]))
                    break

            list_q_values = np.zeros_like(list_rewards)
            #actor_loss = 0
            #critic_loss = 0
            for t in reversed(range(len(list_rewards))):
                q_value = list_rewards[t] + self.gamma * q_value
                list_q_values[t] = q_value
                #advantage = q_value - list_state_values[t]
                #actor_loss += (-list_log_probs[t] * advantage)
                #critic_loss += advantage**2

            #print(list_q_values)
            # better renaming !!!
            list_state_values = torch.stack(list_state_values)
            list_q_values = torch.tensor(list_q_values).float()
            list_log_probs = torch.stack(list_log_probs)

            advantage = list_q_values - list_state_values
            actor_loss = (-list_log_probs * advantage).mean()
            critic_loss = advantage.pow(2).mean()

            actor_critic_loss = actor_loss + critic_loss + 0.001 * entropy_term

            self.actor_critic_optimizer.zero_grad()
            actor_critic_loss.backward()
            self.actor_critic_optimizer.step()

        self.actor_critic.eval()
        test, _ = self.actor_critic(torch.tensor(get_one_hot(self.env.reset())).float())

        return self.actor_critic


    def _reset_parameters(self):
        self.actor_critic.reset_model()
        self.actor_critic_optimizer = optim.Adam(self.actor_critic.parameters())


"""
batch_size = 1 # Doesn't work for > 1 ?
if current_time > 0:
    print("hiya")
    print(agent.avg_reward_list[-1])
    print(ref_grad)
    for i in range(0, len(agent.transition_list), batch_size):
        # print(i)
        # print("HEHAHAH")
        features_batch = features[i:i + batch_size].detach()
        action_batch = actions[i:i + batch_size].detach()
        scalar_reward_batch = torch.matmul(rewards[i:i + batch_size].detach(), torch.tensor(ref_grad).float())

        current_batch_size = scalar_reward_batch.size()[0]
        with autograd.detect_anomaly():
            policy_dists, state_values = actor_critic.forward(features_batch)

        a = nn.LogSoftmax(dim=1)
        # log_probs = torch.log(policy_dists.gather(1, action_batch))
        log_probs = a(policy_dists).gather(1, action_batch)

        b = nn.Softmax(dim=1)
        prob_dist = b(policy_dists).detach()
        log = a(policy_dists).detach()
        entropy = torch.sum(-torch.sum(torch.mul(prob_dist, log), 1), 0)

        returns = np.zeros(current_batch_size)
        td_return = state_values[-1].detach().numpy()

        # print(td_return)
        for t in reversed(range(current_batch_size)):  # phantom addition?
            td_return = scalar_reward_batch[t] + gamma * td_return
            returns[t] = td_return
        # print(features_batch)
        # print(action_batch)
        # print(scalar_reward_batch)
        # print(state_values)
        # print(returns)
        if np.isnan(returns[0]):
            return
        advantage = torch.tensor(returns).float().unsqueeze(1) - state_values  # detach return?

        # print(advantage)

        actor_loss = (-log_probs * advantage).mean()  # Crashing due to NaN exploding gradients

        critic_loss = advantage.pow(2).mean()
        actor_critic_loss = actor_loss + critic_loss  # + entropy

        actor_critic_optimizer.zero_grad()
        with autograd.detect_anomaly():
            actor_critic_loss.backward()

        actor_critic_optimizer.step()
"""