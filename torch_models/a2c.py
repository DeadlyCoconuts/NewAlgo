import torch
import torch.nn as nn


class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_size):  # TODO: Refine network
        super(ActorCritic, self).__init__()

        self.actor = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_outputs),
            nn.Softmax(dim=1)
        )

        self.critic = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, feature_vector):
        policy_dist = self.actor(feature_vector)
        value = self.critic(feature_vector)
        return policy_dist, value
