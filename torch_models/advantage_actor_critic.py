import torch
import torch.nn as nn


class ActorCritic(nn.Module):
    def __init__(self, num_features, num_actions, hidden_size):  # TODO: Refine network
        super(ActorCritic, self).__init__()

        self.actor = nn.Sequential(
            nn.Linear(num_features, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_actions),
            nn.Softmax(dim=0)
        )

        self.critic = nn.Sequential(
            nn.Linear(num_features, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, feature_vector):
        policy_dist = self.actor(feature_vector)
        value = self.critic(feature_vector)
        return policy_dist, value

    def reset_model(self):
        def weight_reset(module):
            if isinstance(module, nn.Linear):
                module.reset_parameters()

        self.actor.apply(weight_reset)
        self.critic.apply(weight_reset)
