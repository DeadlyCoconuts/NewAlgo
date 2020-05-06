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
            nn.Linear(hidden_size)
        )

    def forward(self, x):
        policy_dist = self.actor(x)
        value = self.critic(x)
        return policy_dist, value
