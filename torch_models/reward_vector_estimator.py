import torch
import torch.nn as nn


class RewardVectorEstimator(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_size):
        super(RewardVectorEstimator, self).__init__()

        self.estimator = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_outputs)
        )

    def forward(self, reward_vector):
        reward_estimate = self.estimator(reward_vector)
        return reward_estimate
