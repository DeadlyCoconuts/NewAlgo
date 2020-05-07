import torch
import torch.nn as nn


class RewardVectorEstimator(nn.Module):
    def __init__(self, num_dim_features, num_dim_rewards, hidden_size):
        super(RewardVectorEstimator, self).__init__()

        self.estimator = nn.Sequential(
            nn.Linear(num_dim_features, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_dim_rewards)
        )

    def forward(self, feature_vector):
        reward_estimate = self.estimator(feature_vector)
        return reward_estimate

