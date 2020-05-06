import torch
import torch.nn as nn


class ScalarRewardEstimator(nn.Module):
    def __init__(self, num_dim_features, num_dim_rewards, hidden_size):
        super(ScalarRewardEstimator, self).__init__()

        self.estimator = nn.Sequential(
            nn.Linear(num_dim_features, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_dim_rewards),
            nn.Linear(num_dim_rewards, 1)
        )

    def forward(self, feature_vector):
        scalar_reward_estimate = self.estimator(feature_vector)
        return scalar_reward_estimate

    def add_grad_objective_weights(self, grad_objective, state_dict):  # adds the grad_objective weights to the last module
        state_dict['estimator.3.weight'] = grad_objective  # really ugly code; really shouldn't be doing this...
        state_dict['estimator.3.bias'] = torch.zeros(1)  # TODO: think of a way to beautify this
        self.load_state_dict(state_dict)
        return
