import torch
import torch.nn as nn
import torch.nn.functional as F

#
# class ValueNetwork(nn.Module):
#
#
#     def __init__(self, state_size, action_size, seed=0, fc1_units=64, fc2_units=64):
#         """Initialize parameters and build model.
#         Params
#         ======
#             state_size (int): Dimension of each state
#             action_size (int): Dimension of each action
#             seed (int): Random seed
#             fc1_units (int): Number of nodes in first hidden layer
#             fc2_units (int): Number of nodes in second hidden layer
#         """
#         super(ValueNetwork, self).__init__()
#         self.seed = torch.manual_seed(seed)
#         self.fc1 = nn.Linear(state_size, fc1_units)
#         self.fc2 = nn.Linear(fc1_units, fc2_units)
#         self.fc3 = nn.Linear(fc2_units, action_size)
#
#     def forward(self, state):
#         """Build a network that maps state -> action values."""
#         x = F.relu(self.fc1(state))
#         x = F.relu(self.fc2(x))
#         return self.fc3(x)

class ValueNetwork(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, output_dim)

    def forward(self, state):
        value = F.relu(self.fc1(state))
        value = self.fc2(value)

        return value


class TwoHeadNetwork(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(TwoHeadNetwork, self).__init__()
        self.policy1 = nn.Linear(input_dim, 256)
        self.policy2 = nn.Linear(256, output_dim)

        self.value1 = nn.Linear(input_dim, 256)
        self.value2 = nn.Linear(256, 1)

    def forward(self, state):
        logits = F.relu(self.policy1(state))
        logits = self.policy2(logits)

        value = F.relu(self.value1(state))
        value = self.value2(value)

        return logits, value

# class PolicyNetwork(nn.Module):
#
#
#     def __init__(self, state_size, action_size, seed=0, fc1_units=64, fc2_units=64):
#         """Initialize parameters and build model.
#         Params
#         ======
#             state_size (int): Dimension of each state
#             action_size (int): Dimension of each action
#             seed (int): Random seed
#             fc1_units (int): Number of nodes in first hidden layer
#             fc2_units (int): Number of nodes in second hidden layer
#         """
#         super(PolicyNetwork, self).__init__()
#         self.seed = torch.manual_seed(seed)
#         self.fc1 = nn.Linear(state_size, fc1_units)
#         self.fc2 = nn.Linear(fc1_units, fc2_units)
#         self.fc3 = nn.Linear(fc2_units, action_size)
#
#     def forward(self, state):
#         """Build a network that maps state -> action values."""
#         x = F.relu(self.fc1(state))
#         x = F.relu(self.fc2(x))
#         return self.fc3(x)

class PolicyNetwork(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, output_dim)

    def forward(self, state):
        logits = F.relu(self.fc1(state))
        logits = self.fc2(logits)

        return logits