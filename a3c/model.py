import torch.nn as nn
import torch.nn.functional as F

class ActorCriticNetwork(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(ActorCriticNetwork, self).__init__()
        self.policy1 = nn.Linear(input_dim, 64)
        self.policy2 = nn.Linear(64, 64)
        self.policy3 = nn.Linear(64, output_dim)

        self.value1 = nn.Linear(input_dim, 64)
        self.value2 = nn.Linear(64, 64)
        self.value3 = nn.Linear(64, 1)

    def forward(self, state):
        logits = F.relu(self.policy1(state))
        logits = F.relu(self.policy2(logits))
        logits = self.policy3(logits)

        value = F.relu(self.value1(state))
        value = F.relu(self.value2(value))
        value = self.value3(value)

        return logits, value

