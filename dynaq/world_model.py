import torch
import torch.nn as nn
import torch.nn.functional as F

class WorldModelNetwork(nn.Module):

    def __init__(self, state_size, action_size, seed=0, fc1_units=64, fc2_units=64):
        super(WorldModelNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1_s = nn.Linear(state_size, fc1_units)
        self.fc1_a = nn.Linear(action_size, 1)

        self.action_size = action_size

        units = fc2_units + 1
        self.fc2 = nn.Linear(units, units)

        self.fch = nn.Linear(units, units)


        self.fc3 = nn.Linear(units, state_size)
        self.fc4 = nn.Linear(units, 1)
        self.fc5 = nn.Linear(units, 1)

        self.out_act = nn.Sigmoid()


    def encode_action(self, action):
        return torch.nn.functional.one_hot(torch.squeeze(action, 1), num_classes=self.action_size).float()

    def forward(self, state, action):
        """Build a network that maps state -> action values."""

        xs = F.relu(self.fc1_s(state))

        xa = F.relu(self.fc1_a(action)) #one hot this

        x = torch.cat((xs, xa), dim=1)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fch(x))

        return self.fc3(x), self.fc4(x), self.out_act(self.fc5(x))

    def get_gradients(self):
        grads = []
        for p in self.parameters():
            grad = None if p.grad is None else p.grad.data.cpu().numpy()
            grads.append(grad)
        return grads

    def set_gradients(self, gradients):
        for g, p in zip(gradients, self.parameters()):
            if g is not None:
                p.grad = torch.from_numpy(g)

    def get_weights(self):
        return {k: v.cpu() for k, v in self.state_dict().items()}

    def set_weights(self, weights):
        self.load_state_dict(weights)