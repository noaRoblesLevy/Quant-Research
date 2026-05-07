import torch.nn as nn


class LinearModel(nn.Module):
    def __init__(self, input_features: int):
        super().__init__()
        self.linear = nn.Linear(input_features, 1)

    def forward(self, x):
        return self.linear(x)


class NonLinearModel(nn.Module):
    def __init__(self, input_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_features, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)
