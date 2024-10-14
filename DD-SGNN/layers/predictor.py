import torch.nn as nn
import torch

if torch.cuda.is_available():
    device = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

class Predictor(nn.Module):

    def __init__(self, in_channels=64):
        super().__init__()

        # TODO add another methods ...

        # 2-Linear MLP
        self.predictor = nn.Sequential(nn.Linear(in_channels * 2, in_channels),
                                       nn.ReLU(),
                                       nn.Linear(in_channels, 1)).to(device)

    def forward(self, ux, vx):
        """link (u, v)"""

        x = torch.concat((ux, vx), dim=-1)
        res = self.predictor(x).flatten()

        return res
