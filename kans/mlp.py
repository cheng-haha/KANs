import torch
import torch.nn as nn
import torch.optim as optim

# toy mlp
class MLP(nn.Module):
    def __init__(self, params_list) -> None:
        super().__init__()
        inp, hidden, oup = params_list
        self.mlp = nn.Sequential(
            nn.Linear(inp,hidden),
            nn.ReLU(True),
            nn.Linear(hidden,oup),
        )

    def forward(self,x):
        return self.mlp(x)