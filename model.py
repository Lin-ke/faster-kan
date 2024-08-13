import torch.nn as nn
# multiple layer perceptron
class MLP(nn.Module):
    def __init__(self, dim_in, dim_out, hidden_num = 3, hidden_dim = 1024):
        super().__init__()
        self.model_base = nn.ModuleList([
            nn.ModuleList([nn.Linear(dim_in, hidden_dim) if i == 0 else nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()])
            for i in range(hidden_num)
        ])
        self.lin = nn.Linear(hidden_dim, dim_out)

    def forward(self, x):
        for layer in self.model_base:
            for l in layer:
                x = l(x)
        return self.lin(x)
