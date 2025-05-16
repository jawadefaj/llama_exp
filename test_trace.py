import torch
import torch.nn as nn
import tg4perfetto

class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

model = SimpleMLP()
x = torch.randn(1, 10)

track_fc1 = tg4perfetto.track("FC1")
track_relu = tg4perfetto.track("ReLU")
track_fc2 = tg4perfetto.track("FC2")

with tg4perfetto.open("simple_mlp_trace.json"):
    with torch.no_grad():
        with tg4perfetto.track("Model").trace("Forward Pass"):
            with track_fc1.trace("FC1"):
                x = model.fc1(x)
            with track_relu.trace("ReLU"):
                x = model.relu(x)
            with track_fc2.trace("FC2"):
                x = model.fc2(x)

print("✅ simple_mlp_trace.json written successfully.")
