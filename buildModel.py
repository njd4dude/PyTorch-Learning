import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# get device
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
print(f"Using {device} device")


# define neural network
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


# create instance of model
model = NeuralNetwork().to(device)
print(model)

# make prediction
X = torch.rand(1, 28, 28, device=device)
logits = model(X)
print("logits", logits)
pred_probab = nn.Softmax(dim=1)(logits)
print("pred_probab", pred_probab)
y_pred = pred_probab.argmax(1)
print(f"Predicted class: {y_pred}")

print("\n\n\n")

# Model layers break down mnist fahsion dataset model
input_image = torch.rand(3, 28, 28)
print(input_image.size())
# nn.Flatten
flatten = nn.Flatten()
flat_image = flatten(input_image)
print(flat_image.size())
# nn.Linear
layer1 = nn.Linear(in_features=28 * 28, out_features=20)
hidden1 = layer1(flat_image)
print(hidden1.size())
# nn.ReLU
print(f"Before ReLU: {hidden1}\n")
hidden1 = nn.ReLU()(hidden1)
print(f"After ReLU: {hidden1}")
# nn.Sequential
seq_modules = nn.Sequential(flatten, layer1, nn.ReLU(), nn.Linear(20, 10))
input_image = torch.rand(3, 28, 28)
logits = seq_modules(input_image)
print("logits", logits)
# nn.Softmax
softmax = nn.Softmax(dim=1)
pred_probab = softmax(logits)
print("pred_probab", pred_probab)


# Model Parameters
print(f"Model structure: {model}\n\n")
for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")
