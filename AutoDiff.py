import torch

# Initialize tensors
x = torch.ones(5)  # input tensor
y = torch.zeros(3)  # expected output
w = torch.randn(5, 3, requires_grad=True)
b = torch.randn(3, requires_grad=True)

# Perform the matrix multiplication and addition
z = torch.matmul(x, w) + b

# Compute the loss
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)

# Print the values
print(f"Input tensor (x): {x}")
print(f"Expected output (y): {y}")
print(f"Weights (w): {w}")
print(f"Biases (b): {b}")
print(f"Logits (z): {z}")
print(f"Loss: {loss.item()}")

# Get the gradient functions
print(f"Gradient function for z = {z.grad_fn}")
print(f"Gradient function for loss = {loss.grad_fn}")

# Compute the gradients
loss.backward()
print(w.grad)
print(b.grad)

# Disable gradient tracking
z = torch.matmul(x, w) + b
print(z.requires_grad)

with torch.no_grad():
    z = torch.matmul(x, w) + b
print(z.requires_grad)

z = torch.matmul(x, w) + b
z_det = z.detach()
print(z_det.requires_grad)
