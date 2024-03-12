import torch

# Assuming your tensor is named 'tensor'
tensor = torch.arange(100).reshape(10, 2, 5)
indices = torch.full((10, 1, 5), 1, dtype=torch.int64)
# Now we collect all the values from the tensor using the indices.
# The output is a tensor of shape [5, 4, 2]
output = torch.gather(tensor, 1, indices)
print(output)
print(output.shape)
