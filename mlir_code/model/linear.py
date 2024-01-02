import torch

# Define input and weight matrices
input_size = 5
output_size = 3
batch_size = 2

input_data = torch.randn(batch_size, input_size)  # Input data
weight_matrix = torch.randn(output_size, input_size)  # Weight matrix
bias = torch.randn(output_size)  # Bias vector

# Initialize output
output = torch.zeros(batch_size, output_size)

# Perform matrix multiplication and bias addition using nested for loops
for batch in range(batch_size):
    for out_dim in range(output_size):
        for in_dim in range(input_size):
            output[batch][out_dim] += input_data[batch][in_dim] * weight_matrix[out_dim][in_dim]
        output[batch][out_dim] += bias[out_dim]

print(output)








