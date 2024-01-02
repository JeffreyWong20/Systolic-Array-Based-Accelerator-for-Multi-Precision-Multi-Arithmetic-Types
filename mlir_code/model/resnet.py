# import torch
# import torch.nn as nn
# import torchvision.models as models

# # Load the pre-trained ResNet-18 model
# resnet18 = models.resnet18(pretrained=True)

# # Modify the final fully connected layer for your specific task
# num_classes = 1000  # Replace with the number of classes in your dataset
# resnet18.fc = nn.Linear(resnet18.fc.in_features, num_classes)

# # Optionally, add a softmax layer if your task requires it
# # softmax = nn.Softmax(dim=1)
# # You can add the softmax layer if you're using cross-entropy loss

# # Print the modified ResNet-18 model
# print(resnet18)


# from torchsummary import summary

# # Print the model summary
# summary(resnet18, (3, 224, 224))  # Input shape (3 channels, 224x224 image)



import torch
import torchvision.models as models
import torch.fx

# Load the ResNet-18 model
resnet18 = models.resnet18(pretrained=True)

# Create an example input tensor
input_tensor = torch.randn(1, 3, 224, 224)  # Assuming 3-channel RGB image with 224x224 resolution

# Convert the model to a GraphModule
model_fx = torch.fx.symbolic_trace(resnet18)

# Print the graph
print(model_fx.graph)
