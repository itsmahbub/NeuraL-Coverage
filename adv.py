import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision.models import resnet18

# Define a ResNet-18 model adapted to MNIST (1-channel input, 10 classes)
model = resnet18(pretrained=False, num_classes=10)
model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
model.eval()

def one_hot_encode(class_index, num_classes=10):
    """
    Converts a class index to a one-hot encoded tensor as float.
    
    Parameters:
      class_index (int): Class index (0-9 for MNIST).
      num_classes (int): Total number of classes.
    
    Returns:
      torch.Tensor: A one-hot vector of shape (num_classes,) with type float.
    """
    one_hot = torch.zeros(num_classes, dtype=torch.float)
    one_hot[class_index] = 1.0
    return one_hot

def modify_tensor_to_target_class(model, one_hot_target, epsilon=0.1, num_iterations=10):
    """
    Generates a random 28x28 tensor and updates it using gradient-based methods so that 
    the model is nudged toward predicting the specified target class (supplied as a one-hot vector).
    
    Parameters:
      model (nn.Module): The pretrained MNIST-adapted model.
      one_hot_target (torch.Tensor): A one-hot encoded target with dtype torch.float of shape (num_classes,).
      epsilon (float): Step size for each update.
      num_iterations (int): Number of iterations to perform.
    
    Returns:
      torch.Tensor: The modified input tensor of shape [1, 1, 28, 28].
    """
    # Create a random tensor resembling a MNIST image.
    input_tensor = torch.rand(1, 1, 28, 28, requires_grad=True)
    loss_fn = nn.CrossEntropyLoss()

    # When using soft targets (one-hot vector) with CrossEntropyLoss, the target must be of shape
    # [batch_size, num_classes] and be a floating point tensor.
    soft_target = one_hot_target.unsqueeze(0)  # shape becomes (1, num_classes)

    # for _ in range(num_iterations):
    while True:
        # Zero out previous gradients.
        model.zero_grad()
        if input_tensor.grad is not None:
            input_tensor.grad.zero_()
        
        # Forward pass.
        output = model(input_tensor)  # Expected shape: (1,10)
        print(output)
        predicted_class = output.argmax(dim=1).item()
        if predicted_class == target_class:
            break
        
        # Compute the loss using soft targets.
        loss = loss_fn(output, soft_target)
        loss.backward()
        
        # Update the input tensor using the gradient sign.
        with torch.no_grad():
            input_tensor += epsilon * input_tensor.grad.sign()
    
    return input_tensor

# Convert a target class (e.g., 5) into one-hot encoding with floating type.
target_class = 5
one_hot_target = one_hot_encode(target_class, num_classes=10)

# Modify the random image tensor toward the given target.
modified_image = modify_tensor_to_target_class(model, one_hot_target, epsilon=0.1, num_iterations=10)

# Predict the class of the modified image.
with torch.no_grad():
    output = model(modified_image)
    predicted_class = output.argmax(dim=1).item()
print("Predicted class:", predicted_class)

# Draw the modified image.
modified_image_np = modified_image.squeeze().detach().numpy()
plt.imshow(modified_image_np, cmap='gray')
plt.title("Modified Image Tensor")
plt.axis('off')
plt.savefig('./img.png')


# import torch
# import torch.nn as nn
# from torchvision.models import resnet18
# import matplotlib.pyplot as plt
# import numpy as np

# # Create a ResNet-18 with 10 output classes to mimic an MNIST-trained model.
# model = resnet18(pretrained=False, num_classes=10)
# # Modify the first convolutional layer to accept 1-channel input instead of 3.
# model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

# # For demonstration purposes, we're not loading actual pretrained weights,
# # but in practice you would load MNIST-trained weights here.
# model.eval()

# def modify_tensor_to_target_class(model, target_class, epsilon=0.1, num_iterations=10):
#     """
#     Generates a random 28x28 tensor and iteratively updates it using gradients
#     so that the model becomes more likely to predict the given target class.
#     """
#     # Create a random tensor simulating a MNIST image.
#     input_tensor = torch.rand(1, 1, 28, 28, requires_grad=True)
#     loss_fn = nn.CrossEntropyLoss()

#     for _ in range(num_iterations):
#         model.zero_grad()
#         if input_tensor.grad is not None:
#             input_tensor.grad.data.zero_()

#         # Forward pass.
#         output = model(input_tensor)
#         print(output)
#         # target = torch.tensor([target_class])
#         target = torch.nn.functional.one_hot(torch.tensor([target_class]), num_classes=10)

#         loss = loss_fn(output, target)
#         print(output.shape)
#         print(target.shape)
        
#         # Backward pass.
#         loss.backward()

#         # Update the input tensor using a sign update (an FGSM-inspired step).
#         with torch.no_grad():
#             input_tensor += epsilon * input_tensor.grad.sign()
#             output = model(input_tensor)
#             predicted_class = output.argmax(dim=1).item()
#             if predicted_class == target_class:
#                 break

#     return input_tensor

# # Define the target class (for example, class 5) and get the modified image tensor.
# target_class = 5
# modified_image = modify_tensor_to_target_class(model, target_class, epsilon=0.1, num_iterations=10)

# # Predict the output class of the modified image
# with torch.no_grad():
#     output = model(modified_image)
#     print(output)
#     predicted_class = output.argmax(dim=1).item()
# print("Predicted class:", predicted_class)

# # Convert the modified image tensor to a numpy array for plotting.
# modified_image_np = modified_image.detach().numpy().squeeze(0).squeeze(0)

# # Plot the modified image.
# plt.imshow(modified_image_np, cmap='gray')
# plt.axis('off')  # Hide axis for clarity.
# plt.title('Modified Image Tensor')
# plt.savefig('./img.png')
