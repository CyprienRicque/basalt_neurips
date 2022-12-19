import torch
import torch.nn as nn

class VideoSegmentationModel(nn.Module):
  def __init__(self, input_size, hidden_size, num_discrete_values):
    super().__init__()

    self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
    self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
    self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
    self.fc1 = nn.Linear(128 * 128 * 128 + num_discrete_values, hidden_size)
    self.fc2 = nn.Linear(hidden_size, hidden_size)
    self.fc3 = nn.Linear(hidden_size, 1)
    self.relu = nn.ReLU()

  def forward(self, prev_hidden_states, current_image, discrete_values):
    # Apply a series of convolutional layers to the current image
    x = self.conv1(current_image)
    x = self.relu(x)
    x = self.conv2(x)
    x = self.relu(x)
    x = self.conv3(x)
    x = self.relu(x)

    # Flatten the output of the convolutional layers
    x = x.view(x.size(0), -1)

    # Concatenate the flattened output with the discrete values
    x = torch.cat([x, discrete_values], dim=1)

    # Apply a few fully-connected layers with ReLU activation
    x = self.fc1(x)
    x = self.relu(x)
    x = self.fc2(x)
    x = self.relu(x)
    x = self.fc3(x)

    # Return the segment id
    return torch.argmax(x, dim=1)

model = VideoSegmentationModel(128 * 128 * 3, 128, 2)
