import torch
import torch.nn as nn


class VideoSegmentationModel(nn.Module):
  def __init__(self, input_size, hidden_size, num_discrete_values):
    super().__init__()

    self.rnn = nn.LSTM(input_size, hidden_size)
    self.fc1 = nn.Linear(hidden_size + num_discrete_values, hidden_size)
    self.fc2 = nn.Linear(hidden_size, hidden_size)
    self.fc3 = nn.Linear(hidden_size, 1)
    self.relu = nn.ReLU()

  def forward(self, prev_hidden_states, current_image, discrete_values):
    # Compute the hidden state for the current frame using the previous hidden
    # states and the current image
    x, (h, c) = self.rnn(current_image, prev_hidden_states)

    # Concatenate the hidden state with the discrete values
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
