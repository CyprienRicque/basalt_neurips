import torch
import torch.nn as nn


# class VideoSegmentationModel(nn.Module):
#   def __init__(self, input_size, hidden_size, num_discrete_values):
#     super().__init__()
#
#     self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
#     self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
#     self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
#     self.rnn = nn.LSTM(128 * 128 * 128 + num_discrete_values, hidden_size)
#     self.fc1 = nn.Linear(hidden_size, hidden_size)
#     self.fc2 = nn.Linear(hidden_size, hidden_size)
#     self.fc3 = nn.Linear(hidden_size, 1)
#     self.relu = nn.ReLU()
#
#     # Initialize the hidden states of the LSTM
#     self.hidden_state = None
#     self.cell_state = None
#
#   def forward(self, current_image, discrete_values):
#     # Initialize the hidden states if they have not been initialized yet
#     if self.hidden_state is None:
#       self.hidden_state = torch.zeros(1, current_image.size(0), self.rnn.hidden_size)
#       self.cell_state = torch.zeros(1, current_image.size(0), self.rnn.hidden_size)
#
#     # Apply a series of convolutional layers to the current image
#     x = self.conv1(current_image)
#     x = self.relu(x)
#     x = self.conv2(x)
#     x = self.relu(x)
#     x = self.conv3(x)
#     x = self.relu(x)
#
#     # Flatten the output of the convolutional layers
#     x = x.view(x.size(0), -1)
#
#     # Compute the hidden state for the current frame using the previous hidden
#     # states and the output of the convolutional layers
#     x, (self.hidden_state, self.cell_state) = self.rnn(x, (self.hidden_state, self.cell_state))
#
#     # Concatenate the hidden state with the discrete values
#     x = torch.cat([x, discrete_values], dim=1)
#
#     # Apply a few fully-connected layers with ReLU activation
#     x = self.fc1(x)
#     x = self.relu(x)
#     x = self.fc2(x)
#     x = self.relu(x)
#     x = self.fc3(x)
#
#     # Return the segment id
#     return torch.argmax(x, dim=1)


class VideoSegmentationModel(nn.Module):
    def __init__(self, hidden_size, num_discrete_values, steps):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.rnn = nn.LSTM(128 * 128 * 128 + num_discrete_values, hidden_size)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, steps)

    def forward(self, current_image, discrete_values, hidden_state):
        # Pass image through convolutional layers
        x = self.conv1(current_image)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)

        # Reshape image for RNN
        x = x.view(-1, 128 * 128 * 128)

        # Pass image through RNN
        x, hidden_state = self.rnn(torch.cat((x, discrete_values), dim=1), hidden_state)

        # Pass output of RNN through fully connected layers
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        return torch.argmax(x, dim=1), hidden_state


class HiddenStateManager:
    def __init__(self, model):
        self.model = model
        self.hidden_states = {}

    def init_hidden(self, batch_size):
        """Initialize hidden state"""
        # Create tensors of zeros for hidden and cell states
        hidden = torch.zeros(1, batch_size, self.model.hidden_size)
        cell = torch.zeros(1, batch_size, self.model.hidden_size)

        # Return hidden state as tuple
        return hidden, cell

    def get_hidden(self, episode_id):
        """Get hidden state for episode"""
        if episode_id in self.hidden_states:
            return self.hidden_states[episode_id]
        else:
            return self.init_hidden(1)

    def update_hidden(self, episode_id, hidden_state):
        """Update hidden state for episode"""
        self.hidden_states[episode_id] = hidden_state

    def remove_hidden(self, episode_id):
        """Remove hidden state for episode"""
        if episode_id in self.hidden_states:
            del self.hidden_states[episode_id]
