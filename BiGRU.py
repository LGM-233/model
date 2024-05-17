import torch
import torch.nn as nn

class BiGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(BiGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Define the bidirectional GRU layer
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True,dropout=0.3)

        # Define the output layer
        self.fc = nn.Linear(hidden_size * 2, output_size)  # Multiply by 2 for bidirectional

    def forward(self, x):
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)

        # Forward pass
        out, _ = self.gru(x, h0)

        # Extract the output of the last time step as the representation of the input sequence
        out = out[:, -1, :]

        # Use a fully connected layer for classification or other tasks
        out = self.fc(out)

        return out

if __name__ == '__main__':
    model = BiGRU(input_size=768, hidden_size=256, num_layers=1, output_size=256)

    input = torch.rand((1, 20, 768))
    output = model(input)
    print(output.shape)
