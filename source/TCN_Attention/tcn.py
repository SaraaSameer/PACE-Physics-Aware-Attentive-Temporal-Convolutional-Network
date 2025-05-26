import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import math
import numpy as np

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        # xavier uniform assign weights
        nn.init.xavier_uniform_(self.conv1.weight, gain=np.sqrt(2))
        nn.init.xavier_uniform_(self.conv2.weight, gain=np.sqrt(2))
        if self.downsample is not None:
            nn.init.xavier_uniform_(self.downsample.weight, gain=np.sqrt(2))

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class AttentionBlock(nn.Module):
    def __init__(self, input_dim):
        super(AttentionBlock, self).__init__()
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.sqrt_d = math.sqrt(input_dim)

    def forward(self, x):
        x = x.permute(0, 2, 1)    # Back to [batch, features, seq_len]
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.sqrt_d
        attn = torch.softmax(scores, dim=-1)
        output = torch.matmul(attn, V)
        return output.permute(0, 2, 1)  # Back to [batch, features, seq_len]

class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size=3, dropout=0.2, attention=True, output_window= 30):
        super(TCN, self).__init__()
        layers = []
        num_levels = len(num_channels)
        self.attention = attention
        self.output_window = output_window
        
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_size if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1,
                                   dilation=dilation_size, padding=(kernel_size-1)*dilation_size,
                                   dropout=dropout)]
            if attention:
                layers += [AttentionBlock(out_channels)]

        self.network = nn.Sequential(*layers)
        self.linear = nn.Linear(num_channels[-1], output_size)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.linear.weight, gain=np.sqrt(2))

    def forward(self, x):
        x = x.permute(0, 2, 1)  # [batch, features, seq_len]
        out = self.network(x)
        out = self.linear(out.permute(0, 2, 1))  # [batch, seq_len, output_size]
        return out[:, -self.output_window:, :]  # Return last 30 steps as prediction - equal to output window