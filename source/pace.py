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
        #proper residual connection
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None 
        self.relu = nn.ReLU()
        self.init_weights()
    
    
    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class AttentionBlock(nn.Module):   # Single head attention
    def __init__(self, input_dim):
        super(AttentionBlock, self).__init__()
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.sqrt_d = math.sqrt(input_dim)

    def forward(self, x):
        x = x.permute(0, 2, 1)    
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.sqrt_d
        attn = torch.softmax(scores, dim=-1)
        output = torch.matmul(attn, V)
        return output.permute(0, 2, 1)  

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads=8, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        batch_size, channels, seq_len = x.size()
        x = x.permute(0, 2, 1)  
        residual = x
        Q = self.w_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model)
        output = self.w_o(attn_output)
        output = self.layer_norm(output + residual)
        return output.permute(0, 2, 1)  
    
class ChunkedAttention(nn.Module):
    def __init__(self, input_dim, chunk_size=32, num_heads=8):
        super(ChunkedAttention, self).__init__()
        self.input_dim = input_dim
        self.chunk_size = chunk_size
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads  
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.output_proj = nn.Linear(input_dim, input_dim)
        self.sqrt_d = math.sqrt(self.head_dim)

    def forward(self, x):
        B, C, L = x.shape
        x = x.permute(0, 2, 1)  
        
        Q = self.query(x).view(B, L, self.num_heads, self.head_dim)
        K = self.key(x).view(B, L, self.num_heads, self.head_dim)
        V = self.value(x).view(B, L, self.num_heads, self.head_dim)
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)  
        V = V.transpose(1, 2)
        
        # Process in chunks
        outputs = []
        for i in range(0, L, self.chunk_size):
            end_idx = min(i + self.chunk_size, L)
            q_chunk = Q[:, :, i:end_idx, :]  
            k_chunk = K[:, :, i:end_idx, :]    
            v_chunk = V[:, :, i:end_idx, :]  
            
            # Compute attention scores within the chunk
            scores = torch.matmul(q_chunk, k_chunk.transpose(-2, -1)) / self.sqrt_d
            attn = torch.softmax(scores, dim=-1)
            chunk_out = torch.matmul(attn, v_chunk)
            outputs.append(chunk_out)
        
        # Concatenate all chunks
        output = torch.cat(outputs, dim=2)  
        
        # Reshape back to [B, L, input_dim]
        output = output.transpose(1, 2).contiguous().view(B, L, -1)
        output = self.output_proj(output)
        return output.permute(0, 2, 1)  

class PACE(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size=3, dropout=0.2, 
                 attention='chunk', output_window=30, num_heads=8, chunk_size= 10 ):
        super(PACE, self).__init__()
        self.output_window = output_window
        self.attention = attention
        self.chunk_size = chunk_size
        # Build temporal blocks
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_size if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            # Add temporal block
            layers.append(TemporalBlock(
                in_channels, out_channels, kernel_size, stride=1,
                dilation=dilation_size, 
                padding=(kernel_size-1)*dilation_size,
                dropout=dropout
            ))
            # # Add attention after every few temporal blocks (not after each one)
            if self.attention and (i + 1) % 2 == 0:
                if self.attention == 'single':
                    layers.append(AttentionBlock(out_channels))
                elif self.attention == 'multi':
                    layers.append(MultiHeadAttention(out_channels, num_heads, dropout))
                elif self.attention == 'chunk':
                    layers.append(ChunkedAttention(out_channels, chunk_size=self.chunk_size, num_heads=num_heads))

        self.network = nn.Sequential(*layers)
        # Output layers with better architecture
        self.output_projection = nn.Sequential(
            nn.AdaptiveAvgPool1d(output_window * 2),  # Intermediate pooling
            nn.Conv1d(num_channels[-1], num_channels[-1] // 2, 1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(num_channels[-1] // 2, output_size, 1)
        )
        self.use_linear_output = True
        if self.use_linear_output:
            self.linear_output = nn.Sequential(
                nn.Linear(num_channels[-1], num_channels[-1] // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(num_channels[-1] // 2, output_size)
            )
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        batch_size, seq_len, features = x.size()
        x = x.permute(0, 2, 1)
        out = self.network(x)
        if self.use_linear_output:
            out = out.permute(0, 2, 1) 
            out = self.linear_output(out)  
            if seq_len >= self.output_window:
                return out[:, -self.output_window:, :]
            else:
                # If input sequence is shorter, pad or repeat
                padding_needed = self.output_window - seq_len
                if padding_needed > 0:
                    # Repeat the last timestep
                    last_step = out[:, -1:, :].repeat(1, padding_needed, 1)
                    out = torch.cat([out, last_step], dim=1)
                return out
        else:
            out = self.output_projection(out)  # [batch, output_size, output_window]
            return out.permute(0, 2, 1)  # [batch, output_window, output_size]
        
