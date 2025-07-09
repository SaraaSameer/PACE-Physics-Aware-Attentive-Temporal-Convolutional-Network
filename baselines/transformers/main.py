import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import time
import math
import argparse
from matplotlib import pyplot
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from dataset import BatteryDataset
from model_parameters import evaluate_model_efficiency  

torch.manual_seed(42)

# Argument parser
parser = argparse.ArgumentParser(description='Battery SoH Estimation using Transformer')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--input_window', type=int, default=100)
parser.add_argument('--output_window', type=int, default=30)
parser.add_argument('--feature_size', type=int, default=250)
parser.add_argument('--num_layers', type=int, default=1)
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--clip', type=float, default=0.5)
parser.add_argument('--cuda', action='store_true')
parser.add_argument('--runs', type=int, default=2)
args = parser.parse_args()

device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()       
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class TransformerSoH(nn.Module):
    def __init__(self, input_size, feature_size=250, num_layers=1, dropout=0.1, output_size=30):
        super(TransformerSoH, self).__init__()
        self.model_type = 'Transformer'
        self.input_size = input_size
        self.feature_size = feature_size
      
        self.input_projection = nn.Linear(input_size, feature_size)
        
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(feature_size)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_size, 
            nhead=10 if feature_size % 10 == 0 else 5, 
            dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)        
        self.decoder = nn.Linear(feature_size, 1)
        self.output_projection = nn.Linear(args.input_window, output_size)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1    
        self.input_projection.bias.data.zero_()
        self.input_projection.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)
        self.output_projection.bias.data.zero_()
        self.output_projection.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        batch_size, seq_len, input_size = src.shape
        src = self.input_projection(src)  
        src = src.transpose(0, 1)  
        
        if self.src_mask is None or self.src_mask.size(0) != seq_len:
            device = src.device
            mask = self._generate_square_subsequent_mask(seq_len).to(device)
            self.src_mask = mask

        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        output = self.decoder(output) 
        output = output.transpose(0, 1).squeeze(-1)  
        output = self.output_projection(output) 
        
        return output

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

def run_once(seed, run_id):
    torch.manual_seed(seed)
    train_dataset = BatteryDataset('train_features_ecm.csv', 
                                 input_window=args.input_window, 
                                 output_window=args.output_window)
    test_dataset = BatteryDataset('test_features_ecm.csv', 
                                input_window=args.input_window, 
                                output_window=args.output_window)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    input_size = train_dataset[0][0].shape[1]
    
    model = TransformerSoH(
        input_size=input_size,
        feature_size=args.feature_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        output_size=args.output_window
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.98)
    
    best_val_loss = float('inf')
    patience = 10
    no_improvement = 0
    best_model = None
    
    print(f"\nRun {run_id} - Training Transformer for Battery SoH Estimation")
    print(f"Input size: {input_size}, Feature size: {args.feature_size}")
    print(f"Input window: {args.input_window}, Output window: {args.output_window}")
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_train_loss = 0
        start_time = time.time()
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            output = model(inputs)
            loss = F.mse_loss(output, targets)
            loss.backward()
            
            if args.clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            
            optimizer.step()
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                output = model(inputs)
                loss = F.mse_loss(output, targets)
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(test_loader)
        
        elapsed = time.time() - start_time
        print(f"Epoch {epoch:3d}/{args.epochs} | Time: {elapsed:5.2f}s | "
              f"Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            no_improvement = 0
            best_model = model.state_dict().copy()
        else:
            no_improvement += 1
            if no_improvement >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
        scheduler.step()
    
    model.load_state_dict(best_model)
    model.eval()
    
    y_true, y_pred = [], []
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            output = model(inputs)
            y_true.extend(targets.cpu().numpy().flatten())
            y_pred.extend(output.cpu().numpy().flatten())
    
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    print(f"\nRun {run_id} Results:")
    print(f"MAE: {mae:.4f}, MSE: {mse:.6f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")
    
    try:
        input_shape = (input_size, args.input_window)
        efficiency_metrics = evaluate_model_efficiency(model, input_shape=input_shape, device=device)
        print(f"\nRun {run_id} Model Efficiency:")
        print(f"Parameters: {efficiency_metrics['Parameters (K)']:.2f} K")
        print(f"FLOPs: {efficiency_metrics['FLOPs (M)']:.2f} M")
        return mae, mse, rmse, r2, efficiency_metrics
    except ImportError:
        print(f"\nModel efficiency evaluation not available (model_parameters.py not found)")
        return mae, mse, rmse, r2, None

metrics_all_runs = []
all_efficiency_metrics = []

for run in range(1, args.runs + 1):
    print(f"\n=== Run {run} ===")
    result = run_once(seed=42 + run, run_id=run)
    if len(result) == 5: 
        mae, mse, rmse, r2, efficiency = result
        metrics_all_runs.append((mae, mse, rmse, r2))
        if efficiency:
            all_efficiency_metrics.append(efficiency)
    else:
        metrics_all_runs.append(result)

metrics_all_runs = np.array(metrics_all_runs)
print("\n" + "="*50)
print("FINAL RESULTS SUMMARY")
print("="*50)
for name, values in zip(['MAE', 'MSE', 'RMSE', 'R²'], metrics_all_runs.T):
    print(f"{name:4s} = {np.mean(values):.4f} ± {np.std(values):.4f}")

if all_efficiency_metrics:
    params = [m['Parameters (K)'] for m in all_efficiency_metrics]
    flops = [m['FLOPs (M)'] for m in all_efficiency_metrics]
    print("\nModel Efficiency Summary:")
    print(f"Parameters (K): {np.mean(params):.2f} ± {np.std(params):.2f}")
    print(f"FLOPs (M): {np.mean(flops):.2f} ± {np.std(flops):.2f}")

print("\nTraining completed successfully!")