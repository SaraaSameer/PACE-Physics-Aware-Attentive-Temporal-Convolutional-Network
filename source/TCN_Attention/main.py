import argparse
import os
import random
import torch
import torch.nn
from torch.utils.data import DataLoader
from dataset import BatteryDataset
from tcn import TCN
import torch.nn as nn
import numpy as np
import time
import wandb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler



def collate_fn(batch):
    inputs = torch.stack([item[0] for item in batch])
    targets = torch.stack([item[1] for item in batch])
    return inputs, targets

def main():
    # For deterministic behaviour
    torch.backends.cudnn.deterministic = True
    random.seed(hash("setting random seeds") % 2**32 - 1)
    np.random.seed(hash("improves reproducibility") % 2**32 - 1)
    torch.manual_seed(hash("by removing stochasticity") % 2**32 - 1)
    torch.cuda.manual_seed_all(hash("so runs are repeatable") % 2**32 - 1)

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='train', choices=['train', 'test'])
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--input_window', type=int, default=100)
    parser.add_argument('--output_window', type=int, default=30)
    parser.add_argument('--num_channels', nargs='+', type=int, default=[9, 9, 9])  # number of features
    parser.add_argument('--kernel_size', type=int, default=3)
    parser.add_argument('--model_dir', type=str, default='models', help='Directory to save models')
    parser.add_argument('--plot_dir', type=str, default='plots', help='Directory to save plots')
    parser.add_argument('--wandb_project', type=str, default='battery_soh')
    parser.add_argument('--wandb_entity', type=str, default=None)
    parser.add_argument('--use_wandb', action='store_true')
    args = parser.parse_args()
    
    # Create directory if it doesn't exist
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.plot_dir, exist_ok=True)

    full_train_dataset = BatteryDataset(r'D:\SIT Projects\BatteryML\train_features_ecm.csv',
                                  input_window=args.input_window,
                                  output_window=args.output_window, feature_cols=None)
    test_dataset = BatteryDataset(r'D:\SIT Projects\BatteryML\test_features_ecm.csv',
                                 input_window=args.input_window,
                                 output_window=args.output_window,feature_cols=None)
    num_runs = 1   # Train it for 3 times 
    test_metrics = []
    all_predictions = []

    for run in range(num_runs):
        print(f"\n=== Run {run+1}/{num_runs} ===")  
        if args.use_wandb:

            feature_cols = full_train_dataset.feature_cols
            label_col = full_train_dataset.label_col

            wandb.init(
                project=args.wandb_project,
                config={
                    "epochs": args.epochs,
                    "batch_size": args.batch_size,
                    "learning_rate": args.lr,
                    "input_window": args.input_window,
                    "output_window": args.output_window,
                    "num_channels": args.num_channels,
                    "kernel_size": args.kernel_size,
                    "run": run+1,
                    "total_runs": num_runs,
                    "features": feature_cols,
                    "num_features": len(feature_cols),
                    "label_col": label_col
                },
                reinit=True,
            )
        train_size = int(0.8 * len(full_train_dataset))
        train_dataset = full_train_dataset
        val_dataset = test_dataset
        # train_dataset, val_dataset = torch.utils.data.random_split(
        #     full_train_dataset, [train_size, len(full_train_dataset)-train_size])
    
        # Reinitialize model each run
        model = TCN(input_size=len(full_train_dataset.feature_cols),
                   output_size=1,
                   num_channels=args.num_channels,
                   kernel_size=args.kernel_size, attention= True)
        if torch.cuda.is_available():
            model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        criterion = nn.MSELoss()
        
        if args.use_wandb:
            wandb.watch(model, criterion, log="all", log_freq=10)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                 shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                               shuffle=False, collate_fn=collate_fn)
        
        best_val_loss = float('inf')
        patience = 20  # Number of epochs to wait before stopping
        no_improvement = 0
        best_model_weights = None

        for epoch in range(args.epochs):
            model.train()
            train_loss = 0
            for batch_idx, (inputs, targets) in enumerate(train_loader):
              if torch.cuda.is_available():
                  inputs, targets = inputs.cuda(), targets.cuda()
              optimizer.zero_grad()
              outputs = model(inputs)
              loss = criterion(outputs.squeeze(), targets) 
              loss.backward()
              optimizer.step()
              train_loss += loss.item()

            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():  # Context manager for
                for inputs, targets in val_loader:
                  if torch.cuda.is_available():   # Maintained consistent GPU handling for validation data
                      inputs, targets = inputs.cuda(), targets.cuda()
                  outputs = model(inputs)
                  loss = criterion(outputs.squeeze(), targets)  
                  val_loss += loss.item()

            avg_train = train_loss/len(train_loader)
            avg_val = val_loss/len(val_loader)

            if avg_val < best_val_loss:
                best_val_loss = avg_val
                no_improvement = 0
                # Save best model weights
                best_model_weights = model.state_dict().copy()
            else:
                no_improvement += 1
            
            print(f"Epoch {epoch+1:03d} | Train Loss: {avg_train:.4f}, Val Loss: {avg_val:.4f} | Patience: {no_improvement}/{patience}")
            if args.use_wandb:
                wandb.log({
                    "epoch": epoch+1,
                    "train_loss": avg_train,
                    "val_loss": avg_val,
                    "patience": no_improvement
                })

            if no_improvement >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        # Load best model weights before testing
        model.load_state_dict(best_model_weights)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                                shuffle=False, collate_fn=collate_fn)
        test_mae, test_mse, test_rmse, test_r2, y_true, y_pred = evaluate_metrics(model, test_loader)
        test_metrics.append((test_mae, test_mse, test_rmse, test_r2))
        all_predictions.append((y_true, y_pred))

        plot_predictions(y_true, y_pred, run+1, save_path=args.plot_dir)

        print(f"\nRun {run+1} Test Metrics:")
        print(f"MAE: {test_mae:.6f}  MSE: {test_mse:.6f}")
        print(f"RMSE: {test_rmse:.6f}  R²: {test_r2:.6f}")

    if args.use_wandb:
            wandb.log({
                "test_mae": test_mae,
                "test_mse": test_mse,
                "test_rmse": test_rmse,
                "test_r2": test_r2,
            })
            wandb.log({
                "SOH_TimeSeries": wandb.Image(f'{args.plot_dir}/soh_timeseries_run{run+1}.png'),
                "SOH_Correlation": wandb.Image(f'{args.plot_dir}/soh_correlation_run{run+1}.png')
            })
    wandb.finish()

    # Final report
    mae_values = [m[0] for m in test_metrics]
    mse_values = [m[1] for m in test_metrics]
    rmse_values = [m[2] for m in test_metrics]
    r2_values = [m[3] for m in test_metrics]
    print("\n Final Test Metrics: ")
    print(f"MAE  = {np.mean(mae_values):.4f} ± {np.std(mae_values):.4f}")
    print(f"MSE  = {np.mean(mse_values):.4f} ± {np.std(mse_values):.4f}")
    print(f"RMSE = {np.mean(rmse_values):.4f} ± {np.std(rmse_values):.4f}")
    print(f"R²   = {np.mean(r2_values):.4f} ± {np.std(r2_values):.4f}")

def plot_predictions(y_true, y_pred, run=1, save_path='plots'):
    os.makedirs(save_path, exist_ok=True)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Figure 1: Time series plot (prediction vs actual)
    plt.figure(figsize=(12, 6))
    plt.plot(y_true, label='Actual SOH', marker='o', markersize=3, linestyle='-', alpha=0.7)
    plt.plot(y_pred, label='Predicted SOH', marker='x', markersize=3, linestyle='-', alpha=0.7)
    plt.title(f'Battery SOH: Actual vs Predicted (Run {run})')
    plt.xlabel('Sample Index')
    plt.ylabel('State of Health (SOH)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_path}/soh_timeseries_run{run}.png', dpi=300)
    
    # Figure 2: Scatter plot with perfect prediction line
    plt.figure(figsize=(8, 8))
    
    # Calculate min and max for axis limits
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    
    # Plot the perfect prediction line
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
    
    # Scatter plot
    plt.scatter(y_true, y_pred, alpha=0.6, label='Predictions')
    
    # Labels and styling
    plt.title(f'Actual vs Predicted SOH Values (Run {run})')
    plt.xlabel('Actual SOH')
    plt.ylabel('Predicted SOH')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.legend()
    
    # Calculate and display R²
    r2 = r2_score(y_true, y_pred)
    plt.annotate(f'R² = {r2:.4f}', xy=(0.05, 0.95), xycoords='axes fraction', 
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/soh_correlation_run{run}.png', dpi=300)
    plt.close('all')

def evaluate_metrics(model, loader):
    model.eval()
    y_true, y_pred = [], []
    device = next(model.parameters()).device 
    
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            outputs = model(inputs).squeeze().cpu().numpy()
            y_true.extend(targets.numpy().flatten())
            y_pred.extend(outputs.flatten())
    
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return mae, mse, rmse, r2, y_true, y_pred

if __name__ == '__main__':
    main()