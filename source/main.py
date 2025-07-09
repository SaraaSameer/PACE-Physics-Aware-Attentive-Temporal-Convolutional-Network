import argparse
import os
import random
import torch
import torch.nn
from torch.utils.data import DataLoader
from dataset import BatteryDataset
from pace import PACE
import torch.nn as nn
import numpy as np
import time
import pandas as pd  
import json
from datetime import datetime

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import joblib
import wandb
from model_parameters import evaluate_model_efficiency, fix_input_shape


def create_experiment_directory(args):
    """
    Create experiment-specific directory based on key parameters
    Format: results_new/attention_{type}_output_{window}/timestamp_YYYYMMDD_HHMMSS/
    """
    # Create base experiment identifier
    experiment_name = f"attention_{args.attention_type}_output_{args.output_window}"
    
    # Add chunk size if using chunk attention
    if args.attention_type == 'chunk':
        experiment_name += f"_chunk_{args.chunk_size}"
    
    # Add timestamp for uniqueness
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create full directory path
    experiment_dir = os.path.join(args.results_dir, experiment_name, timestamp)
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Also create subdirectories for different types of outputs
    subdirs = ['models', 'plots', 'scalers']
    for subdir in subdirs:
        os.makedirs(os.path.join(experiment_dir, subdir), exist_ok=True)
    
    print(f"Experiment directory created: {experiment_dir}")
    return experiment_dir, experiment_name

def save_experiment_config(args, experiment_dir):
    """Save experiment configuration for reproducibility"""
    config = vars(args).copy()
    config['timestamp'] = datetime.now().isoformat()
    config['experiment_dir'] = experiment_dir
    
    config_path = os.path.join(experiment_dir, 'experiment_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Experiment config saved to: {config_path}")
    return config_path

# Custom DataLoader wrapper that applies scaling to each batch
class ScaledDataLoader:
    def __init__(self, dataloader, scaler):
        self.dataloader = dataloader
        self.scaler = scaler
        
    def __iter__(self):
        for inputs, targets in self.dataloader:
            batch_size, seq_len, n_features = inputs.shape
            inputs_reshaped = inputs.reshape(-1, n_features).numpy()
            inputs_scaled = self.scaler.transform(inputs_reshaped)
            inputs_scaled = torch.FloatTensor(inputs_scaled.reshape(batch_size, seq_len, n_features))
            if torch.cuda.is_available():
                inputs_scaled = inputs_scaled.cuda()
                targets = targets.cuda()    
            yield inputs_scaled, targets
    
    def __len__(self):
        return len(self.dataloader)
    
def save_results_to_csv(y_true, y_pred, run, epoch, metrics, args, training_time_per_epoch=None, save_dir='results'):
    os.makedirs(save_dir, exist_ok=True)
    results_df = pd.DataFrame({
        'sample_index': range(len(y_true)),
        'ground_truth': y_true,
        'predictions': y_pred,
        'absolute_error': np.abs(np.array(y_true) - np.array(y_pred)),
        'squared_error': (np.array(y_true) - np.array(y_pred))**2,
        'run': run,
        'epoch': epoch
    })
    results_path = os.path.join(save_dir, f'predictions_run_{run}.csv')
    results_df.to_csv(results_path, index=False)
    print(f"Results saved to: {results_path}")
    mae, mse, rmse, r2 = metrics
    metrics_df = pd.DataFrame({
        'run': [run],
        'epoch': [epoch],
        'mae': [mae],
        'mse': [mse],
        'rmse': [rmse],
        'r2': [r2],
        'num_samples': [len(y_true)]
    })
    
    if training_time_per_epoch is not None:
        metrics_df['training_time_per_epoch'] = [training_time_per_epoch]
    
    metrics_path = os.path.join(save_dir, f'metrics_run_{run}.csv')
    metrics_df.to_csv(metrics_path, index=False)
    return results_path, metrics_path

def save_training_history(train_losses, val_losses, run, save_dir='results'):
    os.makedirs(save_dir, exist_ok=True)
    history_df = pd.DataFrame({
        'epoch': range(1, len(train_losses) + 1),
        'train_loss': train_losses,
        'val_loss': val_losses,
        'run': run
    })
    history_path = os.path.join(save_dir, f'training_history_run_{run}.csv')
    history_df.to_csv(history_path, index=False)
    print(f"Training history saved to: {history_path}")
    return history_path

def save_aggregated_results(all_predictions, test_metrics, all_training_times_per_epoch=None, save_dir='results'):
    os.makedirs(save_dir, exist_ok=True)
    all_results = []
    for run, (y_true, y_pred) in enumerate(all_predictions, 1):
        run_df = pd.DataFrame({
            'sample_index': range(len(y_true)),
            'ground_truth': y_true,
            'predictions': y_pred,
            'absolute_error': np.abs(np.array(y_true) - np.array(y_pred)),
            'squared_error': (np.array(y_true) - np.array(y_pred))**2,
            'run': run
        })
        all_results.append(run_df)
    combined_df = pd.concat(all_results, ignore_index=True)
    combined_path = os.path.join(save_dir, 'all_predictions_combined.csv')
    combined_df.to_csv(combined_path, index=False)
    summary_stats = combined_df.groupby('run').agg({
        'absolute_error': ['mean', 'std', 'min', 'max'],
        'squared_error': ['mean', 'std'],
        'ground_truth': ['mean', 'std', 'min', 'max'],
        'predictions': ['mean', 'std', 'min', 'max']
    }).round(6)
    
    summary_path = os.path.join(save_dir, 'summary_statistics.csv')
    summary_stats.to_csv(summary_path)
    
    # Save final metrics summary - FIXED: Create proper DataFrame structure
    mae_values = [m[0] for m in test_metrics]
    mse_values = [m[1] for m in test_metrics]
    rmse_values = [m[2] for m in test_metrics]
    r2_values = [m[3] for m in test_metrics]
    
    # Create the base metrics data
    metrics_data = {
        'metric': ['MAE', 'MSE', 'RMSE', 'R2'],
        'mean': [np.mean(mae_values), np.mean(mse_values), np.mean(rmse_values), np.mean(r2_values)],
        'std': [np.std(mae_values), np.std(mse_values), np.std(rmse_values), np.std(r2_values)],
        'min': [np.min(mae_values), np.min(mse_values), np.min(rmse_values), np.min(r2_values)],
        'max': [np.max(mae_values), np.max(mse_values), np.max(rmse_values), np.max(r2_values)]
    }
    
    # Add training time metrics if available
    if all_training_times_per_epoch is not None:
        metrics_data['metric'].append('Training Time (s/epoch)')
        metrics_data['mean'].append(np.mean(all_training_times_per_epoch))
        metrics_data['std'].append(np.std(all_training_times_per_epoch))
        metrics_data['min'].append(np.min(all_training_times_per_epoch))
        metrics_data['max'].append(np.max(all_training_times_per_epoch))
    
    final_metrics = pd.DataFrame(metrics_data)
    
    metrics_summary_path = os.path.join(save_dir, 'final_metrics_summary.csv')
    final_metrics.to_csv(metrics_summary_path, index=False)
    print(f"All results saved to: {combined_path}")
    print(f"Summary statistics saved to: {summary_path}")
    print(f"Final metrics summary saved to: {metrics_summary_path}")
    return combined_path, summary_path, metrics_summary_path

def export_model_to_onnx(model, input_size, features, run_num, model_dir):
    model.eval()
    dummy_input = torch.randn(1, input_size, len(features), device=next(model.parameters()).device)
    onnx_path = os.path.join(model_dir, f"TCAN_model_run{run_num}.onnx")
    torch.onnx.export(
        model,               # model being run
        dummy_input,         # model input
        onnx_path,           # where to save the model
        export_params=True,  # store the trained parameters
        opset_version=12,    # ONNX version
        input_names=['input'],
        output_names=['output']
    )
    print(f"Model exported to ONNX format at {onnx_path}")
    return onnx_path

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
    parser.add_argument('--num_channels', nargs='+', type=int, default= [32, 64, 64])  # number of features
    parser.add_argument('--kernel_size', type=int, default=3)
    parser.add_argument('--num_runs', type=int, default=3)
    parser.add_argument('--model_dir', type=str, default='models', help='Directory to save models')
    parser.add_argument('--plot_dir', type=str, default='plots', help='Directory to save plots')
    parser.add_argument('--wandb_project', type=str, default='battery_soh')
    parser.add_argument('--wandb_entity', type=str, default=None)
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--scale_data', action='store_true', help='Whether to scale input features')
    parser.add_argument('--results_dir', type=str, default='results_new', help='Base directory for results')  
    # For attention , can also add num_heads
    parser.add_argument('--attention_type', type=str, default='chunk', choices=['none', 'single', 'multi', 'chunk'],
                    help='Type of attention to use in PACE')
    parser.add_argument('--chunk_size', type=int, default=16, help='Chunk size for chunk-based attention')
    args = parser.parse_args()
    
    # Create experiment-specific directory structure
    experiment_dir, experiment_name = create_experiment_directory(args)
    
    # Update all directory paths to use experiment directory
    model_dir = os.path.join(experiment_dir, 'models')
    plot_dir = os.path.join(experiment_dir, 'plots')
    scalers_dir = os.path.join(experiment_dir, 'scalers')
    
    # Save experiment configuration
    save_experiment_config(args, experiment_dir)

    full_train_dataset = BatteryDataset(r'D:\SIT Projects\BatteryML\train_features_ecm.csv',
                                  input_window=args.input_window,
                                  output_window=args.output_window, feature_cols=None)
    test_dataset = BatteryDataset(r'D:\SIT Projects\BatteryML\test_features_ecm.csv',
                                 input_window=args.input_window,
                                 output_window=args.output_window,feature_cols=None)
    
    # Apply feature scaling if needed
    feature_scaler = None
    if args.scale_data:
        print("Applying feature scaling...")
        all_features = []
        for i in range(len(full_train_dataset)):
            features = full_train_dataset[i][0].numpy()
            seq_length, n_features = features.shape
            all_features.append(features.reshape(-1))
        all_features = np.concatenate(all_features).reshape(-1, n_features)
    
        feature_scaler = StandardScaler()
        feature_scaler.fit(all_features)
        
        # Save scaler in experiment directory
        joblib.dump(feature_scaler, os.path.join(scalers_dir, 'feature_scaler.pkl'))
        print(f"Feature scaler fitted on {len(all_features)} samples with {n_features} features")
        print(f"Scaler saved to: {scalers_dir}")
    
    num_runs = args.num_runs   # Train it for 3 times 
    test_metrics = []
    all_predictions = []
    all_training_histories = []
    # Model parameters and flops
    all_efficiency_metrics = []
    all_training_times_per_epoch = []

    for run in range(num_runs):
        print(f"\n=== Run {run+1}/{num_runs} ===")  
        if args.use_wandb:
            feature_cols = full_train_dataset.feature_cols
            label_col = full_train_dataset.label_col

            wandb.init(
                project=args.wandb_project,
                name=f"{experiment_name}_run_{run+1}",  # Add experiment name to wandb run
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
                    "label_col": label_col,
                    "attention_type": args.attention_type,
                    "chunk_size": args.chunk_size,
                    "experiment_name": experiment_name,
                    "experiment_dir": experiment_dir
                },
                reinit=True,
            )
            
        train_dataset = full_train_dataset
        val_dataset = test_dataset
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                 shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                               shuffle=False, collate_fn=collate_fn)
        
        # Apply data scaling in DataLoader if scaler is available
        if feature_scaler is not None:
            train_loader = ScaledDataLoader(train_loader, feature_scaler)
            val_loader = ScaledDataLoader(val_loader, feature_scaler)
    
        # Reinitialize model each run
        model = PACE(input_size=len(full_train_dataset.feature_cols),
                   output_size=1,
                   num_channels=args.num_channels,
                   kernel_size=args.kernel_size, 
                   attention= args.attention_type,
                   chunk_size = args.chunk_size, 
                   output_window = args.output_window)
        if torch.cuda.is_available():
            model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        criterion = nn.MSELoss()
        
        if args.use_wandb:
            wandb.watch(model, criterion, log="all", log_freq=10)
        
        best_val_loss = float('inf')
        patience = 20  # Number of epochs to wait before stopping
        no_improvement = 0
        best_model_weights = None

        train_losses = []
        val_losses = []
        epoch_times = []  
        
        for epoch in range(args.epochs):
            epoch_start_time = time.time()
            model.train()
            train_loss = 0
            for batch_idx, (inputs, targets) in enumerate(train_loader):
              if torch.cuda.is_available():
                  inputs, targets = inputs.cuda(), targets.cuda()
              optimizer.zero_grad()
              outputs = model(inputs)
              loss = criterion(outputs.squeeze(), targets.squeeze()) 
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
                  loss = criterion(outputs.squeeze(), targets.squeeze())  
                  val_loss += loss.item()
            
            epoch_time = time.time() - epoch_start_time
            epoch_times.append(epoch_time)
            avg_train = train_loss/len(train_loader)
            avg_val = val_loss/len(val_loader)
            train_losses.append(avg_train)
            val_losses.append(avg_val)

            if avg_val < best_val_loss:
                best_val_loss = avg_val
                no_improvement = 0
                # Save best model weights
                best_model_weights = model.state_dict().copy()
            else:
                no_improvement += 1
            
            print(f"Epoch {epoch+1:03d} | Train Loss: {avg_train:.4f}, Val Loss: {avg_val:.4f} | Patience: {no_improvement}/{patience} | Time: {epoch_time:.2f}s")
            if args.use_wandb:
                wandb.log({
                    "epoch": epoch+1,
                    "train_loss": avg_train,
                    "val_loss": avg_val,
                    "patience": no_improvement,
                    "time_per_epoch": epoch_time
                })

            if no_improvement >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        avg_time_per_epoch = np.mean(epoch_times)
        all_training_times_per_epoch.append(avg_time_per_epoch)

        history_path = save_training_history(train_losses, val_losses, run+1, experiment_dir)
        all_training_histories.append((train_losses, val_losses))

        # Load best model weights before testing
        model.load_state_dict(best_model_weights)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        input_shape = (len(full_train_dataset.feature_cols), args.input_window)

        corrected_input_shape = fix_input_shape(input_shape)
        efficiency_metrics = evaluate_model_efficiency(model, corrected_input_shape, device)
        all_efficiency_metrics.append(efficiency_metrics)
        
        print(f"\nRun {run+1} Model Efficiency:")
        print(f"Parameters: {efficiency_metrics['Parameters (K)']} K")
        print(f"FLOPs: {efficiency_metrics['FLOPs (M)']} M")
      
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                               shuffle=False, collate_fn=collate_fn)
        
        # Apply scaling to test data if needed
        if feature_scaler is not None:
            test_loader = ScaledDataLoader(test_loader, feature_scaler)
            
        test_mae, test_mse, test_rmse, test_r2, y_true, y_pred = evaluate_metrics(model, test_loader)
        test_metrics.append((test_mae, test_mse, test_rmse, test_r2))
        all_predictions.append((y_true, y_pred))

        final_epoch = len(train_losses)
        metrics_tuple = (test_mae, test_mse, test_rmse, test_r2)
        results_path, metrics_path = save_results_to_csv(
            y_true, y_pred, run+1, final_epoch, metrics_tuple, args, 
            training_time_per_epoch=avg_time_per_epoch, save_dir=experiment_dir
        )

        plot_predictions(y_true, y_pred, run+1, save_path=plot_dir)

        print(f"\nRun {run+1} Test Metrics:")
        print(f"MAE: {test_mae:.6f}  MSE: {test_mse:.6f}")
        print(f"RMSE: {test_rmse:.6f}  R²: {test_r2:.6f}")

        if args.use_wandb:
            wandb.log({
                "test_mae": test_mae,
                "test_mse": test_mse,
                "test_rmse": test_rmse,
                "test_r2": test_r2,
                "model_parameters_k": efficiency_metrics['Parameters (K)'],
                "model_flops_m": efficiency_metrics['FLOPs (M)'],
                "avg_time_per_epoch": avg_time_per_epoch,
                "flops_per_param": efficiency_metrics['FLOPs (M)'] / efficiency_metrics['Parameters (K)'],
                "performance_efficiency": test_r2 / (efficiency_metrics['Parameters (K)'] / 1000),  # R2 per million params
            })
            wandb.log({
                "SOH_TimeSeries": wandb.Image(f'{plot_dir}/soh_timeseries_run{run+1}.png'),
                "SOH_Correlation": wandb.Image(f'{plot_dir}/soh_correlation_run{run+1}.png')
            })
        
        feature_cols = full_train_dataset.feature_cols
        onnx_path = export_model_to_onnx(model, args.input_window, feature_cols, run+1, model_dir)
        
        if args.use_wandb:
            # Use artifact logging instead of symlinks (works better on Windows)
            artifact = wandb.Artifact(f'model_run{run+1}', type='model')
            artifact.add_file(onnx_path)
            wandb.log_artifact(artifact)
            wandb.finish()
    
    save_aggregated_results(all_predictions, test_metrics, all_training_times_per_epoch, experiment_dir)
    
    # Final report
    mae_values = [m[0] for m in test_metrics]
    mse_values = [m[1] for m in test_metrics]
    rmse_values = [m[2] for m in test_metrics]
    r2_values = [m[3] for m in test_metrics]
    final_efficiency = all_efficiency_metrics[0]
    final_avg_time_per_epoch = np.mean(all_training_times_per_epoch)

    print(f"\n=== EXPERIMENT COMPLETED: {experiment_name} ===")
    print(f"Results saved to: {experiment_dir}")
    print("\nFinal Test Metrics:")
    print(f"MAE  = {np.mean(mae_values):.4f} ± {np.std(mae_values):.4f}")
    print(f"MSE  = {np.mean(mse_values):.4f} ± {np.std(mse_values):.4f}")
    print(f"RMSE = {np.mean(rmse_values):.4f} ± {np.std(rmse_values):.4f}")
    print(f"R²   = {np.mean(r2_values):.4f} ± {np.std(r2_values):.4f}")
    print(f"Parameters: {final_efficiency['Parameters (K)']:.2f} K")
    print(f"FLOPs: {final_efficiency['FLOPs (M)']:.2f} M")
    print(f"Training Time: {final_avg_time_per_epoch:.2f} s/epoch")

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
    plt.scatter(y_true, y_pred, alpha=0.6, label='Predictions')
    plt.title(f'Actual vs Predicted SOH Values (Run {run})')
    plt.xlabel('Actual SOH')
    plt.ylabel('Predicted SOH')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.legend()
    
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
            if targets.device != device:
                targets = targets.to(device) 
            outputs = model(inputs).squeeze()
            targets_cpu = targets.cpu().numpy()
            outputs_cpu = outputs.cpu().numpy()
            y_true.extend(targets_cpu.flatten())
            y_pred.extend(outputs_cpu.flatten())
    
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return mae, mse, rmse, r2, y_true, y_pred

if __name__ == '__main__':
    main()