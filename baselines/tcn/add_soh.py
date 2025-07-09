import torch
import argparse
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import BatteryDataset
from model import TCN
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from model_parameters import evaluate_model_efficiency

parser = argparse.ArgumentParser(description='Battery SoH Estimation using TCN')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--input_window', type=int, default=100)
parser.add_argument('--output_window', type=int, default=50)
parser.add_argument('--kernel_size', type=int, default=3)
parser.add_argument('--nhid', type=int, default=32)
parser.add_argument('--levels', type=int, default=3)
parser.add_argument('--clip', type=float, default=-1)
parser.add_argument('--cuda', action='store_true')
parser.add_argument('--runs', type=int, default=2)
args = parser.parse_args()

torch.manual_seed(42)
device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")

metrics_all_runs = []
all_efficiency_metrics = []

def run_once(seed, run_id):
    torch.manual_seed(seed)

    train_dataset = BatteryDataset('train_features_ecm.csv', input_window=args.input_window, output_window=args.output_window)
    test_dataset = BatteryDataset('test_features_ecm.csv', input_window=args.input_window, output_window=args.output_window)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    input_size = train_dataset[0][0].shape[1]
    output_size = args.output_window
    channel_sizes = [args.nhid] * args.levels

    model = TCN(input_size=input_size, output_size=output_size, num_channels=channel_sizes,
                kernel_size=args.kernel_size, dropout=0.2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_val_loss = float('inf')
    patience = 10
    no_improvement = 0
    best_model = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_train_loss = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            inputs = inputs.permute(0, 2, 1)
            optimizer.zero_grad()
            output = model(inputs).squeeze()
            loss = F.mse_loss(output, targets.squeeze())
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
                inputs = inputs.permute(0, 2, 1)
                output = model(inputs).squeeze()
                loss = F.mse_loss(output, targets.squeeze())
                total_val_loss += loss.item()
        avg_val_loss = total_val_loss / len(test_loader)
        print(f"Epoch {epoch}/{args.epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            no_improvement = 0
            best_model = model.state_dict()
        else:
            no_improvement += 1
            if no_improvement >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_model)
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            inputs = inputs.permute(0, 2, 1)
            output = model(inputs).squeeze()
            y_true.extend(targets.cpu().numpy().flatten())
            y_pred.extend(output.cpu().numpy().flatten())

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    input_shape = (input_size, args.input_window)
    efficiency_metrics = evaluate_model_efficiency(model, input_shape=input_shape, device=device)
    all_efficiency_metrics.append(efficiency_metrics)

    print(f"\nRun {run_id} Model Efficiency:")
    print(f"Parameters: {efficiency_metrics['Parameters (K)']:.2f} K")
    print(f"FLOPs: {efficiency_metrics['FLOPs (M)']:.2f} M")

    return mae, mse, rmse, r2

# Run multiple times
for run in range(1, args.runs + 1):
    print(f"\n=== Run {run} ===")
    metrics = run_once(seed=42 + run, run_id=run)
    metrics_all_runs.append(metrics)

# Report final results
metrics_all_runs = np.array(metrics_all_runs)
print("\nFinal Results:")
for name, values in zip(['MAE', 'MSE', 'RMSE', 'R²'], metrics_all_runs.T):
    print(f"{name}  = {np.mean(values):.4f} ± {np.std(values):.4f}")

params = [m['Parameters (K)'] for m in all_efficiency_metrics]
flops = [m['FLOPs (M)'] for m in all_efficiency_metrics]
print("\nModel Efficiency Summary:")
print(f"Parameters (K): {np.mean(params):.2f} ± {np.std(params):.2f}")
print(f"FLOPs (M): {np.mean(flops):.2f} ± {np.std(flops):.2f}")
