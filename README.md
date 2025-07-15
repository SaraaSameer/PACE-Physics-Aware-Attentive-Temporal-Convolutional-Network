# Physics-Aware Attentive Temporal Convolutional Network for Battery Health Estimation

* Sara Sameer, Wei Zhang, Lou Xin, Gao Yulin

A lightweight deep learning framework for accurate and efficient battery State-of-Health (SoH) monitoring. PACE combines temporal convolutional networks with physics-informed features from equivalent circuit models and chunked attention mechanisms to achieve superior performance while maintaining computational efficiency.

## Overview of the Architecture
![Image](https://github.com/user-attachments/assets/7eb5a6ca-5fbd-4ca4-ac06-3466183951c0)

## Results
| Model           | #Params ↓ (×10³) | FLOPs ↓ (×10⁶) | RMSE ↓ (1-Cycle) | MAE ↓ (1-Cycle) | η ↑ (1-Cycle) | RMSE ↓ (30-Cycle) | MAE ↓ (30-Cycle) | η ↑ (30-Cycle) | RMSE ↓ (50-Cycle) | MAE ↓ (50-Cycle) | η ↑ (50-Cycle) |
|----------------|------------------|----------------|------------------|------------------|--------------|--------------------|------------------|----------------|--------------------|------------------|----------------|
| Transformer     | 2559.5           | 133.2          | 0.014            | 0.009            | 27.9         | 0.016              | 0.010            | 24.4           | 0.017              | 0.009            | 23.0           |
| TCN             | 47.0             | 1.7            | 0.036            | 0.022            | 580.2        | 0.053              | 0.039            | 401.8          | 0.057              | 0.042            | 373.6          |
| BCA             | 173.5            | 5.0            | 0.034            | 0.018            | 169.5        | 0.037              | 0.020            | 155.8          | 0.038              | 0.020            | 150.5          |
| **Pace (ours)** | 70.9             | 5.1            | 0.023            | 0.010            | **613.4**    | 0.033              | 0.014            | **427.5**      | 0.035              | 0.015            | **403.1**      |

## Get Started
Install Python 3.6, PyTorch 1.9.0. <br/>
`python main.py --mode train --epochs 100 --batch_size 32 --lr 1e-3 --input_window 100 --output_window 30 --num_channels 32 64 64 --kernel_size 3 --num_runs 3  --model_dir models --wandb_project battery_soh --use_wandb --scale_data --attention_type multi --chunk_size 16`
## Citation
If you find this repo useful, please cite our paper
