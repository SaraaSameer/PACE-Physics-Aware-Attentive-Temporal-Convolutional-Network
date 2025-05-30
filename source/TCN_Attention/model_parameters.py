import torch
import time
from ptflops import get_model_complexity_info

def evaluate_model_efficiency(model, input_shape=(8, 100), device='cuda'):
    """
    Compute Parameters (K), FLOPs (M), and GPU Inference Time (ms)
    
    Args:
        model (nn.Module): Model to evaluate
        input_shape (tuple): Input shape excluding batch (e.g., (features, sequence_length))
        device (str): 'cuda' or 'cpu'
    
    Returns:
        dict: { 'Parameters (K)', 'FLOPs (M)', 'Inference Time (ms)' }
    """
    model = model.to(device)
    model.eval()

    # Calculate Params and FLOPs with error handling
    try:
        macs, params = get_model_complexity_info(model, input_shape, as_strings=False,
                                                 print_per_layer_stat=False, verbose=False)
        
        # Handle case where ptflops returns None
        if params is None:
            print("Warning: ptflops returned None for parameters. Calculating manually...")
            params = sum(p.numel() for p in model.parameters())
        
        if macs is None:
            print("Warning: ptflops returned None for FLOPs. Setting to 0...")
            macs = 0
            
        params_k = round(params / 1e3, 2)  # thousands
        flops_m = round(macs / 1e6, 2)     # millions
        
    except Exception as e:
        print(f"Error calculating FLOPs with ptflops: {e}")
        print("Falling back to manual parameter calculation...")
        
        # Manual parameter calculation
        params = sum(p.numel() for p in model.parameters())
        params_k = round(params / 1e3, 2)
        flops_m = "N/A"  # Can't calculate FLOPs due to dimension mismatch

    # Inference time
    dummy_input = torch.randn(1, *input_shape).to(device)
    
    # Handle potential inference issues
    try:
        with torch.no_grad():
            # Test if model can actually process the input
            test_output = model(dummy_input)
            
            # Time the inference
            if device == 'cuda':
                torch.cuda.synchronize()
            start = time.time()
            for _ in range(10):
                _ = model(dummy_input)
            if device == 'cuda':
                torch.cuda.synchronize()
            end = time.time()
        avg_time_ms = round((end - start) / 10 * 1000, 2)  # ms
        
    except Exception as e:
        print(f"Error during inference timing: {e}")
        avg_time_ms = "N/A"

    return {
        "Parameters (K)": params_k,
        "FLOPs (M)": flops_m,
        "Inference Time (ms)": avg_time_ms
    }

def fix_input_shape_for_tcn(input_shape):
    """
    Helper function to fix input shape for TCN models
    TCN expects (channels, sequence_length) but your data might be (sequence_length, channels)
    
    Args:
        input_shape (tuple): Original input shape
    
    Returns:
        tuple: Corrected input shape for TCN
    """
    if len(input_shape) == 2:
        # If input is (100, 9), TCN likely expects (9, 100)
        if input_shape[0] > input_shape[1]:
            print(f"Swapping input shape from {input_shape} to {input_shape[::-1]} for TCN")
            return input_shape[::-1]
    return input_shape