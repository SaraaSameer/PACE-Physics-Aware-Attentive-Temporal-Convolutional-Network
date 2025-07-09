import torch
import torch.nn as nn
from torchprofile import profile_macs

def calculate_flops_alternative_methods(model, input_shape, device='cuda'):
    model = model.to(device)
    model.eval()
    results = {}

    try:
        dummy_input = torch.randn(1, *input_shape).to(device)
        macs = profile_macs(model, dummy_input)
        flops_m = round(macs / 1e6, 2)
        results['torchprofile'] = flops_m
        print(f"TorchProfile FLOPs: {flops_m} M")
    except Exception as e:
        print(f"TorchProfile failed: {e}")
        results['torchprofile'] = "Failed"
    
    try:
        from fvcore.nn import FlopCountMode, flop_count
        dummy_input = torch.randn(1, *input_shape).to(device)
        flops_dict, _ = flop_count(model, (dummy_input,), supported_ops=None)
        total_flops = sum(flops_dict.values())
        flops_m = round(total_flops / 1e6, 2)
        results['fvcore'] = flops_m
        print(f"FVCore FLOPs: {flops_m} M")
    except ImportError:
        print("FVCore not installed. Install with: pip install fvcore")
        results['fvcore'] = "Not installed"
    except Exception as e:
        print(f"FVCore failed: {e}")
        results['fvcore'] = "Failed"
    
    try:
        flops_manual = calculate_tcn_flops_manual(model, input_shape)
        results['manual'] = flops_manual
        print(f"Manual TCN FLOPs: {flops_manual} M")
    except Exception as e:
        print(f"Manual calculation failed: {e}")
        results['manual'] = "Failed"
    
    try:
        from ptflops import get_model_complexity_info
        # Try swapping dimensions for TCN
        corrected_shape = (input_shape[1], input_shape[0]) if len(input_shape) == 2 else input_shape
        print(f"Trying ptflops with corrected shape: {corrected_shape}")
        
        macs, params = get_model_complexity_info(model, corrected_shape, as_strings=False,
                                                 print_per_layer_stat=False, verbose=False)
        if macs is not None:
            flops_m = round(macs / 1e6, 2)
            results['ptflops_corrected'] = flops_m
            print(f"PTFlops (corrected) FLOPs: {flops_m} M")
        else:
            results['ptflops_corrected'] = "Failed"
    except Exception as e:
        print(f"PTFlops corrected failed: {e}")
        results['ptflops_corrected'] = "Failed"
    
    return results

def calculate_tcn_flops_manual(model, input_shape):
    total_flops = 0
    sequence_length = max(input_shape)  
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv1d):
            in_channels = module.in_channels
            out_channels = module.out_channels
            kernel_size = module.kernel_size[0]
            
            output_length = sequence_length  
            
            conv_flops = out_channels * in_channels * kernel_size * output_length
            total_flops += conv_flops
            
            print(f"Layer {name}: {conv_flops:,} FLOPs")
    
    return round(total_flops / 1e6, 2)

def evaluate_model_efficiency(model, input_shape, device='cuda'):
  
    print("=" * 50)
    print("MODEL EFFICIENCY ANALYSIS")
    print("=" * 50)
  
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)  
    print(f"Total Parameters: {total_params:,} ({total_params/1e3:.2f}K)")
    print(f"Trainable Parameters: {trainable_params:,} ({trainable_params/1e3:.2f}K)")
    print("\nFLOPs Calculation (Multiple Methods):")
    flops_results = calculate_flops_alternative_methods(model, input_shape, device)
 
    best_flops = None
    for method, result in flops_results.items():
        if isinstance(result, (int, float)) and result > 0:
            best_flops = result
            print(f"✓ Using {method} result: {result} M FLOPs")
            break
    
    if best_flops is None:
        print("⚠ All FLOP calculations failed - using manual approximation")
        best_flops = "Manual calc needed"
    
    return {
        'Parameters (K)': round(total_params / 1e3, 2),
        'FLOPs (M)': best_flops,
        'FLOP Methods': flops_results
    }

def fix_input_shape_for_tcn(input_shape):
    if len(input_shape) == 2:
        if input_shape[0] > input_shape[1]:
            print(f"Swapping input shape from {input_shape} to {input_shape[::-1]} for TCN")
            return input_shape[::-1]
    return input_shape