import numpy as np
import pandas as pd
import onnxruntime as ort
from ecm_simulatedvoltage import Thevenien_1RC

# === 1. Load data with only raw features ===
df = pd.read_csv('model_test_file - base features.csv')
df = df[df['cell_id'] == 'b1c0'].sort_values(by='cycle').reset_index(drop=True)

# Required base features
required_cols = ['cycle', 'V_avg', 'I_avg', 'Time_Stamp', 'Tavg']
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Missing column: {col}")

# === 2. Slice last 100 rows ===
window_size = 100
window_df = df.tail(window_size).copy()
cycle = window_df['cycle'].values
V_avg = window_df['V_avg'].values
I_avg = window_df['I_avg'].values
time = window_df['Time_Stamp'].values

# === 3. Estimate ECM ===
ecm = Thevenien_1RC()
ecm_params = ecm.extract_1rc_features(time, V_avg, I_avg)

# Add ECM as columns
for k, v in ecm_params.items():
    window_df[k] = v

# === 4. Construct final feature array ===
feature_cols = ['cycle', 'V_avg', 'I_avg', 'Time_Stamp', 'R0', 'R1', 'C1', 'Uocv', 'Tavg']
X = window_df[feature_cols].values.astype(np.float32)
X = X.reshape(1, window_size, len(feature_cols))  # shape: (1, 100, 8)

# === 5. Load ONNX model and run inference ===
session = ort.InferenceSession('source\TCN_Attention\TCAN_model_run1.onnx', providers=['CPUExecutionProvider'])
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
pred_soh = session.run([output_name], {input_name: X})[0].flatten()

# === 6. Output ===
print("🔋 Predicted 30-step SoH:")
print(pred_soh)
np.savetxt("predicted_soh_output.csv", pred_soh, delimiter=",", header="Predicted_SoH", comments='')

