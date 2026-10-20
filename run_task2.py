import os
import numpy as np
import torch
import preprocessing as prep
from task2_ssl import CSIEncoder, train_ssl, evaluate_linear_probe

splits = prep.load_cached_datasets()
a_dev_norm, a_val_norm = prep.normalize_windows(splits['room_a_dev']['windows'], splits['room_a_val']['windows'], mode='static_removal')
b_dev_norm, b_val_norm = prep.normalize_windows(splits['room_b_dev']['windows'], splits['room_b_val']['windows'], mode='static_removal')

all_dev = np.vstack([a_dev_norm, b_dev_norm])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}, Dev windows: {all_dev.shape}")

model_path = r"D:\ubi\p2\models\ssl_encoder.pth"
encoder = CSIEncoder(in_channels=256, hidden_dim=128, feature_dim=128, proj_dim=64).to(device)
encoder = train_ssl(encoder, all_dev, epochs=15, batch_size=128, lr=1e-3, device=device, save_path=model_path)

res_a = evaluate_linear_probe(encoder, a_dev_norm, splits['room_a_dev']['labels'], a_val_norm, splits['room_a_val']['labels'], device=device)
res_b = evaluate_linear_probe(encoder, b_dev_norm, splits['room_b_dev']['labels'], b_val_norm, splits['room_b_val']['labels'], device=device)

# Combined probe evaluation
comb_train_w = np.vstack([a_dev_norm, b_dev_norm])
comb_train_y = np.concatenate([splits['room_a_dev']['labels'], splits['room_b_dev']['labels']])
comb_val_w = np.vstack([a_val_norm, b_val_norm])
comb_val_y = np.concatenate([splits['room_a_val']['labels'], splits['room_b_val']['labels']])
res_comb = evaluate_linear_probe(encoder, comb_train_w, comb_train_y, comb_val_w, comb_val_y, device=device)

print(f"[*] Task 2 Linear-Probe Macro-F1 (Room A):  {res_a['macro_f1']:.4f}")
print(f"[*] Task 2 Linear-Probe Macro-F1 (Room B):  {res_b['macro_f1']:.4f}")
print(f"[*] Task 2 Linear-Probe Macro-F1 (Combined): {res_comb['macro_f1']:.4f}")
