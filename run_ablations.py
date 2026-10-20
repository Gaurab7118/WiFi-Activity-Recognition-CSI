"""
run_ablations.py - Comprehensive Ablation Studies for CS60055 Hackathon Challenge 2
1. Preprocessing Normalization Ablation (Raw vs Z-Score vs Static Removal)
2. Representation Ablation (Raw features vs Random Init vs SSL Pretrained)
3. Domain Alignment & Metric Ablation (Cosine vs Euclidean, Align vs No-Align)
"""

import os
import json
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import preprocessing as prep
from task1_routing import RoomRouter
from task2_ssl import CSIEncoder, extract_embeddings
from task3_adaptation import run_cross_room_transfer

def run_normalization_ablation(splits, device):
    print("\n--- Running Normalization Ablation ---")
    modes = ['none', 'zscore', 'static_removal']
    results = {}
    
    for mode in modes:
        a_dev, a_val = prep.normalize_windows(splits['room_a_dev']['windows'], splits['room_a_val']['windows'], mode=mode)
        b_dev, b_val = prep.normalize_windows(splits['room_b_dev']['windows'], splits['room_b_val']['windows'], mode=mode)
        
        # 1. Task 1 Room Accuracy
        router = RoomRouter(model_type='logistic')
        router.fit(a_dev, b_dev)
        t1_res = router.evaluate(a_val, b_val)
        
        # 2. Simple Activity Probe (Mean features across time)
        feat_a_dev = np.mean(a_dev, axis=1)
        feat_a_val = np.mean(a_val, axis=1)
        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(feat_a_dev, splits['room_a_dev']['labels'])
        preds_a = clf.predict(feat_a_val)
        f1_a = f1_score(splits['room_a_val']['labels'], preds_a, average='macro', zero_division=0)
        
        results[mode] = {
            'room_accuracy': float(t1_res['accuracy']),
            'activity_f1_room_a': float(f1_a)
        }
        print(f"  Mode '{mode:14s}': Room Acc = {t1_res['accuracy']*100:.2f}%, Activity Macro-F1 = {f1_a:.4f}")
        
    return results

def run_representation_ablation(splits, ssl_encoder, device):
    print("\n--- Running Representation Quality Ablation ---")
    a_dev_norm, a_val_norm = prep.normalize_windows(splits['room_a_dev']['windows'], splits['room_a_val']['windows'], mode='static_removal')
    b_dev_norm, b_val_norm = prep.normalize_windows(splits['room_b_dev']['windows'], splits['room_b_val']['windows'], mode='static_removal')
    
    y_a_train = splits['room_a_dev']['labels']
    y_a_val = splits['room_a_val']['labels']
    
    results = {}
    
    # 1. Raw Temporal Average Features (256-dim)
    feat_train_raw = np.mean(a_dev_norm, axis=1)
    feat_val_raw = np.mean(a_val_norm, axis=1)
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(feat_train_raw, y_a_train)
    f1_raw = f1_score(y_a_val, clf.predict(feat_val_raw), average='macro', zero_division=0)
    results['raw_csi'] = float(f1_raw)
    print(f"  1. Raw CSI Baseline Macro-F1:       {f1_raw:.4f}")
    
    # 2. Randomly Initialized Untrained Encoder
    random_encoder = CSIEncoder(in_channels=256, hidden_dim=128, feature_dim=128, proj_dim=64).to(device)
    random_encoder.eval()
    feat_train_rnd = extract_embeddings(random_encoder, a_dev_norm, device=device)
    feat_val_rnd = extract_embeddings(random_encoder, a_val_norm, device=device)
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(feat_train_rnd, y_a_train)
    f1_rnd = f1_score(y_a_val, clf.predict(feat_val_rnd), average='macro', zero_division=0)
    results['random_untrained'] = float(f1_rnd)
    print(f"  2. Random Untrained ConvNet Macro-F1: {f1_rnd:.4f}")
    
    # 3. SSL Pretrained Encoder
    feat_train_ssl = extract_embeddings(ssl_encoder, a_dev_norm, device=device)
    feat_val_ssl = extract_embeddings(ssl_encoder, a_val_norm, device=device)
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(feat_train_ssl, y_a_train)
    f1_ssl = f1_score(y_a_val, clf.predict(feat_val_ssl), average='macro', zero_division=0)
    results['ssl_pretrained'] = float(f1_ssl)
    print(f"  3. SSL Pre-trained Encoder Macro-F1:  {f1_ssl:.4f}")
    
    return results

def run_alignment_and_metric_ablation(splits, ssl_encoder, device):
    print("\n--- Running Domain Alignment & Distance Metric Ablation (Task 3, K=5) ---")
    a_dev_norm, a_val_norm = prep.normalize_windows(splits['room_a_dev']['windows'], splits['room_a_val']['windows'], mode='static_removal')
    b_dev_norm, b_val_norm = prep.normalize_windows(splits['room_b_dev']['windows'], splits['room_b_val']['windows'], mode='static_removal')
    
    a_dev_emb = extract_embeddings(ssl_encoder, a_dev_norm, device=device)
    a_val_emb = extract_embeddings(ssl_encoder, a_val_norm, device=device)
    b_dev_emb = extract_embeddings(ssl_encoder, b_dev_norm, device=device)
    b_val_emb = extract_embeddings(ssl_encoder, b_val_norm, device=device)
    
    configs = [
        ('Euclidean', False),
        ('Euclidean', True),
        ('Cosine', False),
        ('Cosine', True)
    ]
    results = {}
    
    for metric_name, align in configs:
        key = f"{metric_name}_{'Aligned' if align else 'NoAlign'}"
        res = run_cross_room_transfer(
            source_emb_dev=a_dev_emb, source_labels_dev=splits['room_a_dev']['labels'],
            target_emb_dev=b_dev_emb, target_labels_dev=splits['room_b_dev']['labels'],
            target_captures_dev=splits['room_b_dev']['capture_ids'],
            target_emb_val=b_val_emb, target_labels_val=splits['room_b_val']['labels'],
            direction='A_to_B', k_captures=5, seed=42, domain_align=align, metric=metric_name.lower()
        )
        results[key] = {
            'overall_f1': res['overall_macro_f1'],
            'exclusive_f1': res['exclusive_macro_f1'],
            'shared_f1': res['shared_macro_f1']
        }
        print(f"  {key:20s}: Overall F1 = {res['overall_macro_f1']:.4f}, Exclusive F1 = {res['exclusive_macro_f1']:.4f}, Shared F1 = {res['shared_macro_f1']:.4f}")
        
    return results

def main():
    print("=====================================================")
    print("        Executing Comprehensive Ablation Studies     ")
    print("=====================================================")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    splits = prep.load_cached_datasets()
    
    model_path = r"D:\ubi\p2\models\ssl_encoder.pth"
    ssl_encoder = CSIEncoder(in_channels=256, hidden_dim=128, feature_dim=128, proj_dim=64).to(device)
    ssl_encoder.load_state_dict(torch.load(model_path, map_location=device))
    ssl_encoder.eval()
    
    norm_res = run_normalization_ablation(splits, device)
    rep_res = run_representation_ablation(splits, ssl_encoder, device)
    align_res = run_alignment_and_metric_ablation(splits, ssl_encoder, device)
    
    out_file = r"D:\ubi\p2\results\ablation_results.json"
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w") as f:
        json.dump({
            'normalization_ablation': norm_res,
            'representation_ablation': rep_res,
            'alignment_metric_ablation': align_res
        }, f, indent=2)
    print(f"\n[*] Ablation results successfully saved to {out_file}")

if __name__ == "__main__":
    main()
