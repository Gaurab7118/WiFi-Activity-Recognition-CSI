"""
main.py - End-to-End Execution Pipeline for CS60055 Ubiquitous Computing Hackathon Challenge 2
Cross-Room Few-Shot Adaptation (WiFi Activity Recognition with Very Few Labels)
"""

import os
import argparse
import json
import numpy as np
import torch
from sklearn.metrics import classification_report

import preprocessing as prep
from task1_routing import RoomRouter, run_window_length_study
from task2_ssl import CSIEncoder, load_or_train_encoder, extract_embeddings, evaluate_linear_probe
from task3_adaptation import (
    evaluate_few_shot_suite, run_cross_room_transfer, classify_nearest_prototype,
    ROOM_A_CLASSES, ROOM_B_CLASSES, CLASS_NAMES
)

def main():
    parser = argparse.ArgumentParser(description="CS60055 Challenge 2: Cross-Room Few-Shot CSI Pipeline")
    parser.add_argument('--room_a_path', type=str, default=r"data\Experiment-2", help="Path to Room A files")
    parser.add_argument('--room_b_path', type=str, default=r"data\Experiment-3", help="Path to Room B files")
    parser.add_argument('--cache_dir', type=str, default=r"processed_data", help="Directory with cached npz data")
    parser.add_argument('--epochs', type=int, default=15, help="SSL pretraining epochs")
    parser.add_argument('--force_retrain_ssl', action='store_true', help="Force retraining of SSL encoder")
    parser.add_argument('--generate_figures', action='store_true', default=True, help="Generate all report figures")
    args = parser.parse_args()

    print("================================================================================")
    print("      CS60055 Ubiquitous Computing: Hackathon Challenge 2 Pipeline              ")
    print("      Cross-Room Few-Shot WiFi Activity Recognition (Asus RT-AC86U Nexmon)      ")
    print("================================================================================")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Compute Device: {device}\n")

    # --------------------------------------------------------------------------
    # 1. Preprocessing and Data Verification
    # --------------------------------------------------------------------------
    print("[1/4] Verifying and Loading Preprocessed CSI Windows...")
    if not os.path.exists(os.path.join(args.cache_dir, "room_a_dev.npz")):
        print("  [*] Cached data not found. Running preprocessing pipeline...")
        import preprocess_and_cache
        preprocess_and_cache.main()
        
    splits = prep.load_cached_datasets(args.cache_dir)
    print(f"  [+] Room A Dev Windows: {splits['room_a_dev']['windows'].shape} (Captures: {len(np.unique(splits['room_a_dev']['capture_ids']))})")
    print(f"  [+] Room A Val Windows: {splits['room_a_val']['windows'].shape} (Captures: {len(np.unique(splits['room_a_val']['capture_ids']))})")
    print(f"  [+] Room B Dev Windows: {splits['room_b_dev']['windows'].shape} (Captures: {len(np.unique(splits['room_b_dev']['capture_ids']))})")
    print(f"  [+] Room B Val Windows: {splits['room_b_val']['windows'].shape} (Captures: {len(np.unique(splits['room_b_val']['capture_ids']))})")
    print("  [+] Strict capture-based splitting enforced (1 file = 1 unit).")
    
    # Static-removal normalization (fitted on dev, applied to val)
    a_dev_norm, a_val_norm = prep.normalize_windows(splits['room_a_dev']['windows'], splits['room_a_val']['windows'], mode='static_removal')
    b_dev_norm, b_val_norm = prep.normalize_windows(splits['room_b_dev']['windows'], splits['room_b_val']['windows'], mode='static_removal')
    all_dev_norm = np.vstack([a_dev_norm, b_dev_norm])
    print("  [+] Leakage-safe normalization applied (Static DC Multipath Cancellation).\n")

    # --------------------------------------------------------------------------
    # 2. Task 1: Where Am I? (Room Routing)
    # --------------------------------------------------------------------------
    print("[2/4] Executing Task 1: Room Routing (Where Am I?)...")
    router = RoomRouter(model_type='logistic')
    router.fit(splits['room_a_dev']['windows'], splits['room_b_dev']['windows'])
    
    t1_results = router.evaluate(splits['room_a_val']['windows'], splits['room_b_val']['windows'])
    print(f"  -> Room Classification Accuracy: {t1_results['accuracy'] * 100:.2f}%")
    print(f"  -> Room Routing Macro-F1:        {t1_results['macro_f1']:.4f}")
    print(f"  -> ROC-AUC:                      {t1_results['roc_auc']:.4f}")
    print(f"  -> Confusion Matrix [A, B]:      {t1_results['confusion_matrix']}")

    # --------------------------------------------------------------------------
    # 3. Task 2: Learn Without Labels (SSL)
    # --------------------------------------------------------------------------
    print("\n[3/4] Executing Task 2: Self-Supervised Learning (SimCLR CSIEncoder)...")
    model_path = r"models\ssl_encoder.pth"
    encoder = load_or_train_encoder(
        all_dev_norm, model_path=model_path, epochs=args.epochs,
        device=device, force_retrain=args.force_retrain_ssl
    )
    
    # Linear-probe evaluation
    t2_res_a = evaluate_linear_probe(encoder, a_dev_norm, splits['room_a_dev']['labels'], a_val_norm, splits['room_a_val']['labels'], device=device)
    t2_res_b = evaluate_linear_probe(encoder, b_dev_norm, splits['room_b_dev']['labels'], b_val_norm, splits['room_b_val']['labels'], device=device)
    print(f"  -> Room A Linear-Probe Macro-F1: {t2_res_a['macro_f1']:.4f}")
    print(f"  -> Room B Linear-Probe Macro-F1: {t2_res_b['macro_f1']:.4f}")

    # --------------------------------------------------------------------------
    # 4. Task 3: New Room, Few Labels (Cross-Room Adaptation)
    # --------------------------------------------------------------------------
    print("\n[4/4] Executing Task 3: Cross-Room Few-Shot Transfer...")
    a_dev_emb = extract_embeddings(encoder, a_dev_norm, device=device)
    a_val_emb = extract_embeddings(encoder, a_val_norm, device=device)
    b_dev_emb = extract_embeddings(encoder, b_dev_norm, device=device)
    b_val_emb = extract_embeddings(encoder, b_val_norm, device=device)
    
    k_values = [1, 5, 10, 20]
    
    print("\n  --- Transfer Direction A -> B (Target: Room B) ---")
    res_a2b = evaluate_few_shot_suite(
        source_emb_dev=a_dev_emb, source_labels_dev=splits['room_a_dev']['labels'],
        target_emb_dev=b_dev_emb, target_labels_dev=splits['room_b_dev']['labels'],
        target_captures_dev=splits['room_b_dev']['capture_ids'],
        target_emb_val=b_val_emb, target_labels_val=splits['room_b_val']['labels'],
        direction='A_to_B', k_values=k_values, n_seeds=5, domain_align=True, metric='cosine'
    )
    for k in k_values:
        r = res_a2b[k]
        print(f"    K = {k:2d}: Overall F1 = {r['overall_f1_mean']:.4f} ± {r['overall_f1_std']:.4f} | Exclusive F1 = {r['exclusive_f1_mean']:.4f} ± {r['exclusive_f1_std']:.4f} | Shared F1 = {r['shared_f1_mean']:.4f}")

    print("\n  --- Transfer Direction B -> A (Target: Room A) ---")
    res_b2a = evaluate_few_shot_suite(
        source_emb_dev=b_dev_emb, source_labels_dev=splits['room_b_dev']['labels'],
        target_emb_dev=a_dev_emb, target_labels_dev=splits['room_a_dev']['labels'],
        target_captures_dev=splits['room_a_dev']['capture_ids'],
        target_emb_val=a_val_emb, target_labels_val=splits['room_a_val']['labels'],
        direction='B_to_A', k_values=k_values, n_seeds=5, domain_align=True, metric='cosine'
    )
    for k in k_values:
        r = res_b2a[k]
        print(f"    K = {k:2d}: Overall F1 = {r['overall_f1_mean']:.4f} ± {r['overall_f1_std']:.4f} | Exclusive F1 = {r['exclusive_f1_mean']:.4f} ± {r['exclusive_f1_std']:.4f} | Shared F1 = {r['shared_f1_mean']:.4f}")

    if args.generate_figures:
        print("\n[*] Generating / Updating all publication-ready report figures...")
        import generate_all_figures
        generate_all_figures.main()

    print("\n================================================================================")
    print("Pipeline Execution Complete. All results and figures are ready in 'results/' & 'figures/'.")
    print("================================================================================")

if __name__ == "__main__":
    main()