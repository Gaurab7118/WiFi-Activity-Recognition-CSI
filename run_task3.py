"""
run_task3.py - Full execution of Task 3 (Cross-Room Few-Shot Adaptation)
and comparison with Oracle vs Predicted Room Routing.
"""

import os
import json
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import preprocessing as prep
from task1_routing import RoomRouter
from task2_ssl import CSIEncoder, extract_embeddings
from task3_adaptation import (
    evaluate_few_shot_suite, run_cross_room_transfer, classify_nearest_prototype,
    ROOM_A_CLASSES, ROOM_B_CLASSES, SHARED_CLASSES,
    ROOM_A_EXCLUSIVE, ROOM_B_EXCLUSIVE, CLASS_NAMES
)

def main():
    print("=====================================================")
    print("   Running Task 3: Cross-Room Few-Shot Adaptation   ")
    print("=====================================================")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = r"D:\ubi\p2\results"
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. Load Data
    splits = prep.load_cached_datasets()
    
    a_dev_w = splits['room_a_dev']['windows']
    a_val_w = splits['room_a_val']['windows']
    b_dev_w = splits['room_b_dev']['windows']
    b_val_w = splits['room_b_val']['windows']
    
    # Static removal normalization (leakage safe)
    a_dev_norm, a_val_norm = prep.normalize_windows(a_dev_w, a_val_w, mode='static_removal')
    b_dev_norm, b_val_norm = prep.normalize_windows(b_dev_w, b_val_w, mode='static_removal')
    
    # 2. Load Pretrained SSL Encoder
    model_path = r"D:\ubi\p2\models\ssl_encoder.pth"
    encoder = CSIEncoder(in_channels=256, hidden_dim=128, feature_dim=128, proj_dim=64).to(device)
    encoder.load_state_dict(torch.load(model_path, map_location=device))
    encoder.eval()
    print("[*] Loaded SSL encoder weights.")
    
    # 3. Extract Frozen Embeddings
    print("[*] Extracting frozen SSL embeddings...")
    a_dev_emb = extract_embeddings(encoder, a_dev_norm, device=device)
    a_val_emb = extract_embeddings(encoder, a_val_norm, device=device)
    b_dev_emb = extract_embeddings(encoder, b_dev_norm, device=device)
    b_val_emb = extract_embeddings(encoder, b_val_norm, device=device)
    print(f"  Room A Dev embeddings: {a_dev_emb.shape}, Val: {a_val_emb.shape}")
    print(f"  Room B Dev embeddings: {b_dev_emb.shape}, Val: {b_val_emb.shape}")
    
    # 4. Train Task 1 Room Router for Predicted Routing
    print("\n[*] Training Task 1 Room Router for Routing Integration...")
    router = RoomRouter(model_type='logistic')
    router.fit(a_dev_w, b_dev_w)
    
    # Evaluate router on val
    router_eval = router.evaluate(a_val_w, b_val_w)
    print(f"  Router Validation Accuracy: {router_eval['accuracy'] * 100:.2f}%")
    
    # 5. Evaluate Few-Shot Suite for A -> B
    print("\n-----------------------------------------------------")
    print("   Direction: Room A -> Room B (Target: Room B)      ")
    print("-----------------------------------------------------")
    print("Shared classes (sit, stand, walk): 0 labels in Room B!")
    print("Target-exclusive classes (empty, lying): K captures in Room B.")
    
    k_values = [1, 5, 10, 20]
    results_a_to_b = evaluate_few_shot_suite(
        source_emb_dev=a_dev_emb,
        source_labels_dev=splits['room_a_dev']['labels'],
        target_emb_dev=b_dev_emb,
        target_labels_dev=splits['room_b_dev']['labels'],
        target_captures_dev=splits['room_b_dev']['capture_ids'],
        target_emb_val=b_val_emb,
        target_labels_val=splits['room_b_val']['labels'],
        direction='A_to_B',
        k_values=k_values,
        n_seeds=5,
        domain_align=True,
        metric='cosine'
    )
    
    for k in k_values:
        res = results_a_to_b[k]
        print(f"  K = {k:2d}: Overall Macro-F1 = {res['overall_f1_mean']:.4f} ± {res['overall_f1_std']:.4f} | Target-Exclusive Macro-F1 = {res['exclusive_f1_mean']:.4f} ± {res['exclusive_f1_std']:.4f} | Shared Macro-F1 = {res['shared_f1_mean']:.4f} ± {res['shared_f1_std']:.4f}")
        
    # 6. Evaluate Few-Shot Suite for B -> A
    print("\n-----------------------------------------------------")
    print("   Direction: Room B -> Room A (Target: Room A)      ")
    print("-----------------------------------------------------")
    print("Shared classes (sit, stand, walk): 0 labels in Room A!")
    print("Target-exclusive classes (fall, sitdown, standup): K captures in Room A.")
    
    results_b_to_a = evaluate_few_shot_suite(
        source_emb_dev=b_dev_emb,
        source_labels_dev=splits['room_b_dev']['labels'],
        target_emb_dev=a_dev_emb,
        target_labels_dev=splits['room_a_dev']['labels'],
        target_captures_dev=splits['room_a_dev']['capture_ids'],
        target_emb_val=a_val_emb,
        target_labels_val=splits['room_a_val']['labels'],
        direction='B_to_A',
        k_values=k_values,
        n_seeds=5,
        domain_align=True,
        metric='cosine'
    )
    
    for k in k_values:
        res = results_b_to_a[k]
        print(f"  K = {k:2d}: Overall Macro-F1 = {res['overall_f1_mean']:.4f} ± {res['overall_f1_std']:.4f} | Target-Exclusive Macro-F1 = {res['exclusive_f1_mean']:.4f} ± {res['exclusive_f1_std']:.4f} | Shared Macro-F1 = {res['shared_f1_mean']:.4f} ± {res['shared_f1_std']:.4f}")

    # 7. Detailed Per-Class Metrics at K = 5
    print("\n-----------------------------------------------------")
    print("   Detailed Per-Class Metrics at K = 5               ")
    print("-----------------------------------------------------")
    res_k5_a2b = run_cross_room_transfer(
        source_emb_dev=a_dev_emb, source_labels_dev=splits['room_a_dev']['labels'],
        target_emb_dev=b_dev_emb, target_labels_dev=splits['room_b_dev']['labels'],
        target_captures_dev=splits['room_b_dev']['capture_ids'],
        target_emb_val=b_val_emb, target_labels_val=splits['room_b_val']['labels'],
        direction='A_to_B', k_captures=5, seed=42, domain_align=True, metric='cosine'
    )
    print("\n[A -> B, K=5] Classification Report:")
    target_names_b = [CLASS_NAMES[c] for c in ROOM_B_CLASSES]
    report_b = classification_report(res_k5_a2b['ground_truth'], res_k5_a2b['predictions'],
                                     labels=ROOM_B_CLASSES, target_names=target_names_b, digits=4)
    print(report_b)
    print("[A -> B, K=5] Confusion Matrix:")
    print(np.array(res_k5_a2b['confusion_matrix']))

    res_k5_b2a = run_cross_room_transfer(
        source_emb_dev=b_dev_emb, source_labels_dev=splits['room_b_dev']['labels'],
        target_emb_dev=a_dev_emb, target_labels_dev=splits['room_a_dev']['labels'],
        target_captures_dev=splits['room_a_dev']['capture_ids'],
        target_emb_val=a_val_emb, target_labels_val=splits['room_a_val']['labels'],
        direction='B_to_A', k_captures=5, seed=42, domain_align=True, metric='cosine'
    )
    print("\n[B -> A, K=5] Classification Report:")
    target_names_a = [CLASS_NAMES[c] for c in ROOM_A_CLASSES]
    report_a = classification_report(res_k5_b2a['ground_truth'], res_k5_b2a['predictions'],
                                     labels=ROOM_A_CLASSES, target_names=target_names_a, digits=4)
    print(report_a)
    print("[B -> A, K=5] Confusion Matrix:")
    print(np.array(res_k5_b2a['confusion_matrix']))

    # 8. Oracle vs Predicted Room Routing Comparison
    print("\n-----------------------------------------------------")
    print("   Oracle vs Predicted Room Routing Comparison      ")
    print("-----------------------------------------------------")
    # All validation queries: A val + B val
    all_val_windows = np.vstack([a_val_w, b_val_w])
    all_val_norm = np.vstack([a_val_norm, b_val_norm])
    all_val_emb = np.vstack([a_val_emb, b_val_emb])
    true_rooms = np.array([0] * len(a_val_w) + [1] * len(b_val_w))
    
    # Task 1 router predictions:
    pred_rooms = router.predict(all_val_windows)
    
    # In Oracle routing:
    # If true room == 0 (Room A), route to B->A model (target Room A)
    # If true room == 1 (Room B), route to A->B model (target Room B)
    oracle_preds = np.zeros(len(all_val_windows), dtype=np.int64)
    oracle_preds[:len(a_val_w)] = res_k5_b2a['predictions']
    oracle_preds[len(a_val_w):] = res_k5_a2b['predictions']
    
    # In Predicted routing:
    # Route according to pred_rooms
    pred_routed_preds = np.zeros(len(all_val_windows), dtype=np.int64)
    for i in range(len(all_val_windows)):
        q_emb = all_val_emb[i:i+1]
        if pred_rooms[i] == 0:
            pred_routed_preds[i] = classify_nearest_prototype(q_emb, res_k5_b2a['prototypes'], metric='cosine')[0]
        else:
            pred_routed_preds[i] = classify_nearest_prototype(q_emb, res_k5_a2b['prototypes'], metric='cosine')[0]
            
    routing_match = np.mean(oracle_preds == pred_routed_preds) * 100
    print(f"Oracle vs Predicted Room Routing agreement: {routing_match:.2f}%")
    print(f"Task 1 Router Accuracy on Val: {np.mean(true_rooms == pred_rooms) * 100:.2f}%")
    
    # Save results to JSON
    all_results = {
        'results_a_to_b': results_a_to_b,
        'results_b_to_a': results_b_to_a,
        'k5_a2b_report': classification_report(res_k5_a2b['ground_truth'], res_k5_a2b['predictions'],
                                              labels=ROOM_B_CLASSES, target_names=target_names_b, output_dict=True),
        'k5_b2a_report': classification_report(res_k5_b2a['ground_truth'], res_k5_b2a['predictions'],
                                              labels=ROOM_A_CLASSES, target_names=target_names_a, output_dict=True),
        'router_eval': router_eval,
        'routing_agreement': float(routing_match)
    }
    
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.int64, np.int32, np.integer)):
                return int(obj)
            if isinstance(obj, (np.float64, np.float32, np.floating)):
                return float(obj)
            return super().default(obj)

    with open(os.path.join(out_dir, "task3_results.json"), "w") as f:
        json.dump(all_results, f, indent=2, cls=NumpyEncoder)
    print(f"\n[*] Full Task 3 results saved to {os.path.join(out_dir, 'task3_results.json')}")

if __name__ == "__main__":
    main()
