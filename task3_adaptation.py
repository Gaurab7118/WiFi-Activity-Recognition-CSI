"""
task3_adaptation.py - Task 3: New Room, Few Labels (Cross-Room Adaptation)
CS60055 Hackathon Challenge 2
"""

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, classification_report
import preprocessing as prep
from task2_ssl import extract_embeddings

# Universal Activity ID Definitions
# 0: sit, 1: stand, 2: walk
# 3: fall, 4: sitdown, 5: standup (Room A exclusive)
# 6: empty, 7: lying (Room B exclusive)

ROOM_A_CLASSES = [0, 1, 2, 3, 4, 5]
ROOM_B_CLASSES = [0, 1, 2, 6, 7]
SHARED_CLASSES = [0, 1, 2]
ROOM_A_EXCLUSIVE = [3, 4, 5]
ROOM_B_EXCLUSIVE = [6, 7]

CLASS_NAMES = {
    0: 'sit',
    1: 'stand',
    2: 'walk',
    3: 'fall',
    4: 'sitdown',
    5: 'standup',
    6: 'empty',
    7: 'lying'
}

def compute_prototypes(embeddings, labels, classes):
    """
    Computes class prototypes (centroids) in the embedding space.
    embeddings: (N, D) numpy array
    labels: (N,) numpy array
    classes: list of class IDs
    """
    prototypes = {}
    for c in classes:
        mask = (labels == c)
        if np.sum(mask) > 0:
            prototypes[c] = np.mean(embeddings[mask], axis=0)
        else:
            prototypes[c] = np.zeros(embeddings.shape[1], dtype=np.float32)
    return prototypes

def classify_nearest_prototype(query_embeddings, prototypes, metric='cosine'):
    """
    Assigns each query embedding to the nearest prototype.
    metric: 'cosine' or 'euclidean'
    """
    class_list = sorted(prototypes.keys())
    proto_matrix = np.stack([prototypes[c] for c in class_list], axis=0) # (num_classes, D)
    
    if metric == 'cosine':
        # Normalize
        q_norm = query_embeddings / (np.linalg.norm(query_embeddings, axis=1, keepdims=True) + 1e-8)
        p_norm = proto_matrix / (np.linalg.norm(proto_matrix, axis=1, keepdims=True) + 1e-8)
        sim = np.matmul(q_norm, p_norm.T) # (N, num_classes)
        best_idx = np.argmax(sim, axis=1)
    elif metric == 'euclidean':
        diff = query_embeddings[:, np.newaxis, :] - proto_matrix[np.newaxis, :, :] # (N, num_classes, D)
        dist = np.linalg.norm(diff, axis=-1)
        best_idx = np.argmin(dist, axis=1)
    else:
        raise ValueError(f"Unknown metric: {metric}")
        
    return np.array([class_list[i] for i in best_idx])

def run_cross_room_transfer(
    source_emb_dev, source_labels_dev,
    target_emb_dev, target_labels_dev, target_captures_dev,
    target_emb_val, target_labels_val,
    direction='A_to_B',
    k_captures=5,
    seed=42,
    domain_align=True,
    metric='cosine'
):
    """
    Executes cross-room few-shot adaptation for a single (direction, K, seed) configuration.
    
    - direction == 'A_to_B':
        Source: Room A (all dev labels available).
        Target: Room B.
        Shared classes [0, 1, 2] have ZERO target labels (transferred from Room A).
        Target-exclusive classes [6, 7] have K support captures from Room B dev.
        Evaluated on target Room B validation set (5 classes: [0, 1, 2, 6, 7]).
        
    - direction == 'B_to_A':
        Source: Room B (all dev labels available).
        Target: Room A.
        Shared classes [0, 1, 2] have ZERO target labels (transferred from Room B).
        Target-exclusive classes [3, 4, 5] have K support captures from Room A dev.
        Evaluated on target Room A validation set (6 classes: [0, 1, 2, 3, 4, 5]).
    """
    if direction == 'A_to_B':
        target_classes = ROOM_B_CLASSES
        shared_classes = SHARED_CLASSES
        exclusive_classes = ROOM_B_EXCLUSIVE
    elif direction == 'B_to_A':
        target_classes = ROOM_A_CLASSES
        shared_classes = SHARED_CLASSES
        exclusive_classes = ROOM_A_EXCLUSIVE
    else:
        raise ValueError(f"Unknown direction: {direction}")

    # 1. Compute source prototypes for shared classes
    source_prototypes = compute_prototypes(source_emb_dev, source_labels_dev, shared_classes)

    # 2. Domain Alignment: align source prototypes into target feature space
    if domain_align:
        mu_src = np.mean(source_emb_dev, axis=0)
        mu_tgt = np.mean(target_emb_dev, axis=0)
        domain_shift = mu_tgt - mu_src
        for c in shared_classes:
            source_prototypes[c] = source_prototypes[c] + domain_shift

    # 3. Sample K support captures for target-exclusive classes
    support_indices = prep.sample_support_captures(
        None, target_labels_dev, target_captures_dev,
        target_classes=exclusive_classes, k_captures=k_captures, seed=seed
    )
    
    target_support_emb = target_emb_dev[support_indices]
    target_support_labels = target_labels_dev[support_indices]
    exclusive_prototypes = compute_prototypes(target_support_emb, target_support_labels, exclusive_classes)

    # 4. Combine prototypes into target room classifier
    combined_prototypes = {}
    for c in shared_classes:
        combined_prototypes[c] = source_prototypes[c]
    for c in exclusive_classes:
        combined_prototypes[c] = exclusive_prototypes[c]

    # 5. Predict on target validation set
    preds = classify_nearest_prototype(target_emb_val, combined_prototypes, metric=metric)

    # 6. Compute metrics
    overall_macro_f1 = f1_score(target_labels_val, preds, average='macro', zero_division=0)
    
    # Target-exclusive Macro-F1 (only on exclusive classes)
    excl_mask = np.isin(target_labels_val, exclusive_classes)
    if np.sum(excl_mask) > 0:
        excl_macro_f1 = f1_score(target_labels_val[excl_mask], preds[excl_mask],
                                 labels=exclusive_classes, average='macro', zero_division=0)
    else:
        excl_macro_f1 = 0.0

    # Shared-classes Macro-F1
    shared_mask = np.isin(target_labels_val, shared_classes)
    if np.sum(shared_mask) > 0:
        shared_macro_f1 = f1_score(target_labels_val[shared_mask], preds[shared_mask],
                                   labels=shared_classes, average='macro', zero_division=0)
    else:
        shared_macro_f1 = 0.0

    cm = confusion_matrix(target_labels_val, preds, labels=target_classes)

    return {
        'overall_macro_f1': float(overall_macro_f1),
        'exclusive_macro_f1': float(excl_macro_f1),
        'shared_macro_f1': float(shared_macro_f1),
        'predictions': preds,
        'ground_truth': target_labels_val,
        'confusion_matrix': cm.tolist(),
        'target_classes': target_classes,
        'prototypes': combined_prototypes
    }

def evaluate_few_shot_suite(
    source_emb_dev, source_labels_dev,
    target_emb_dev, target_labels_dev, target_captures_dev,
    target_emb_val, target_labels_val,
    direction='A_to_B',
    k_values=[1, 5, 10, 20],
    n_seeds=5,
    domain_align=True,
    metric='cosine'
):
    """
    Evaluates cross-room adaptation over multiple values of K and random seeds.
    Returns aggregated metrics (mean and std).
    """
    results_by_k = {}
    
    for k in k_values:
        f1_overall_list = []
        f1_excl_list = []
        f1_shared_list = []
        last_cm = None
        last_preds = None
        
        for seed_idx in range(n_seeds):
            seed = 42 + seed_idx * 17
            res = run_cross_room_transfer(
                source_emb_dev, source_labels_dev,
                target_emb_dev, target_labels_dev, target_captures_dev,
                target_emb_val, target_labels_val,
                direction=direction,
                k_captures=k,
                seed=seed,
                domain_align=domain_align,
                metric=metric
            )
            f1_overall_list.append(res['overall_macro_f1'])
            f1_excl_list.append(res['exclusive_macro_f1'])
            f1_shared_list.append(res['shared_macro_f1'])
            if seed_idx == 0:
                last_cm = res['confusion_matrix']
                last_preds = res['predictions']
                
        results_by_k[k] = {
            'overall_f1_mean': float(np.mean(f1_overall_list)),
            'overall_f1_std': float(np.std(f1_overall_list)),
            'exclusive_f1_mean': float(np.mean(f1_excl_list)),
            'exclusive_f1_std': float(np.std(f1_excl_list)),
            'shared_f1_mean': float(np.mean(f1_shared_list)),
            'shared_f1_std': float(np.std(f1_shared_list)),
            'confusion_matrix': last_cm,
            'sample_predictions': last_preds.tolist() if last_preds is not None else []
        }
        
    return results_by_k