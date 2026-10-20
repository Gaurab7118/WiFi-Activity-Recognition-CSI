"""
preprocessing.py - Data loading, deterministic windowing, capture-based splitting,
and leakage-safe normalization for WiFi CSI Hackathon Challenge 2 (CS60055).
"""

import os
import glob
import numpy as np
import pandas as pd
try:
    import csiread
except ImportError:
    csiread = None

# Universal Activity Definitions
SHARED_ACTIVITIES = ['sit', 'stand', 'walk']
ROOM_A_EXCLUSIVE = ['fall', 'sitdown', 'standup']
ROOM_B_EXCLUSIVE = ['empty', 'lying']

# 8 global unique classes
ALL_ACTIVITIES = ['sit', 'stand', 'walk', 'fall', 'sitdown', 'standup', 'empty', 'lying']
ACTIVITY_TO_ID = {name: idx for idx, name in enumerate(ALL_ACTIVITIES)}
ID_TO_ACTIVITY = {idx: name for idx, name in enumerate(ALL_ACTIVITIES)}

# Activity mapping from raw directory/file prefixes to canonical names
ROOM_A_NAME_MAP = {
    'sit': 'sit',
    'stand': 'stand',
    'walk': 'walk',
    'fall': 'fall',
    'sitdown': 'sitdown',
    'standup': 'standup'
}

ROOM_B_NAME_MAP = {
    'Sitting': 'sit',
    'Standing': 'stand',
    'Walking': 'walk',
    'Empty': 'empty',
    'Lying': 'lying'
}

def window_csi_sequence(sequence, window_size=150):
    """
    Slices a continuous CSI amplitude sequence of shape (T, 256) into
    deterministic, non-overlapping windows of shape (N, 150, 256).
    """
    num_windows = len(sequence) // window_size
    if num_windows == 0:
        return np.zeros((0, window_size, 256), dtype=np.float32)
    windows = sequence[:num_windows * window_size]
    return windows.reshape(num_windows, window_size, 256).astype(np.float32)

def build_room_b_splits(room_b_path, seed=42, val_size=25):
    """
    Deterministically splits Room B captures into development and validation sets
    with exactly 25 captures per class for validation (seed 42).
    Split is performed strictly by capture.
    """
    csv_files = glob.glob(os.path.join(room_b_path, "*.csv"))
    class_groups = {}
    for f in csv_files:
        prefix = os.path.basename(f).split('_')[0]
        if prefix in ROOM_B_NAME_MAP:
            class_groups.setdefault(prefix, []).append(f)
            
    rng = np.random.RandomState(seed)
    train_files = []
    val_files = []
    
    for prefix in sorted(class_groups.keys()):
        files = sorted(class_groups[prefix])
        indices = rng.permutation(len(files))
        val_idx = set(indices[:val_size])
        
        for idx, f in enumerate(files):
            if idx in val_idx:
                val_files.append((f, prefix))
            else:
                train_files.append((f, prefix))
                
    return train_files, val_files

def load_cached_datasets(cache_dir=r"D:\ubi\p2\processed_data"):
    """
    Loads all four splits from cached .npz files.
    Returns:
        dict containing 'a_dev', 'a_val', 'b_dev', 'b_val'
    """
    splits = {}
    for name in ['room_a_dev', 'room_a_val', 'room_b_dev', 'room_b_val']:
        fpath = os.path.join(cache_dir, f"{name}.npz")
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"Cached file {fpath} not found. Run preprocess_and_cache.py first.")
        data = np.load(fpath)
        splits[name] = {
            'windows': data['windows'].astype(np.float32),
            'labels': data['labels'].astype(np.int64),
            'capture_ids': data['capture_ids'].astype(np.int64),
            'rooms': data['rooms'].astype(np.int64),
            'files': data['files']
        }
    return splits

def normalize_windows(train_windows, test_windows=None, mode='static_removal'):
    """
    Applies leakage-safe normalization fitted on development data.
    
    Modes:
      - 'static_removal': Removes static room multipath by subtracting temporal mean
                          per subcarrier for each window: X_dyn[t, c] = X[t, c] - mean_t(X[:, c]).
                          This preserves dynamic human motion and provides room-invariance.
      - 'zscore': Global mean & std fitted strictly on train_windows, applied to train and test.
      - 'subcarrier_zscore': Per-subcarrier mean & std fitted strictly on train_windows.
      - 'none': Returns raw amplitude without normalization.
    """
    if mode == 'none':
        if test_windows is not None:
            return train_windows.copy(), test_windows.copy()
        return train_windows.copy()

    elif mode == 'static_removal':
        # Window-level static multipath cancellation
        # train_windows: (N, 150, 256)
        train_mean = np.mean(train_windows, axis=1, keepdims=True) # (N, 1, 256)
        train_norm = train_windows - train_mean
        
        # Optional scaling by dev standard deviation for numerical stability
        scale = np.std(train_norm) + 1e-6
        train_norm = train_norm / scale
        
        if test_windows is not None:
            test_mean = np.mean(test_windows, axis=1, keepdims=True)
            test_norm = (test_windows - test_mean) / scale
            return train_norm.astype(np.float32), test_norm.astype(np.float32)
        return train_norm.astype(np.float32)

    elif mode == 'zscore':
        mean = np.mean(train_windows)
        std = np.std(train_windows) + 1e-6
        train_norm = (train_windows - mean) / std
        if test_windows is not None:
            test_norm = (test_windows - mean) / std
            return train_norm.astype(np.float32), test_norm.astype(np.float32)
        return train_norm.astype(np.float32)

    elif mode == 'subcarrier_zscore':
        # Mean & std per subcarrier across dev windows and time steps
        mean = np.mean(train_windows, axis=(0, 1), keepdims=True) # (1, 1, 256)
        std = np.std(train_windows, axis=(0, 1), keepdims=True) + 1e-6
        train_norm = (train_windows - mean) / std
        if test_windows is not None:
            test_norm = (test_windows - mean) / std
            return train_norm.astype(np.float32), test_norm.astype(np.float32)
        return train_norm.astype(np.float32)

    else:
        raise ValueError(f"Unknown normalization mode: {mode}")

def sample_support_captures(windows, labels, capture_ids, target_classes, k_captures, seed=None):
    """
    Samples exactly K captures per target class.
    All windows belonging to the sampled captures are included in the support set.
    This strictly enforces the 'Split by capture / 1 capture = 1 unit' rule.
    
    Returns:
        support_indices: boolean mask or integer indices of selected windows
    """
    rng = np.random.RandomState(seed)
    chosen_indices = []
    
    for cls in target_classes:
        cls_mask = (labels == cls)
        cls_capture_ids = np.unique(capture_ids[cls_mask])
        
        if len(cls_capture_ids) == 0:
            continue
            
        n_sample = min(k_captures, len(cls_capture_ids))
        sampled_captures = rng.choice(cls_capture_ids, size=n_sample, replace=False)
        
        # Select all windows belonging to these sampled captures
        for cap_id in sampled_captures:
            cap_indices = np.where((capture_ids == cap_id) & cls_mask)[0]
            chosen_indices.extend(cap_indices)
            
    return np.array(chosen_indices, dtype=np.int64)