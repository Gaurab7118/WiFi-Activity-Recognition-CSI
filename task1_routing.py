"""
task1_routing.py - Task 1: Where Am I? (Room Classification & Routing)
CS60055 Hackathon Challenge 2
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import joblib

def extract_room_features(windows, sub_window_len=None):
    """
    Extracts room fingerprint features from CSI windows.
    If sub_window_len is provided (e.g. 25, 50, 75, 100, 150),
    truncates the window along the time dimension: windows[:, :sub_window_len, :].
    
    Features:
      - Temporal mean amplitude across subcarriers (256 dims) - static multipath fingerprint
      - Temporal standard deviation per subcarrier (256 dims) - temporal dispersion
    Total feature vector: 512 dimensions (or 256 dims for pure mean)
    """
    if sub_window_len is not None and sub_window_len < windows.shape[1]:
        w = windows[:, :sub_window_len, :]
    else:
        w = windows
        
    mean_feat = np.mean(w, axis=1) # (N, 256)
    std_feat = np.std(w, axis=1)   # (N, 256)
    return np.hstack([mean_feat, std_feat])

class RoomRouter:
    def __init__(self, model_type='logistic'):
        self.model_type = model_type
        if model_type == 'logistic':
            self.model = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
        elif model_type == 'rf':
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")
            
    def fit(self, X_train_A, X_train_B):
        feat_A = extract_room_features(X_train_A)
        feat_B = extract_room_features(X_train_B)
        
        X_train = np.vstack([feat_A, feat_B])
        y_train = np.array([0] * len(feat_A) + [1] * len(feat_B), dtype=np.int64)
        
        self.model.fit(X_train, y_train)
        return self
        
    def predict(self, windows, sub_window_len=None):
        feat = extract_room_features(windows, sub_window_len=sub_window_len)
        return self.model.predict(feat)
        
    def predict_proba(self, windows, sub_window_len=None):
        feat = extract_room_features(windows, sub_window_len=sub_window_len)
        return self.model.predict_proba(feat)
        
    def evaluate(self, val_A, val_B, sub_window_len=None):
        feat_A = extract_room_features(val_A, sub_window_len=sub_window_len)
        feat_B = extract_room_features(val_B, sub_window_len=sub_window_len)
        
        X_val = np.vstack([feat_A, feat_B])
        y_val = np.array([0] * len(feat_A) + [1] * len(feat_B), dtype=np.int64)
        
        preds = self.model.predict(X_val)
        probs = self.model.predict_proba(X_val)[:, 1] if hasattr(self.model, "predict_proba") else None
        
        acc = accuracy_score(y_val, preds)
        prec = precision_score(y_val, preds, average='macro')
        rec = recall_score(y_val, preds, average='macro')
        f1 = f1_score(y_val, preds, average='macro')
        auc = roc_auc_score(y_val, probs) if probs is not None else 1.0
        cm = confusion_matrix(y_val, preds)
        
        return {
            'accuracy': float(acc),
            'precision': float(prec),
            'recall': float(rec),
            'macro_f1': float(f1),
            'roc_auc': float(auc),
            'confusion_matrix': cm.tolist(),
            'predictions': preds,
            'ground_truth': y_val
        }

def run_window_length_study(train_A, train_B, val_A, val_B, lengths=[10, 25, 50, 75, 100, 125, 150]):
    """
    Evaluates room classification accuracy as a function of CSI window length.
    Generates data for the required Slide 11 plot: Room accuracy vs window length.
    """
    results = {}
    router = RoomRouter(model_type='logistic')
    router.fit(train_A, train_B)
    
    for L in lengths:
        res = router.evaluate(val_A, val_B, sub_window_len=L)
        results[L] = res['accuracy']
        print(f"  Window Length T = {L:3d}: Room Accuracy = {res['accuracy'] * 100:.2f}%")
        
    return results

if __name__ == "__main__":
    import preprocessing as prep
    print("=== Running Task 1: Room Routing Evaluation ===")
    splits = prep.load_cached_datasets()
    
    train_A = splits['room_a_dev']['windows']
    train_B = splits['room_b_dev']['windows']
    val_A = splits['room_a_val']['windows']
    val_B = splits['room_b_val']['windows']
    
    print(f"Train windows: Room A={len(train_A)}, Room B={len(train_B)}")
    print(f"Val windows:   Room A={len(val_A)}, Room B={len(val_B)}")
    
    router = RoomRouter(model_type='logistic')
    router.fit(train_A, train_B)
    
    metrics = router.evaluate(val_A, val_B)
    print("\n--- Full Window (T=150) Results ---")
    print(f"Accuracy:  {metrics['accuracy'] * 100:.2f}%")
    print(f"Macro-F1:  {metrics['macro_f1']:.4f}")
    print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
    print("Confusion Matrix (Row=True [A, B], Col=Pred [A, B]):")
    print(np.array(metrics['confusion_matrix']))
    
    print("\n--- Window Length vs Accuracy Study ---")
    length_results = run_window_length_study(train_A, train_B, val_A, val_B)