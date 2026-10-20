"""
task2_ssl.py - Task 2: Learn Without Labels (Self-Supervised Learning)
CS60055 Hackathon Challenge 2
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report

# ============================================================================
# 1. Fast Vectorized CSI Batch Augmentations
# ============================================================================
def batch_augment_csi(x, mask_ratio=0.15, noise_std=0.05, scale_min=0.85, scale_max=1.15):
    """
    x: torch.Tensor of shape (B, 256, 150)
    Vectorized batch augmentation across GPU/CPU.
    """
    B, C, T = x.shape
    device = x.device
    
    # 1. Amplitude Scaling
    scales = torch.empty(B, 1, 1, device=device).uniform_(scale_min, scale_max)
    aug = x * scales
    
    # 2. Gaussian Noise
    if noise_std > 0:
        aug = aug + torch.randn_like(aug) * noise_std
        
    # 3. Frequency Subcarrier Masking (simulating fading nulls)
    if mask_ratio > 0:
        mask = (torch.rand(B, C, 1, device=device) > mask_ratio).float()
        aug = aug * mask
        
    # 4. Temporal jitter / roll
    shift = random.randint(-8, 8)
    aug = torch.roll(aug, shifts=shift, dims=-1)
    
    return aug

# ============================================================================
# 2. ResNet-1D CSI Encoder Architecture
# ============================================================================
class ResidualBlock1D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Conv1d(channels, channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(channels)
        )
        self.gelu = nn.GELU()

    def forward(self, x):
        return self.gelu(x + self.conv(x))

class CSIEncoder(nn.Module):
    def __init__(self, in_channels=256, hidden_dim=128, feature_dim=128, proj_dim=64):
        super().__init__()
        # Stem: compress 256 subcarriers to hidden_dim
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU()
        )
        
        # Stage 1: (150 -> 75)
        self.res1 = ResidualBlock1D(hidden_dim)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Stage 2: (75 -> 37)
        self.trans1 = nn.Sequential(
            nn.Conv1d(hidden_dim, feature_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(feature_dim),
            nn.GELU()
        )
        self.res2 = ResidualBlock1D(feature_dim)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Stage 3: (37 -> feature_dim)
        self.res3 = ResidualBlock1D(feature_dim)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # SimCLR Projection Head
        self.projection_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, proj_dim)
        )

    def extract_features(self, x):
        """
        x: (Batch, 150, 256) or (Batch, 256, 150)
        Returns: (Batch, feature_dim)
        """
        if x.shape[1] == 150 and x.shape[2] == 256:
            x = x.transpose(1, 2)
        h = self.stem(x)
        h = self.pool1(self.res1(h))
        h = self.trans1(h)
        h = self.pool2(self.res2(h))
        h = self.res3(h)
        feat = self.global_pool(h).squeeze(-1)
        return feat

    def forward(self, x):
        feat = self.extract_features(x)
        proj = self.projection_head(feat)
        return feat, proj

# ============================================================================
# 3. NT-Xent (InfoNCE) Loss
# ============================================================================
class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j):
        batch_size = z_i.shape[0]
        z_i = F.normalize(z_i, dim=-1)
        z_j = F.normalize(z_j, dim=-1)
        
        representations = torch.cat([z_i, z_j], dim=0) # (2B, D)
        sim_matrix = torch.matmul(representations, representations.T) / self.temperature
        
        # Mask diagonal self-similarity
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z_i.device)
        sim_matrix.masked_fill_(mask, -1e9)
        
        targets = torch.cat([
            torch.arange(batch_size, 2 * batch_size, device=z_i.device),
            torch.arange(0, batch_size, device=z_i.device)
        ])
        
        loss = F.cross_entropy(sim_matrix, targets)
        return loss

# ============================================================================
# 4. Training Loop
# ============================================================================
def train_ssl(encoder, train_windows, epochs=15, batch_size=128, lr=1e-3, device='cpu', save_path=None):
    """
    Trains CSIEncoder using SimCLR contrastive learning on unlabeled dev windows.
    """
    # train_windows: (N, 150, 256) -> transpose to (N, 256, 150)
    data_tensor = torch.from_numpy(train_windows).float().transpose(1, 2)
    dataset = TensorDataset(data_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    criterion = NTXentLoss(temperature=0.1)
    optimizer = torch.optim.AdamW(encoder.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    encoder.train()
    print(f"[*] Pre-training CSIEncoder with SimCLR for {epochs} epochs ({len(train_windows)} windows)...")
    
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for (batch_x,) in dataloader:
            batch_x = batch_x.to(device)
            view1 = batch_augment_csi(batch_x)
            view2 = batch_augment_csi(batch_x)
            
            optimizer.zero_grad()
            _, proj1 = encoder(view1)
            _, proj2 = encoder(view2)
            
            loss = criterion(proj1, proj2)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        scheduler.step()
        avg_loss = total_loss / len(dataloader)
        if epoch % 3 == 0 or epoch == 1 or epoch == epochs:
            print(f"  Epoch [{epoch:2d}/{epochs:2d}] - NT-Xent Loss: {avg_loss:.4f}")
            
    encoder.eval()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(encoder.state_dict(), save_path)
        print(f"[*] Encoder model checkpoint saved to {save_path}")
        
    return encoder

def load_or_train_encoder(train_windows, model_path=r"D:\ubi\p2\models\ssl_encoder.pth", epochs=15, device='cpu', force_retrain=False):
    """Loads existing pretrained encoder or trains a new one if not found."""
    encoder = CSIEncoder(in_channels=256, hidden_dim=128, feature_dim=128, proj_dim=64).to(device)
    if os.path.exists(model_path) and not force_retrain:
        print(f"[*] Loading pretrained SSL encoder checkpoint from {model_path}...")
        encoder.load_state_dict(torch.load(model_path, map_location=device))
        encoder.eval()
    else:
        encoder = train_ssl(encoder, train_windows, epochs=epochs, batch_size=128, lr=1e-3, device=device, save_path=model_path)
    return encoder

# ============================================================================
# 5. Feature Extraction and Linear Probe Evaluation
# ============================================================================
def extract_embeddings(encoder, windows, batch_size=256, device='cpu'):
    """Extracts 128-dim frozen feature embeddings from encoder."""
    encoder.eval()
    data_t = torch.from_numpy(windows).float().transpose(1, 2)
    loader = DataLoader(TensorDataset(data_t), batch_size=batch_size, shuffle=False)
    
    features_list = []
    with torch.no_grad():
        for (batch,) in loader:
            batch = batch.to(device)
            feat = encoder.extract_features(batch)
            features_list.append(feat.cpu().numpy())
            
    return np.concatenate(features_list, axis=0)

def evaluate_linear_probe(encoder, train_windows, train_labels, val_windows, val_labels, device='cpu'):
    """
    Evaluates representation quality via linear probing (Macro-F1).
    """
    train_emb = extract_embeddings(encoder, train_windows, device=device)
    val_emb = extract_embeddings(encoder, val_windows, device=device)
    
    clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    clf.fit(train_emb, train_labels)
    preds = clf.predict(val_emb)
    
    macro_f1 = f1_score(val_labels, preds, average='macro')
    report = classification_report(val_labels, preds, output_dict=True, zero_division=0)
    
    return {
        'macro_f1': float(macro_f1),
        'predictions': preds,
        'report': report,
        'train_embeddings': train_emb,
        'val_embeddings': val_emb
    }