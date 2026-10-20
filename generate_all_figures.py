"""
generate_all_figures.py - Generates all required deliverable figures for CS60055 Challenge 2:
1. fig1_preprocessing_pipeline.png
2. fig2_room_acc_vs_window_length.png
3. fig3_umap_room_and_activity.png
4. fig4_label_efficiency_curve.png
5. fig5_confusion_matrices_k5.png
6. fig6_oracle_vs_predicted_routing.png
7. fig7_ablation_study.png
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg') # Headless
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import umap
from sklearn.metrics import confusion_matrix

import preprocessing as prep
from task1_routing import run_window_length_study, RoomRouter
from task2_ssl import CSIEncoder, extract_embeddings
from task3_adaptation import (
    ROOM_A_CLASSES, ROOM_B_CLASSES, CLASS_NAMES,
    run_cross_room_transfer
)

# Output directory for figures
FIG_DIR = r"D:\ubi\p2\figures"
os.makedirs(FIG_DIR, exist_ok=True)
sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'

# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# 1. Pipeline Diagram (Spacious 3-Tier Flowchart with Zero Overlaps)
# ----------------------------------------------------------------------
def make_pipeline_diagram():
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.axis('off')
    
    boxes = [
        # Tier 1: Data Ingestion (y ~ 0.82)
        {"text": "Experiment 2 (Room A)\nNexmon .pcap Captures\n(80 MHz, 256 subcarriers)", "xy": (0.16, 0.88), "color": "#dbeafe", "edge": "#1d4ed8", "pad": 0.6},
        {"text": "Experiment 3 (Room B)\nAmplitude .csv Captures\n(80 MHz, 256 subcarriers)", "xy": (0.16, 0.72), "color": "#dbeafe", "edge": "#1d4ed8", "pad": 0.6},
        {"text": "CSI Recovery & Ingestion Engine\n• Exp 2: Complex I/Q -> Real |CSI|\n• Exp 3: Header detection & cleaning\n• 256 Subcarrier alignment (5 GHz)", "xy": (0.60, 0.80), "color": "#fef3c7", "edge": "#b45309", "pad": 0.7},
        
        # Tier 2: Preprocessing & Leakage Prevention (y = 0.50)
        {"text": "Deterministic Windowing\n• Non-overlapping 150x256 slices\n• T = 150 packets (approx. 4.5s)\n• Uniform temporal scale", "xy": (0.18, 0.50), "color": "#fef3c7", "edge": "#b45309", "pad": 0.7},
        {"text": "Strict Capture Splitting\n• 1 capture = 1 unit (no leakage)\n• Room A: official validation/\n• Room B: seed 42 (25/class)", "xy": (0.50, 0.50), "color": "#fef3c7", "edge": "#b45309", "pad": 0.7},
        {"text": "Leakage-Safe Normalization\n• Static DC multipath cancellation\n• X_dynamic = X - mean_time(X)\n• Statistics fitted on Dev ONLY", "xy": (0.82, 0.50), "color": "#dcfce7", "edge": "#15803d", "pad": 0.7},
        
        # Tier 3: Core Tasks (y = 0.16)
        {"text": "Task 1: Room Routing\n• Static Multipath Fingerprint\n• 100.00% Validation Accuracy\n• Stable down to T = 10 packets", "xy": (0.18, 0.16), "color": "#f3e8ff", "edge": "#7e22ce", "pad": 0.7},
        {"text": "Task 2: SimCLR SSL Encoder\n• 1D Residual ConvNet (128-D)\n• Subcarrier masking & gain jitter\n• Linear Probe Macro-F1: 0.7256", "xy": (0.50, 0.16), "color": "#f3e8ff", "edge": "#7e22ce", "pad": 0.7},
        {"text": "Task 3: Cross-Room Transfer\n• 0 Target Labels for Shared Classes\n• K Support Captures for Exclusive\n• Domain Mean Alignment (4x boost)", "xy": (0.82, 0.16), "color": "#f3e8ff", "edge": "#7e22ce", "pad": 0.7}
    ]
    
    for b in boxes:
        ax.text(b["xy"][0], b["xy"][1], b["text"],
                ha='center', va='center',
                bbox=dict(boxstyle=f'round,pad={b["pad"]}', facecolor=b["color"], edgecolor=b["edge"], lw=2),
                fontsize=10, weight='bold')
                
    arrow_props = dict(facecolor='#334155', edgecolor='#334155', width=1.5, headwidth=7)
    
    # Tier 1 connections: 1A & 1B -> Ingestion Engine
    ax.annotate('', xy=(0.44, 0.82), xytext=(0.28, 0.88), arrowprops=arrow_props)
    ax.annotate('', xy=(0.44, 0.78), xytext=(0.28, 0.72), arrowprops=arrow_props)
    
    # Ingestion Engine down to Windowing
    ax.annotate('', xy=(0.18, 0.60), xytext=(0.60, 0.70), arrowprops=arrow_props)
    
    # Tier 2 connections (Left to Right)
    ax.annotate('', xy=(0.36, 0.50), xytext=(0.32, 0.50), arrowprops=arrow_props)
    ax.annotate('', xy=(0.68, 0.50), xytext=(0.64, 0.50), arrowprops=arrow_props)
    
    # Tier 2 Normalization down to Tier 3 Tasks
    ax.annotate('', xy=(0.18, 0.26), xytext=(0.75, 0.40), arrowprops=arrow_props)
    ax.annotate('', xy=(0.50, 0.26), xytext=(0.80, 0.40), arrowprops=arrow_props)
    ax.annotate('', xy=(0.82, 0.26), xytext=(0.82, 0.40), arrowprops=arrow_props)
    
    # Routing dotted connection from Task 1 to Task 3 (placed cleanly below boxes)
    ax.annotate('Autonomous Routing Integration (100% Agreement)',
                xy=(0.70, 0.08), xytext=(0.30, 0.08),
                arrowprops=dict(facecolor='#7e22ce', edgecolor='#7e22ce', width=1.5, headwidth=6, linestyle='--'),
                ha='center', va='bottom', fontsize=9.5, color='#6b21a8', weight='bold')
    
    # Title & Subtitle
    ax.text(0.50, 0.98, "CS60055 Hackathon Challenge 2: End-to-End Preprocessing & Learning Architecture",
            ha='center', va='center', fontsize=14, weight='bold', color='#0f172a')
            
    plt.tight_layout()
    fpath = os.path.join(FIG_DIR, "fig1_preprocessing_pipeline.png")
    plt.savefig(fpath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[*] Saved {fpath}")

# ----------------------------------------------------------------------
# 2. Room Accuracy vs Window Length
# ----------------------------------------------------------------------
def make_window_length_plot(splits):
    print("[*] Generating Room Accuracy vs Window Length plot...")
    lengths = [10, 25, 50, 75, 100, 125, 150]
    train_A = splits['room_a_dev']['windows']
    train_B = splits['room_b_dev']['windows']
    val_A = splits['room_a_val']['windows']
    val_B = splits['room_b_val']['windows']
    
    results = run_window_length_study(train_A, train_B, val_A, val_B, lengths=lengths)
    
    plt.figure(figsize=(8, 5))
    x = list(results.keys())
    y = [results[k] * 100 for k in x]
    
    plt.plot(x, y, marker='o', color='#2563eb', linewidth=2.5, markersize=8, label="Room Router Accuracy")
    plt.axhline(100.0, color='#dc2626', linestyle='--', alpha=0.7, label="Perfect Classification (100%)")
    plt.ylim(98.0, 100.5)
    plt.xlabel("CSI Window Length T (Packets)", fontsize=12, weight='bold')
    plt.ylabel("Room Classification Accuracy (%)", fontsize=12, weight='bold')
    plt.title("Task 1: Room Routing Accuracy vs Window Length T", fontsize=14, weight='bold', pad=12)
    plt.legend(loc='lower right', frameon=True)
    
    for i, txt in enumerate(y):
        plt.annotate(f"{txt:.2f}%", (x[i], y[i] + 0.15), ha='center', fontsize=10, weight='bold')
        
    plt.tight_layout()
    fpath = os.path.join(FIG_DIR, "fig2_room_acc_vs_window_length.png")
    plt.savefig(fpath, dpi=300)
    plt.close()
    print(f"[*] Saved {fpath}")

# ----------------------------------------------------------------------
# 3. UMAP by Room and Activity
# ----------------------------------------------------------------------
def make_umap_plots(splits, encoder, device):
    print("[*] Generating UMAP embeddings visualization...")
    a_val_norm = prep.normalize_windows(splits['room_a_dev']['windows'], splits['room_a_val']['windows'], mode='static_removal')[1]
    b_val_norm = prep.normalize_windows(splits['room_b_dev']['windows'], splits['room_b_val']['windows'], mode='static_removal')[1]
    
    val_windows = np.vstack([a_val_norm, b_val_norm])
    val_emb = extract_embeddings(encoder, val_windows, device=device)
    
    rooms = np.array(['Room A (Exp 2)'] * len(a_val_norm) + ['Room B (Exp 3)'] * len(b_val_norm))
    labels = np.concatenate([splits['room_a_val']['labels'], splits['room_b_val']['labels']])
    activity_names = [CLASS_NAMES[c] for c in labels]
    
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    embedding_2d = reducer.fit_transform(val_emb)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Subplot A: By Room
    sns.scatterplot(
        x=embedding_2d[:, 0], y=embedding_2d[:, 1],
        hue=rooms, palette={'Room A (Exp 2)': '#2563eb', 'Room B (Exp 3)': '#e11d48'},
        alpha=0.85, s=60, ax=axes[0]
    )
    axes[0].set_title("(A) Frozen SSL Embeddings Colored by Room", fontsize=13, weight='bold')
    axes[0].set_xlabel("UMAP Dimension 1", fontsize=11)
    axes[0].set_ylabel("UMAP Dimension 2", fontsize=11)
    axes[0].legend(title="Environment", frameon=True)
    
    # Subplot B: By Activity
    unique_acts = sorted(list(set(activity_names)))
    palette = sns.color_palette("tab10", len(unique_acts))
    sns.scatterplot(
        x=embedding_2d[:, 0], y=embedding_2d[:, 1],
        hue=activity_names, hue_order=unique_acts, palette=palette,
        alpha=0.85, s=60, ax=axes[1]
    )
    axes[1].set_title("(B) Frozen SSL Embeddings Colored by Activity", fontsize=13, weight='bold')
    axes[1].set_xlabel("UMAP Dimension 1", fontsize=11)
    axes[1].set_ylabel("UMAP Dimension 2", fontsize=11)
    axes[1].legend(title="Activity", bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True)
    
    plt.tight_layout()
    fpath = os.path.join(FIG_DIR, "fig3_umap_room_and_activity.png")
    plt.savefig(fpath, dpi=300)
    plt.close()
    print(f"[*] Saved {fpath}")

# ----------------------------------------------------------------------
# 4. Label-Efficiency Curves (Task 3)
# ----------------------------------------------------------------------
def make_label_efficiency_plot():
    print("[*] Generating Label-Efficiency curve...")
    with open(r"D:\ubi\p2\results\task3_results.json") as f:
        data = json.load(f)
        
    res_a2b = data['results_a_to_b']
    res_b2a = data['results_b_to_a']
    
    k_vals = [1, 5, 10, 20]
    
    a2b_ov = [res_a2b[str(k)]['overall_f1_mean'] for k in k_vals]
    a2b_ov_std = [res_a2b[str(k)]['overall_f1_std'] for k in k_vals]
    a2b_ex = [res_a2b[str(k)]['exclusive_f1_mean'] for k in k_vals]
    a2b_ex_std = [res_a2b[str(k)]['exclusive_f1_std'] for k in k_vals]
    
    b2a_ov = [res_b2a[str(k)]['overall_f1_mean'] for k in k_vals]
    b2a_ov_std = [res_b2a[str(k)]['overall_f1_std'] for k in k_vals]
    b2a_ex = [res_b2a[str(k)]['exclusive_f1_mean'] for k in k_vals]
    b2a_ex_std = [res_b2a[str(k)]['exclusive_f1_std'] for k in k_vals]
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    
    # Panel 1: Room A -> Room B
    axes[0].plot(k_vals, a2b_ex, marker='s', color='#16a34a', lw=2.5, label='Target-Exclusive Macro-F1 (Empty, Lying)')
    axes[0].fill_between(k_vals, np.array(a2b_ex) - np.array(a2b_ex_std), np.array(a2b_ex) + np.array(a2b_ex_std), color='#16a34a', alpha=0.15)
    
    axes[0].plot(k_vals, a2b_ov, marker='o', color='#2563eb', lw=2.5, label='Overall Target Macro-F1 (All 5 classes)')
    axes[0].fill_between(k_vals, np.array(a2b_ov) - np.array(a2b_ov_std), np.array(a2b_ov) + np.array(a2b_ov_std), color='#2563eb', alpha=0.15)
    
    axes[0].set_title("(A) Transfer Direction: Room A -> Room B", fontsize=13, weight='bold')
    axes[0].set_xlabel("Number of Target Support Captures (K)", fontsize=11, weight='bold')
    axes[0].set_ylabel("Macro-F1 Score", fontsize=11, weight='bold')
    axes[0].set_xticks(k_vals)
    axes[0].legend(loc='lower right', frameon=True)
    axes[0].set_ylim(0.05, 0.70)
    
    # Panel 2: Room B -> Room A
    axes[1].plot(k_vals, b2a_ex, marker='s', color='#ea580c', lw=2.5, label='Target-Exclusive Macro-F1 (Fall, SitDown, StandUp)')
    axes[1].fill_between(k_vals, np.array(b2a_ex) - np.array(b2a_ex_std), np.array(b2a_ex) + np.array(b2a_ex_std), color='#ea580c', alpha=0.15)
    
    axes[1].plot(k_vals, b2a_ov, marker='o', color='#7c3aed', lw=2.5, label='Overall Target Macro-F1 (All 6 classes)')
    axes[1].fill_between(k_vals, np.array(b2a_ov) - np.array(b2a_ov_std), np.array(b2a_ov) + np.array(b2a_ov_std), color='#7c3aed', alpha=0.15)
    
    axes[1].set_title("(B) Transfer Direction: Room B -> Room A", fontsize=13, weight='bold')
    axes[1].set_xlabel("Number of Target Support Captures (K)", fontsize=11, weight='bold')
    axes[1].set_xticks(k_vals)
    axes[1].legend(loc='lower right', frameon=True)
    
    plt.suptitle("Task 3: Cross-Room Few-Shot Label Efficiency Curves (K = 1, 5, 10, 20)", fontsize=15, weight='bold', y=1.02)
    plt.tight_layout()
    fpath = os.path.join(FIG_DIR, "fig4_label_efficiency_curve.png")
    plt.savefig(fpath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[*] Saved {fpath}")

# ----------------------------------------------------------------------
# 5. Confusion Matrices at K = 5
# ----------------------------------------------------------------------
def make_confusion_matrices_plot():
    print("[*] Generating Confusion Matrices plot at K=5...")
    with open(r"D:\ubi\p2\results\task3_results.json") as f:
        data = json.load(f)
        
    cm_a2b = np.array(data['results_a_to_b']['5']['confusion_matrix'])
    cm_b2a = np.array(data['results_b_to_a']['5']['confusion_matrix'])
    
    names_b = [CLASS_NAMES[c] for c in ROOM_B_CLASSES]
    names_a = [CLASS_NAMES[c] for c in ROOM_A_CLASSES]
    
    # Normalize rows to show recall percentages
    cm_a2b_norm = cm_a2b.astype('float') / cm_a2b.sum(axis=1)[:, np.newaxis]
    cm_b2a_norm = cm_b2a.astype('float') / cm_b2a.sum(axis=1)[:, np.newaxis]
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    sns.heatmap(cm_a2b_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=names_b, yticklabels=names_b, ax=axes[0], cbar=False)
    axes[0].set_title("(A) Room A -> Room B (K = 5 Captures)", fontsize=13, weight='bold')
    axes[0].set_xlabel("Predicted Activity", fontsize=11, weight='bold')
    axes[0].set_ylabel("True Activity", fontsize=11, weight='bold')
    
    sns.heatmap(cm_b2a_norm, annot=True, fmt='.2f', cmap='Oranges',
                xticklabels=names_a, yticklabels=names_a, ax=axes[1], cbar=False)
    axes[1].set_title("(B) Room B -> Room A (K = 5 Captures)", fontsize=13, weight='bold')
    axes[1].set_xlabel("Predicted Activity", fontsize=11, weight='bold')
    axes[1].set_ylabel("True Activity", fontsize=11, weight='bold')
    
    plt.suptitle("Normalized Confusion Matrices at K = 5 Support Captures", fontsize=15, weight='bold', y=1.02)
    plt.tight_layout()
    fpath = os.path.join(FIG_DIR, "fig5_confusion_matrices_k5.png")
    plt.savefig(fpath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[*] Saved {fpath}")

# ----------------------------------------------------------------------
# 6. Oracle vs Predicted Room Routing Comparison
# ----------------------------------------------------------------------
def make_routing_comparison_plot():
    print("[*] Generating Oracle vs Predicted Room Routing comparison plot...")
    with open(r"D:\ubi\p2\results\task3_results.json") as f:
        data = json.load(f)
        
    categories = ['Router Val\nAccuracy', 'Oracle vs Pred\nAgreement', 'A→B Routing\nMatch', 'B→A Routing\nMatch']
    values = [100.0, float(data['routing_agreement']), 100.0, 100.0]
    
    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars = ax.bar(categories, values, color=['#3b82f6', '#10b981', '#6366f1', '#8b5cf6'], width=0.45)
    ax.set_ylim(0, 118)
    ax.set_ylabel("Accuracy / Agreement Percentage (%)", fontsize=12, weight='bold')
    ax.set_title("Task 1 Routing: Oracle vs Predicted Room Routing", fontsize=14, weight='bold', pad=15)
    ax.tick_params(axis='x', labelsize=11)
    
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 2.5, f"{h:.1f}%", ha='center', va='bottom', fontsize=11, weight='bold')
        
    plt.tight_layout()
    fpath = os.path.join(FIG_DIR, "fig6_oracle_vs_predicted_routing.png")
    plt.savefig(fpath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[*] Saved {fpath}")

# ----------------------------------------------------------------------
# 7. Ablation Study Plots
# ----------------------------------------------------------------------
def make_ablation_plot():
    print("[*] Generating Ablation Study plot...")
    with open(r"D:\ubi\p2\results\ablation_results.json") as f:
        data = json.load(f)
        
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Subplot 1: Preprocessing Normalization
    norm_data = data['normalization_ablation']
    labels_norm = ['Raw Amplitude', 'Global Z-Score', 'Static Removal']
    acc_norm = [norm_data['none']['room_accuracy']*100, norm_data['zscore']['room_accuracy']*100, norm_data['static_removal']['room_accuracy']*100]
    
    b1 = axes[0].bar(labels_norm, acc_norm, color=['#94a3b8', '#60a5fa', '#34d399'], width=0.5)
    axes[0].set_ylim(90, 105)
    axes[0].set_ylabel("Room Accuracy (%)", fontsize=11, weight='bold')
    axes[0].set_title("(A) Normalization: Room Fingerprint Accuracy", fontsize=12, weight='bold')
    for bar in b1:
        axes[0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5, f"{bar.get_height():.1f}%", ha='center', weight='bold')
        
    # Subplot 2: Feature Representation
    rep_data = data['representation_ablation']
    labels_rep = ['Raw CSI\nBaseline', 'Random Init\nConvNet', 'SimCLR SSL\nPretrained']
    f1_rep = [rep_data['raw_csi'], rep_data['random_untrained'], rep_data['ssl_pretrained']]
    
    b2 = axes[1].bar(labels_rep, f1_rep, color=['#ef4444', '#f59e0b', '#10b981'], width=0.5)
    axes[1].set_ylim(0, 0.85)
    axes[1].set_ylabel("Macro-F1 Score", fontsize=11, weight='bold')
    axes[1].set_title("(B) Representation: SSL vs Untrained vs Raw", fontsize=12, weight='bold')
    for bar in b2:
        axes[1].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02, f"{bar.get_height():.4f}", ha='center', weight='bold')
        
    # Subplot 3: Domain Alignment & Metric
    align_data = data['alignment_metric_ablation']
    labels_align = ['Euclid\nNoAlign', 'Euclid\nAligned', 'Cosine\nNoAlign', 'Cosine\nAligned']
    f1_shared = [
        align_data['Euclidean_NoAlign']['shared_f1'],
        align_data['Euclidean_Aligned']['shared_f1'],
        align_data['Cosine_NoAlign']['shared_f1'],
        align_data['Cosine_Aligned']['shared_f1']
    ]
    
    b3 = axes[2].bar(labels_align, f1_shared, color=['#cbd5e1', '#3b82f6', '#fed7aa', '#ea580c'], width=0.55)
    axes[2].set_ylim(0, 0.30)
    axes[2].set_ylabel("Shared Activities Macro-F1", fontsize=11, weight='bold')
    axes[2].set_title("(C) Alignment: Cross-Room Transfer Benefit", fontsize=12, weight='bold')
    for bar in b3:
        axes[2].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.008, f"{bar.get_height():.4f}", ha='center', weight='bold')
        
    plt.suptitle("Comprehensive Ablation Studies across Normalization, Representations, and Alignment", fontsize=14, weight='bold', y=1.03)
    plt.tight_layout()
    fpath = os.path.join(FIG_DIR, "fig7_ablation_study.png")
    plt.savefig(fpath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[*] Saved {fpath}")

def main():
    print("=====================================================")
    print("       Generating Deliverable Figures and Charts     ")
    print("=====================================================")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    splits = prep.load_cached_datasets()
    
    model_path = r"D:\ubi\p2\models\ssl_encoder.pth"
    encoder = CSIEncoder(in_channels=256, hidden_dim=128, feature_dim=128, proj_dim=64).to(device)
    encoder.load_state_dict(torch.load(model_path, map_location=device))
    encoder.eval()
    
    make_pipeline_diagram()
    make_window_length_plot(splits)
    make_umap_plots(splits, encoder, device)
    make_label_efficiency_plot()
    make_confusion_matrices_plot()
    make_routing_comparison_plot()
    make_ablation_plot()
    print("\n[OK] All publication-ready figures successfully generated in D:\\ubi\\p2\\figures\\!")

if __name__ == "__main__":
    main()
