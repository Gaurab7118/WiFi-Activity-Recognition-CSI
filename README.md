# CS60055 Ubiquitous Computing — Hackathon Challenge 2
## Cross-Room Few-Shot Adaptation: WiFi Activity Recognition with Very Few Labels

An end-to-end Python framework for cross-environment WiFi Channel State Information (CSI) human activity recognition using self-supervised representation learning and few-shot adaptation.

---

## 1. Overall System Architecture

The core challenge in WiFi sensing is that walls, furniture, and room dimensions heavily reshape multipath propagation. A model trained in one room inevitably encounters a massive domain shift when deployed in another. Our architecture addresses this through a three-stage decoupled design:

```mermaid
flowchart TD
    subgraph Data_Sources["1. Heterogeneous Data Ingestion"]
        RA["Room A (Exp 2)\nRaw .pcap files\n(80 MHz, 256 subcarriers)"] --> D1["Nexmon CSI Decoder\nComplex -> Abs(CSI)"]
        RB["Room B (Exp 3)\nAmplitude .csv files\n(Header-aware detection)"] --> D2["CSV Reader Engine\n500x256 numeric matrix"]
    end

    subgraph Preprocessing["2. Preprocessing & Leakage-Safe Splitting"]
        D1 --> W1["Deterministic Windowing\n(150 x 256 non-overlapping)"]
        D2 --> W2["Deterministic Windowing\n(150 x 256 non-overlapping)"]
        W1 --> S1["Strict Capture Splitting\n1 capture = 1 unit\n(Dev: 926, Val: 147)"]
        W2 --> S2["Strict Capture Splitting\nSeed 42, 25/class\n(Dev: 3228, Val: 375)"]
        S1 --> N1["Static DC Removal\n(Subtract temporal mean)\nFitted ONLY on Dev"]
        S2 --> N2["Static DC Removal\n(Subtract temporal mean)\nFitted ONLY on Dev"]
    end

    subgraph Core_Tasks["3. Three Core Benchmark Tasks"]
        N1 & N2 --> T1["Task 1: Room Router\n(Static Fingerprint)\nAccuracy: 100.00%"]
        N1 & N2 --> T2["Task 2: SimCLR SSL Encoder\n(1D Residual ConvNet)\nLinear Probe Macro-F1: 0.7256"]
        T2 --> T3["Task 3: Cross-Room Few-Shot Transfer\n(Prototypical Networks + Domain Alignment)\nA->B and B->A for K in {1, 5, 10, 20}"]
        T1 -.->|Routes Query Windows| T3
    end

    subgraph Deliverables["4. Evaluation & Artifacts"]
        T1 --> F1["figures/fig2_room_acc_vs_window_length.png"]
        T2 --> F2["figures/fig3_umap_room_and_activity.png"]
        T3 --> F3["figures/fig4_label_efficiency_curve.png"]
        T3 --> F4["figures/fig5_confusion_matrices_k5.png"]
        T1 & T3 --> F5["figures/fig6_oracle_vs_predicted_routing.png"]
        Core_Tasks --> F6["figures/fig7_ablation_study.png"]
    end

    style Data_Sources fill:#eff6ff,stroke:#3b82f6,stroke-width:2px
    style Preprocessing fill:#fefce8,stroke:#eab308,stroke-width:2px
    style Core_Tasks fill:#f5f3ff,stroke:#8b5cf6,stroke-width:2px
    style Deliverables fill:#ecfdf5,stroke:#10b981,stroke-width:2px
```

---

## 2. Step-by-Step Solution Breakdown

```mermaid
sequenceDiagram
    autonumber
    actor User as Researcher
    participant Pipe as Preprocessing Engine
    participant T1 as Task 1 (Room Router)
    participant T2 as Task 2 (SimCLR SSL)
    participant T3 as Task 3 (Few-Shot Adapter)

    User->>Pipe: Run preprocess_and_cache.py
    Note over Pipe: 1. Decode Nexmon complex CSI to real amplitude<br/>2. Slice continuous signals into 150x256 windows<br/>3. Split strictly by capture (seed 42 for Room B)<br/>4. Remove static DC multipath (X - X_mean)
    Pipe-->>User: Cache room_a_dev, room_a_val, room_b_dev, room_b_val (.npz)

    User->>T1: Run task1_routing.py
    Note over T1: Extract mean & std subcarrier fingerprint [512-D]<br/>Train Logistic Regression on Dev windows
    T1-->>User: 100.00% Validation Accuracy (Stable down to T=10 packets)

    User->>T2: Run run_task2.py
    Note over T2: SimCLR contrastive training on 4,154 unlabeled windows<br/>Augmentations: subcarrier masking, gain scaling, jitter<br/>Train ResNet-1D encoder (models/ssl_encoder.pth)
    T2-->>User: Linear Probe Macro-F1: 0.7256 (Room A), 0.6702 (Room B)

    User->>T3: Run run_task3.py
    Note over T3: Zero target labels for shared activities (SIT, STAND, WALK)<br/>K support captures for room-only activities<br/>Domain Mean Alignment + Prototypical Nearest Centroid
    T3-->>User: Macro-F1 across K=1, 5, 10, 20 & 100% Predicted Routing Agreement
```

### Step 1: Ingestion & Preprocessing
- **Room A (Experiment 2, $3.5 \times 4.5\text{ m}$):** Raw Nexmon `.pcap` files decoded to complex $I/Q$ frames and converted to real amplitudes across 256 subcarriers:
  $$\|H(f, t)\| = \sqrt{I(f, t)^2 + Q(f, t)^2}$$
- **Room B (Experiment 3, $4.5 \times 5.0\text{ m}$):** Header-aware CSV reader dynamically handles mixed formatting (text headers `Subcarrier_1..256` vs. raw numeric rows) across all 1,201 files without errors.
- **Deterministic Windowing:** Slices continuous streams into non-overlapping $150 \times 256$ matrices.
- **Strict Capture Splitting (No Leakage):** Enforces 1 file = 1 unit. All windows from a capture stay together. Room A uses the official `validation/` set (25 captures/class). Room B uses a deterministic `seed=42` split (25 captures/class).
- **Static DC Removal:** Subtracts temporal mean per subcarrier ($\tilde{H}(f, t) = H(f, t) - \frac{1}{T}\sum_{\tau} H(f, \tau)$) to eliminate stationary room multipath. Normalization statistics are fitted **only** on development data.

### Step 2: Task 1 — Room Routing ("Where Am I?")
- **Concept:** Static frequency response $[\bar{\mathbf{x}}, \boldsymbol{\sigma}] \in \mathbb{R}^{512}$ captures unique room multipath geometry.
- **Result:** **100.00% validation accuracy**, Macro-F1: 1.0000, ROC-AUC: 1.0000.
- **Window Length:** Reaches **99.81% accuracy** at only $T = 10$ packets (~300 ms), and **100.00%** for all $T \ge 50$.

### Step 3: Task 2 — Self-Supervised Learning ("Learn Without Labels")
- **Architecture:** 1D Residual Convolutional Network (`CSIEncoder`) with 3 residual stages and a 128-dimensional embedding bottleneck.
- **Pre-training:** Contrastive SimCLR (NT-Xent loss, $\tau = 0.1$) trained on 4,154 unlabeled development windows using subcarrier frequency masking (15%), amplitude scaling ($0.85-1.15$), Gaussian noise, and temporal shifts.
- **Linear Probe:** Frozen representations achieve **0.7256 Macro-F1** on Room A and **0.6702 Macro-F1** on Room B (+39.44% higher than untrained random backbones).

### Step 4: Task 3 — Cross-Room Few-Shot Adaptation ("New Room, Few Labels")
- **Transfer Protocol:**
  - Shared activities ($\text{SIT}, \text{STAND}, \text{WALK}$): **0 target labels**. Prototypes transferred from source room.
  - Target-exclusive activities: Learned from $K \in \{1, 5, 10, 20\}$ support captures.
- **Domain Mean Alignment:** Aligns source prototypes into the target feature space ($\tilde{\mathbf{p}}_c^{\text{src}} = \mathbf{p}_c^{\text{src}} - \boldsymbol{\mu}_{\text{src}} + \boldsymbol{\mu}_{\text{tgt}}$), producing a **$4\times$ improvement** on transferred shared activities ($0.0561 \rightarrow 0.2064$).
- **Routing Integration:** Autonomous routing using Task 1 matches Oracle routing with **100.00% agreement**.

---

## 3. Quantitative Results Summary

### Task 1: Room Routing Accuracy vs Window Length
| Window Length $T$ | Accuracy | Precision | Recall | Macro-F1 |
| :---: | :---: | :---: | :---: | :---: |
| $T = 10$ (~300 ms) | 99.81% | 0.9986 | 0.9966 | 0.9976 |
| $T = 25$ (~750 ms) | 99.81% | 0.9986 | 0.9966 | 0.9976 |
| $T \ge 50$ (1.5 s+) | **100.00%** | **1.0000** | **1.0000** | **1.0000** |

### Task 2: Linear Probe Activity Recognition (Frozen Embeddings)
| Evaluation Split | Macro-F1 | Weighted F1 | Accuracy |
| :--- | :---: | :---: | :---: |
| **Room A Validation (6 classes)** | **0.7256** | 0.7268 | 72.79% |
| **Room B Validation (5 classes)** | **0.6702** | 0.6710 | 67.20% |
| **Combined Validation (8 classes)** | **0.6436** | 0.6455 | 64.75% |

### Task 3: Cross-Room Few-Shot Label Efficiency Curves
| Transfer Direction | Metric | $K = 1$ | $K = 5$ | $K = 10$ | $K = 20$ |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **$A \rightarrow B$** (Target: Room B) | Target-Exclusive Macro-F1 | $0.3098 \pm 0.046$ | **$0.5166 \pm 0.028$** | $0.5546 \pm 0.037$ | **$0.5749 \pm 0.025$** |
| | Overall Target Macro-F1 | $0.2210 \pm 0.026$ | $0.2743 \pm 0.007$ | $0.2827 \pm 0.014$ | $0.2854 \pm 0.021$ |
| **$B \rightarrow A$** (Target: Room A) | Target-Exclusive Macro-F1 | $0.1540 \pm 0.047$ | **$0.3458 \pm 0.053$** | $0.3841 \pm 0.051$ | **$0.3852 \pm 0.048$** |
| | Overall Target Macro-F1 | $0.1174 \pm 0.020$ | $0.1852 \pm 0.019$ | $0.2006 \pm 0.012$ | $0.2114 \pm 0.015$ |

---

## 4. How to Run the Project (Step by Step)

### Quick Start (Run Entire Pipeline at Once)
```powershell
cd D:\ubi\p2
.\venv\python.exe main.py
```

### Run Step by Step
```powershell
cd D:\ubi\p2

# 1. Evaluate Task 1 (Room Routing & Window Length Analysis)
.\venv\python.exe task1_routing.py

# 2. Evaluate Task 2 (Self-Supervised Learning & Linear Probe)
.\venv\python.exe run_task2.py

# 3. Evaluate Task 3 (Few-Shot Transfer across K=1, 5, 10, 20)
.\venv\python.exe run_task3.py

# 4. Run Ablation Studies (Normalization, Representations, Alignment)
.\venv\python.exe run_ablations.py

# 5. Generate All Publication-Grade Figures
.\venv\python.exe generate_all_figures.py
```

---

## 5. Repository File Map

```
D:\ubi\p2\
├── main.py                     # Master CLI pipeline orchestrator
├── preprocessing.py            # Data loading, windowing, splitting, normalization
├── preprocess_and_cache.py     # High-speed data caching script
├── task1_routing.py            # Task 1 Room Router implementation
├── task2_ssl.py                # Task 2 SimCLR SSL encoder & linear probe
├── task3_adaptation.py         # Task 3 Prototypical few-shot adaptation
├── run_task2.py                # Standalone Task 2 runner
├── run_task3.py                # Standalone Task 3 runner
├── run_ablations.py            # Full ablation study suite
├── generate_all_figures.py     # Publication figure generator
├── requirement.txt             # Python dependencies
├── REPORT.md                   # Detailed technical report
│
├── data/                       # Raw input data (Experiment-2, Experiment-3)
├── models/
│   └── ssl_encoder.pth         # Trained weights for CSIEncoder
├── processed_data/             # Preprocessed .npz arrays for instant loading
├── results/                    # Quantitative experiment JSON logs
└── figures/                    # Generated high-resolution publication plots
    ├── fig1_preprocessing_pipeline.png
    ├── fig2_room_acc_vs_window_length.png
    ├── fig3_umap_room_and_activity.png
    ├── fig4_label_efficiency_curve.png
    ├── fig5_confusion_matrices_k5.png
    ├── fig6_oracle_vs_predicted_routing.png
    └── fig7_ablation_study.png
```
