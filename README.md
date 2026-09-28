# 🎓 Student Performance Prediction — Deep Learning Assignment

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-orange?logo=tensorflow)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red?logo=streamlit)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-green?logo=scikit-learn)

A complete deep learning project that predicts student academic outcomes (Pass/Fail) using a feedforward neural network.

---

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate dataset
python src/generate_data.py

# 3. Train model (generates all graphs and metrics)
python src/train.py

# 4. Launch web dashboard
streamlit run app.py
```

---

## 📋 Project Overview

| Item | Detail |
|------|--------|
| **Task** | Binary Classification (Pass / Fail) |
| **Dataset** | 1,200 synthetic student records |
| **Features** | 5 (attendance, marks, assignment, study hours, prev. performance) |
| **Architecture** | 5 → 32 → 16 → 2 (MLP) |
| **Hidden Activation** | ReLU |
| **Output Activation** | Softmax |
| **Loss Function** | Categorical Cross-Entropy |
| **Optimizer** | SGD (lr=0.01, momentum=0.9) |
| **Total Parameters** | 754 |

---

## 🗂️ Project Structure

```
DL-A1/
├── data/
│   └── student_performance.csv      # Generated dataset (1200 records)
├── models/
│   ├── student_performance_model.keras  # Saved trained model
│   └── scaler.pkl                   # Fitted StandardScaler
├── outputs/
│   ├── metrics.json                 # Test metrics (loss, acc, F1, etc.)
│   └── figures/
│       ├── training_loss.png        # Training loss curve
│       ├── validation_loss.png      # Validation loss curve
│       ├── training_accuracy.png    # Training accuracy curve
│       ├── validation_accuracy.png  # Validation accuracy curve
│       ├── training_curves.png      # Combined train/val curves
│       └── confusion_matrix.png     # Test set confusion matrix
├── src/
│   ├── generate_data.py             # Synthetic data generator
│   └── train.py                     # Complete training pipeline
├── docs/
│   ├── assignment_report.md         # Full university report (1000+ words)
│   └── viva_questions.md            # 25 viva Q&A
├── app.py                           # Streamlit web dashboard
├── requirements.txt
└── README.md
```

---

## 🧠 Model Architecture

```
Input (5 features)
      │
Dense(32, ReLU)    ← 192 parameters
      │
Dense(16, ReLU)    ← 528 parameters
      │
Dense(2, Softmax)  ← 34 parameters
      │
Output: [P(Fail), P(Pass)]

Total: 754 trainable parameters
```

---

## 📊 Training Configuration

```python
Optimizer:      SGD(lr=0.01, momentum=0.9)
Loss:           Categorical Cross-Entropy
Epochs:         150 (with Early Stopping, patience=15)
Batch Size:     32
Split:          70% Train / 15% Val / 15% Test
Preprocessing:  StandardScaler (fit on train only)
```

---

## 📈 Results (Actual Measured Test Metrics)

The model was evaluated on the held-out test set ($N=225$ samples, 15% stratified test split).

| Metric | Measured Value | Percentage |
|--------|----------------|------------|
| **Test Loss** | `0.1608` | — |
| **Test Accuracy** | `0.9422` | **94.22%** |
| **Precision (Pass)** | `0.9426` | **94.26%** |
| **Recall (Pass)** | `0.9504` | **95.04%** |
| **F1-Score (Pass)** | `0.9465` | **94.65%** |
| **Precision (Fail)** | `0.9417` | **94.17%** |
| **Recall (Fail)** | `0.9327` | **93.27%** |
| **Macro Average F1** | `0.9419` | **94.19%** |
| **Weighted Average F1** | `0.9422` | **94.22%** |

Full metrics details saved in: [`outputs/metrics.json`](outputs/metrics.json)

---

## 🔮 Standalone CLI Prediction

You can test predictions directly from the command line without opening the browser:

```bash
# Example 1: Passing student
python src/predict.py --attendance 85 --internal 75 --assignment 80 --hours 6 --previous 75

# Example 2: Struggling student
python src/predict.py --attendance 35 --internal 25 --assignment 30 --hours 1 --previous 30
```

---

## 🌐 Streamlit Dashboard Pages

| Page | Content |
|------|---------|
| 🏠 Home / Problem | Problem statement, key concepts (ReLU, Softmax, CCE, SGD, Forward/Backprop) |
| 📊 Dataset | Data preview, statistics, class distribution, feature histograms |
| 🧠 Model Architecture | Layer diagram, parameter count, training config |
| 📈 Training Graphs | All 5 training/validation plots + confusion matrix |
| 📋 Evaluation | Test metrics dashboard, confusion matrix breakdown |
| 🔮 Student Prediction | Interactive sliders → real-time Pass/Fail prediction with probabilities |
| ℹ️ About | Project info, tech stack, how-to-run |

---

## 🛠️ Tech Stack

| Library | Version | Purpose |
|---------|---------|---------|
| TensorFlow | ≥2.13 | Neural network framework |
| Keras | (bundled) | High-level model API |
| Pandas | ≥2.0 | Data loading & manipulation |
| NumPy | ≥1.24 | Numerical operations |
| Scikit-learn | ≥1.3 | Preprocessing, metrics |
| Matplotlib | ≥3.7 | Plot generation |
| Seaborn | ≥0.12 | Confusion matrix heatmap |
| Streamlit | ≥1.28 | Interactive web dashboard |
| Joblib | ≥1.3 | Model/scaler serialization |

---

## 📖 Documentation

- **Assignment Report:** [`docs/assignment_report.md`](docs/assignment_report.md)
- **Viva Q&A (25 questions):** [`docs/viva_questions.md`](docs/viva_questions.md)

---

## 🎓 Academic Information

**Assignment Combination:**
- Hidden Activation: **ReLU**
- Output Activation: **Softmax**
- Loss Function: **Categorical Cross-Entropy**
- Optimizer: **SGD**
