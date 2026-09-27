# Assignment Report: Student Performance Prediction using Deep Learning

**Course:** Deep Learning  
**Assignment:** Binary Classification with Neural Networks  
**Submitted by:** [Your Name] | [Roll Number]  
**Date:** September 2026

---

## 1. Problem Statement

The objective of this assignment is to design, implement, and evaluate a **feedforward deep neural network** that predicts whether a student will **PASS** or **FAIL** an examination based on their academic performance indicators. This is a supervised binary classification problem addressed using a multi-layer perceptron (MLP) built with TensorFlow/Keras.

The assigned combination for this implementation is:
- **Hidden Layer Activation:** ReLU (Rectified Linear Unit)
- **Output Layer Activation:** Softmax
- **Loss Function:** Categorical Cross-Entropy
- **Optimizer:** SGD (Stochastic Gradient Descent) with Momentum

---

## 2. Dataset Description

### 2.1 Source
A synthetic dataset of **1,200 student records** was programmatically generated to simulate realistic academic data. The generation uses controlled random distributions with added Gaussian noise to ensure the problem is non-trivially separable, mimicking real-world complexity.

### 2.2 Features

| Feature | Description | Range | Distribution |
|---------|-------------|-------|--------------|
| `attendance` | Percentage of classes attended | 30 – 100 | Normal(75, 15) |
| `internal_marks` | Marks scored in internal tests | 10 – 100 | Normal(55, 18) |
| `assignment_score` | Score on assignments | 10 – 100 | Normal(65, 15) |
| `study_hours` | Average daily study hours | 0.5 – 12 | Normal(5, 2.5) |
| `prev_performance` | Previous semester percentage | 20 – 100 | Normal(60, 18) |

### 2.3 Target Variable

- **Class 0 (Fail):** Student fails the examination
- **Class 1 (Pass):** Student passes the examination

A weighted composite score determines the label:

```
score = 0.25×(attendance/100) + 0.30×(internal_marks/100)
      + 0.15×(assignment_score/100) + 0.15×(study_hours/12)
      + 0.15×(prev_performance/100)
```

Gaussian noise (σ=0.07) is added to ensure class overlap and realistic ambiguity. Students with a noisy score ≥ 0.52 are labeled Pass (~65% of records), others Fail (~35%).

### 2.4 Class Distribution
- **Pass (1):** ~780 records (~65%)
- **Fail (0):** ~420 records (~35%)

The dataset exhibits mild class imbalance, which is representative of real academic scenarios.

---

## 3. Data Preprocessing

### 3.1 Data Quality Checks
Upon loading, the dataset was verified for:
- **Missing values:** None found across all features
- **Duplicate records:** Removed using `drop_duplicates()`
- **Class distribution:** Verified using `value_counts()`

### 3.2 Train/Validation/Test Split
The dataset was split in a **70/15/15** ratio using stratified splitting to maintain class distribution across all sets:

```
Total: 1200 records
├── Training Set:   840 records (70%)
├── Validation Set: 180 records (15%)
└── Test Set:       180 records (15%)
```

Stratified splitting ensures each subset has proportional Pass/Fail samples, preventing bias due to random ordering.

### 3.3 Feature Scaling
**StandardScaler** from scikit-learn was applied:
- Fitted **only on training data** to prevent data leakage
- Applied (transform-only) to validation and test sets
- Transforms features to zero mean and unit variance: `z = (x - μ) / σ`
- Scaler serialized to `models/scaler.pkl` for inference

### 3.4 Label Encoding
Target labels were **one-hot encoded** using `to_categorical()`:
- Fail (0) → `[1, 0]`
- Pass (1) → `[0, 1]`

This is required by the Softmax output with Categorical Cross-Entropy loss.

---

## 4. Model Architecture

### 4.1 Network Design

The model is a fully-connected feedforward neural network (MLP) with the following architecture:

```
┌─────────────────────────────────────────┐
│  INPUT LAYER      5 neurons             │
│  (5 features)                           │
└──────────────────┬──────────────────────┘
                   │ Weighted connections
┌──────────────────▼──────────────────────┐
│  HIDDEN LAYER 1   32 neurons  [ReLU]    │
│  Parameters: 5×32 + 32 = 192           │
└──────────────────┬──────────────────────┘
                   │
┌──────────────────▼──────────────────────┐
│  HIDDEN LAYER 2   16 neurons  [ReLU]    │
│  Parameters: 32×16 + 16 = 528          │
└──────────────────┬──────────────────────┘
                   │
┌──────────────────▼──────────────────────┐
│  OUTPUT LAYER     2 neurons  [Softmax]  │
│  Parameters: 16×2 + 2 = 34             │
└─────────────────────────────────────────┘

Total Trainable Parameters: 754
```

### 4.2 Activation Functions

**ReLU (Hidden Layers):**
$$f(x) = \max(0, x)$$

ReLU was chosen for hidden layers because:
- Computationally efficient (simple thresholding operation)
- Alleviates the vanishing gradient problem present in sigmoid/tanh
- Produces sparse activations, making the network efficient
- Empirically performs well in classification tasks

**Softmax (Output Layer):**
$$\sigma(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}$$

Softmax was chosen for the output because:
- Converts raw logits to a proper probability distribution
- All output values sum to 1.0
- Naturally extends to multi-class problems
- Compatible with Categorical Cross-Entropy loss

### 4.3 Loss Function

**Categorical Cross-Entropy:**
$$L = -\sum_{i=1}^{C} y_i \log(\hat{y}_i)$$

Where $y_i$ is the one-hot encoded true label and $\hat{y}_i$ is the predicted probability. This loss penalizes confident wrong predictions heavily and is the standard choice for multi-class classification with Softmax.

### 4.4 Optimizer

**SGD with Momentum:**
$$v_t = \beta v_{t-1} + \eta \nabla_\theta J(\theta)$$
$$\theta = \theta - v_t$$

Configuration: `lr=0.01, momentum=0.9`

SGD with momentum was selected because:
- Classic, well-understood optimizer with strong theoretical guarantees
- Momentum (β=0.9) helps escape local minima and accelerates convergence
- Learning rate of 0.01 provides stable gradient updates for this dataset size
- Avoids the adaptive learning rate complications of Adam/RMSprop

---

## 5. Training Procedure

### 5.1 Hyperparameters

| Hyperparameter | Value | Rationale |
|----------------|-------|-----------|
| Maximum Epochs | 150 | Sufficient for convergence |
| Batch Size | 32 | Balance between speed and gradient noise |
| Learning Rate | 0.01 | Standard starting point for SGD |
| Momentum | 0.9 | Standard momentum value |
| Early Stopping Patience | 15 | Prevents overfitting |
| Monitor Metric | val_loss | Generalization performance |

### 5.2 Early Stopping
Early stopping with `patience=15` and `restore_best_weights=True` was applied:
- Monitors validation loss each epoch
- Stops training if no improvement after 15 consecutive epochs
- Restores the model weights from the best epoch

### 5.3 Training Loop
Each epoch:
1. Forward pass: input → hidden layers (ReLU) → output (Softmax)
2. Loss computation: Categorical Cross-Entropy
3. Backward pass: Gradients computed via backpropagation
4. Weight update: SGD with momentum
5. Validation evaluation on held-out validation set

---

## 6. Results

### 6.1 Test Set Metrics

The model was evaluated on the held-out test set ($N = 180$ students). The actual measured performance metrics are as follows:

| Metric | Measured Value | Percentage |
|--------|----------------|------------|
| **Test Loss** | **0.4028** | — |
| **Test Accuracy** | **0.8278** | **82.78%** |
| **Precision (Pass)** | **0.8357** | **83.57%** |
| **Recall (Pass)** | **0.9360** | **93.60%** |
| **F1-Score (Pass)** | **0.8830** | **88.30%** |
| **Precision (Fail)** | **0.8000** | **80.00%** |
| **Recall (Fail)** | **0.5818** | **58.18%** |
| **F1-Score (Fail)** | **0.6737** | **67.37%** |
| **Macro Average F1** | **0.7784** | **77.84%** |
| **Weighted Average F1** | **0.8191** | **81.91%** |

#### Detailed Classification Report:
```
              precision    recall  f1-score   support

        Fail       0.80      0.58      0.67        55
        Pass       0.84      0.94      0.88       125

    accuracy                           0.83       180
   macro avg       0.82      0.76      0.78       180
weighted avg       0.82      0.83      0.82       180
```

### 6.2 Training Observations

The training curves (generated and saved in `outputs/figures/`) show:
- **Convergence:** Training commenced with a loss of 0.6473 and reached 0.4059. Validation loss converged to 0.4196 at epoch 8.
- **Early Stopping:** Training triggered early stopping at Epoch 23 with `patience=15`, successfully preventing overfitting and restoring the optimal weights from Epoch 8.
- **Generalization:** The close proximity between training loss (0.4059) and test loss (0.4028) confirms absence of overfitting and excellent generalization on unseen data.

### 6.3 Confusion Matrix Analysis

On the 180 test samples:
- **True Positives (TP): 117** students correctly predicted as **PASS**
- **True Negatives (TN): 32** students correctly predicted as **FAIL**
- **False Positives (FP): 23** students incorrectly predicted as PASS (actually FAIL)
- **False Negatives (FN): 8** students incorrectly predicted as FAIL (actually PASS)

Total correctly classified: $117 + 32 = 149$ out of 180 (**82.78% accuracy**). The high recall on the Pass class (93.6%) ensures almost all successful students are recognized, while the precision of 80.0% on Fail ensures high reliability when alerting struggling students.

---

## 7. Graphs and Visualizations

The following graphs were generated and saved to `outputs/figures/`:

1. **training_loss.png** — Training loss per epoch; shows model learning progression
2. **validation_loss.png** — Validation loss per epoch; monitors generalization
3. **training_accuracy.png** — Training accuracy per epoch
4. **validation_accuracy.png** — Validation accuracy per epoch
5. **training_curves.png** — Combined plot of train/val loss and accuracy
6. **confusion_matrix.png** — Heatmap of test set predictions vs actuals

---

## 8. Challenges and Solutions

| Challenge | Solution |
|-----------|----------|
| Class imbalance (65/35) | Used stratified splitting to maintain proportions |
| Data leakage risk | Scaler fitted only on training data |
| Overfitting risk | Early stopping with patience=15, restore_best_weights |
| Non-interactive environment | Used `matplotlib.use('Agg')` backend |
| Reproducibility | Fixed `numpy` and `tensorflow` seeds to 42 |

---

## 9. Conclusion

This assignment successfully implemented a feedforward neural network for binary student performance classification. The model architecture (5→32→16→2) with ReLU activations, Softmax output, Categorical Cross-Entropy loss, and SGD optimizer demonstrates:

1. **Effective learning:** Training and validation losses converge smoothly
2. **Good generalization:** Minimal gap between training and validation metrics
3. **Practical utility:** The Streamlit dashboard enables real-time student-level predictions
4. **Sound engineering:** Proper preprocessing (StandardScaler, stratified split, one-hot encoding) prevents common pitfalls

The project pipeline is fully reproducible: from synthetic data generation through preprocessing, model training, evaluation, visualization, to a deployed interactive web application.

---

## 10. References

1. LeCun, Y., Bengio, Y., & Hinton, G. (2015). Deep learning. *Nature*, 521, 436–444.
2. Glorot, X., Bordes, A., & Bengio, Y. (2011). Deep sparse rectifier neural networks. *AISTATS*.
3. Chollet, F. (2021). *Deep Learning with Python* (2nd ed.). Manning Publications.
4. TensorFlow Documentation: https://www.tensorflow.org/api_docs
5. Scikit-learn Documentation: https://scikit-learn.org/stable/
