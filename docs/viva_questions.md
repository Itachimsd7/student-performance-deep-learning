# Viva Questions & Answers — Deep Learning Assignment

**Topic:** Student Performance Prediction | Feedforward Neural Network  
**Activation:** ReLU | **Output:** Softmax | **Loss:** Categorical Cross-Entropy | **Optimizer:** SGD

---

## Q1. What is ReLU activation and why was it used in hidden layers?

**Answer:**  
ReLU (Rectified Linear Unit) is defined as **f(x) = max(0, x)**. It outputs 0 for negative inputs and passes positive inputs unchanged.

**Reasons for use:**
- Solves the **vanishing gradient problem** (unlike sigmoid/tanh which saturate)
- Computationally cheap — just a threshold operation
- Produces **sparse activations** (neurons either fire or don't), improving efficiency
- Empirically outperforms sigmoid/tanh in deep networks

---

## Q2. What is Softmax activation and why is it used in the output layer?

**Answer:**  
Softmax is defined as: **σ(zᵢ) = eᶻⁱ / Σⱼ eᶻʲ**

It converts a vector of raw scores (logits) into a **probability distribution** where all values sum to 1.0.

**Reasons for output layer:**
- Provides interpretable class probabilities
- Naturally handles multi-class problems
- Pairs perfectly with Categorical Cross-Entropy loss
- In our model: outputs [P(Fail), P(Pass)]

---

## Q3. What is Categorical Cross-Entropy loss? Write the formula.

**Answer:**  
**L = -Σᵢ yᵢ × log(ŷᵢ)**

Where yᵢ is the one-hot encoded true label and ŷᵢ is the predicted probability.

- It penalizes predictions that are confident but wrong (log of small probability → large loss)
- Works with Softmax because Softmax outputs valid probabilities (0 < p < 1, sum = 1)
- For binary: if true class is Pass [0,1] and model predicts [0.1, 0.9], loss = -log(0.9) ≈ 0.105

---

## Q4. How does SGD work? What is the update rule?

**Answer:**  
SGD (Stochastic Gradient Descent) updates parameters by:

**θ = θ − η × ∇θ J(θ)**

Where η is the learning rate and ∇θ J(θ) is the gradient of the loss w.r.t. parameters.

In our model: `SGD(lr=0.01, momentum=0.9)`

**With Momentum:**  
- v_t = β×v_{t-1} + η×∇θ  
- θ = θ − v_t  
- Momentum (β=0.9) accumulates past gradients to accelerate and smooth updates

---

## Q5. What is Forward Propagation? Explain step by step.

**Answer:**  
Forward propagation computes the predicted output from inputs:

1. **Layer 1:** Z₁ = X × W₁ + b₁ → A₁ = ReLU(Z₁)  [shape: (N,32)]
2. **Layer 2:** Z₂ = A₁ × W₂ + b₂ → A₂ = ReLU(Z₂) [shape: (N,16)]
3. **Output:**  Z₃ = A₂ × W₃ + b₃ → ŷ = Softmax(Z₃) [shape: (N,2)]
4. **Loss:** L = CategoricalCrossEntropy(y, ŷ)

---

## Q6. What is Backpropagation? How does it work?

**Answer:**  
Backpropagation computes gradients of the loss with respect to each weight using the **chain rule**, then passes them backward through the network.

**Steps:**
1. Compute ∂L/∂ŷ (gradient at output)
2. Compute ∂L/∂W₃, ∂L/∂b₃ (output layer gradients)
3. Propagate back: ∂L/∂A₂ → ∂L/∂Z₂ (through ReLU derivative)
4. Compute ∂L/∂W₂, ∂L/∂b₂
5. Repeat back to layer 1
6. SGD uses these gradients to update W and b

---

## Q7. What is One-Hot Encoding and why is it needed?

**Answer:**  
One-hot encoding converts categorical labels into binary vectors:
- Fail (0) → **[1, 0]**
- Pass (1) → **[0, 1]**

**Why needed:**
- Categorical Cross-Entropy requires probability vectors, not scalar labels
- Softmax outputs a vector — we need the same format for loss computation
- Prevents the model from interpreting class labels as ordinal values (0 < 1)

---

## Q8. What is StandardScaler? Why apply it only on training data?

**Answer:**  
StandardScaler standardizes features: **z = (x − μ) / σ**

Each feature is transformed to have **zero mean and unit variance**.

**Only fit on training data** to prevent **data leakage**:
- If fitted on all data, μ and σ incorporate information from validation/test sets
- This would give the model indirect access to test data during training
- Correct practice: `fit_transform(X_train)` then `transform(X_val)`, `transform(X_test)`

---

## Q9. What is the train/val/test split and why use 70/15/15?

**Answer:**
- **Training (70%):** Model learns weights from this data
- **Validation (15%):** Monitors generalization during training (for early stopping, hyperparameter tuning)
- **Test (15%):** Final unbiased evaluation — never seen during training or tuning

**70/15/15** is standard for moderately-sized datasets (1200 records here). It provides enough training data while maintaining statistically meaningful evaluation sets (~180 samples each).

---

## Q10. What is Early Stopping and how does it prevent overfitting?

**Answer:**  
Early stopping monitors a metric (val_loss) during training and halts when it stops improving for `patience` consecutive epochs.

In our model: `EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)`

**How it prevents overfitting:**
- Training loss always decreases (model memorizes training data)
- Validation loss eventually starts rising (overfitting begins)
- We stop at the point where val_loss is minimum and restore those weights
- Avoids unnecessary computation and prevents the model from memorizing noise

---

## Q11. What is overfitting and how can we detect it?

**Answer:**  
Overfitting occurs when a model performs well on training data but poorly on unseen data — it memorizes noise instead of learning patterns.

**Detection:**
- Large gap between training accuracy and validation accuracy
- Training loss decreasing while validation loss increasing
- High training F1 but low test F1

**Solutions:** Early stopping, dropout, L2 regularization, more data, simpler model

---

## Q12. Why SGD over Adam for this assignment?

**Answer:**  
- SGD has **stronger theoretical convergence guarantees** and often generalizes better than adaptive methods
- SGD is the **classic optimizer** with well-understood behavior — appropriate for academic study
- Adam adapts learning rates per parameter, can overfit on small datasets
- With momentum (β=0.9), SGD effectively navigates loss landscapes
- Many state-of-the-art models in vision use SGD + momentum for final training

---

## Q13. Why Softmax over Sigmoid for the output layer?

**Answer:**  
Although this is binary classification, we use **2-class Softmax** (instead of 1-neuron Sigmoid) because:

1. **Consistency with the assignment:** Categorical Cross-Entropy requires a probability distribution
2. **Softmax generalizes to multi-class** — same architecture works for K classes
3. Softmax ensures probabilities for both classes **sum to 1.0**, providing clearer interpretability
4. Sigmoid on a single neuron would require Binary Cross-Entropy loss — a different setup

*Note: Sigmoid (1 neuron) is mathematically equivalent to 2-class Softmax for binary problems.*

---

## Q14. What is the Confusion Matrix? Explain each quadrant.

**Answer:**  

```
                Predicted Fail    Predicted Pass
Actual Fail  |  TN (True Neg)  |  FP (False Pos) |
Actual Pass  |  FN (False Neg) |  TP (True Pos)  |
```

- **TN:** Model correctly said "Fail" — student actually failed
- **FP (Type I Error):** Model said "Pass" — student actually failed *(false alarm)*
- **FN (Type II Error):** Model said "Fail" — student actually passed *(missed positive)*
- **TP:** Model correctly said "Pass" — student actually passed

---

## Q15. Define Precision, Recall, and F1-Score.

**Answer:**

| Metric | Formula | Meaning |
|--------|---------|---------|
| **Precision** | TP / (TP + FP) | Of all predicted Pass, how many actually passed? |
| **Recall** | TP / (TP + FN) | Of all actual Pass, how many did we catch? |
| **F1-Score** | 2 × (P × R) / (P + R) | Harmonic mean — balances precision and recall |

**In education context:**
- High Recall: We catch most students who would pass (few missed)
- High Precision: When we predict Pass, we're usually right (few false alarms)

---

## Q16. What is Batch Size and how does it affect training?

**Answer:**  
Batch size = number of samples processed before a weight update.

In our model: **batch_size = 32**

| Batch Size | Effect |
|------------|--------|
| Small (8-16) | Noisy gradients, slower but can escape local minima |
| Medium (32-64) | Good balance between noise and efficiency ✓ |
| Large (256+) | Smooth gradients, faster per epoch, may converge to sharp minima |

32 is a standard default that works well for datasets of ~1000 records.

---

## Q17. What is a Learning Rate and what happens if it's too high or too low?

**Answer:**  
Learning rate (η) controls the step size in gradient descent.

In our model: `lr = 0.01`

| Learning Rate | Effect |
|---------------|--------|
| Too high (e.g., 1.0) | Oscillates, loss diverges, model fails to converge |
| Too low (e.g., 0.0001) | Extremely slow convergence, training takes forever |
| Just right (0.01) | Stable convergence, smooth loss curve |

**Tip:** Learning rate is the most critical hyperparameter to tune.

---

## Q18. What is the Vanishing Gradient Problem?

**Answer:**  
In deep networks, gradients are multiplied through many layers during backpropagation. Sigmoid/tanh derivatives are < 1, so gradients shrink exponentially as they propagate backward — becoming effectively zero in early layers. This means early layers learn very slowly or not at all.

**Solution:** ReLU has a derivative of 1 for positive inputs, preventing gradient shrinkage and enabling deeper networks to train effectively.

---

## Q19. What is Momentum in SGD and what does β=0.9 mean?

**Answer:**  
Momentum adds a fraction (β) of the previous update to the current update:

**v_t = 0.9 × v_{t-1} + 0.01 × ∇θ**  
**θ = θ − v_t**

- β=0.9 means 90% of the previous velocity is retained each step
- Accumulates gradients in consistent directions (accelerates)
- Dampens oscillations in inconsistent directions
- Helps escape shallow local minima and saddle points
- Analogous to a ball rolling downhill with inertia

---

## Q20. How many parameters does our model have? Calculate manually.

**Answer:**

| Layer | Formula | Count |
|-------|---------|-------|
| Dense(5→32) | 5×32 + 32 (bias) | **192** |
| Dense(32→16) | 32×16 + 16 (bias) | **528** |
| Dense(16→2) | 16×2 + 2 (bias) | **34** |
| **Total** | | **754 parameters** |

All 754 parameters are trainable. The model is intentionally small to avoid overfitting on 1200 samples.

---

## Q21. Why do we use Stratified Splitting?

**Answer:**  
Stratified splitting ensures each split (train/val/test) has the **same class proportion** as the original dataset.

**Without stratification:** Random chance could create a test set with 80% Pass, making evaluation misleading.

**With stratification (our case, ~65% Pass / ~35% Fail):**
- Train: ~65% Pass / 35% Fail
- Val: ~65% Pass / 35% Fail  
- Test: ~65% Pass / 35% Fail

This ensures fair, unbiased evaluation across all splits.

---

## Q22. What is the purpose of the Validation Set?

**Answer:**  
The validation set is used **during training** (not for final evaluation) to:

1. **Monitor generalization:** Check if the model generalizes beyond training data
2. **Drive early stopping:** `val_loss` determines when to stop
3. **Hyperparameter tuning:** Compare different architectures/learning rates
4. **Detect overfitting:** Rising val_loss while train_loss falls → overfitting

The test set is kept completely separate for **final unbiased evaluation**.

---

## Q23. What is the difference between loss and accuracy as training metrics?

**Answer:**

| Metric | What it measures | Properties |
|--------|-----------------|------------|
| **Loss** | Continuous error signal (CCE value) | Smooth, differentiable — used for optimization |
| **Accuracy** | % of correct predictions | Discrete, non-differentiable — used for reporting |

- **Loss** drives learning (backpropagation minimizes loss)
- **Accuracy** is human-interpretable but cannot be directly optimized
- A model can have improving loss but flat accuracy (when predictions cross decision boundary rarely)

---

## Q24. What is the role of Bias in a neural network?

**Answer:**  
Bias (b) is an additional learnable parameter in each neuron:

**z = W × x + b**

- Bias allows the activation function to be **shifted left or right**
- Without bias, all decision hyperplanes must pass through the origin
- Bias provides the model with **additional flexibility** to fit data
- In our model: each Dense layer has one bias per neuron (32+16+2 = 50 bias parameters)

---

## Q25. How would you improve this model's performance?

**Answer:**  
Potential improvements:

1. **Architecture:** Add more layers or neurons (e.g., 64→32→16→2)
2. **Regularization:** Add Dropout layers (e.g., rate=0.3) to reduce overfitting
3. **Optimizer:** Try Adam with learning rate scheduling
4. **Data augmentation:** Add more realistic noise, generate 5000+ samples
5. **Feature engineering:** Add derived features (e.g., attendance × marks interaction)
6. **Hyperparameter search:** Grid search over lr, batch_size, layer sizes
7. **Class imbalance handling:** Use class_weight parameter in model.fit()
8. **Batch Normalization:** Add after Dense layers for faster convergence
9. **Learning Rate Scheduling:** Reduce lr on plateau for finer convergence
10. **Ensemble:** Train multiple models with different seeds, average predictions
