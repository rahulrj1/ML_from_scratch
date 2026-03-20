"""
Topic 1.3: Logistic Regression
=================================
Binary classification, sigmoid, and decision boundaries.
Same training loop as linear regression — but now we predict
a CLASS instead of a number.

KEY TAKEAWAYS:
  - Logistic regression = linear regression + sigmoid → outputs probability.
  - Sigmoid squashes any number into (0, 1) → interpret as P(class = 1).
  - Loss = binary cross-entropy (from Topic 0.5), NOT MSE.
  - Decision boundary: predict 1 if P > 0.5, else predict 0.
  - Training loop is identical: predict → loss → gradient → update.
"""

import numpy as np

# ============================================================
# FROM REGRESSION TO CLASSIFICATION
# ============================================================

print("=== FROM REGRESSION TO CLASSIFICATION ===")
print()
print("Linear regression:   y = w·x + b  → output is any number")
print("Logistic regression: y = sigmoid(w·x + b)  → output is between 0 and 1")
print()
print("That's the ONLY difference. We wrap the output in sigmoid.")
print("Now the output is a probability: P(class = 1).")
print()

# ============================================================
# SIGMOID — THE SQUASHING FUNCTION
# ============================================================

print("=== SIGMOID ===")

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

print("sigmoid(z) = 1 / (1 + e^(-z))")
print()
print("It takes ANY number and squashes it to (0, 1):")
print()
for z in [-10, -5, -2, -1, 0, 1, 2, 5, 10]:
    s = sigmoid(z)
    bar = "█" * int(s * 30)
    print(f"  sigmoid({z:3d}) = {s:.4f}  {bar}")
print()
print("Big negative → close to 0")
print("Zero         → exactly 0.5")
print("Big positive → close to 1")
print()

# ============================================================
# STEP 1: CREATE CLASSIFICATION DATA
# ============================================================

print("=== STEP 1: CREATE DATA ===")

np.random.seed(42)

# Two clusters of points
n_per_class = 50
n_samples = n_per_class * 2

# Class 0: centered around (-1, -1)
X_class0 = np.random.randn(n_per_class, 2) * 0.8 + np.array([-1, -1])
# Class 1: centered around (1, 1)
X_class1 = np.random.randn(n_per_class, 2) * 0.8 + np.array([1, 1])

X = np.vstack([X_class0, X_class1])
y = np.array([0] * n_per_class + [1] * n_per_class, dtype=float)

print(f"Dataset: {n_samples} samples, 2 features, 2 classes")
print(f"  Class 0: {n_per_class} samples (centered around [-1, -1])")
print(f"  Class 1: {n_per_class} samples (centered around [+1, +1])")
print()
print("First 3 from each class:")
for i in [0, 1, 2]:
    print(f"  Class 0:  x = [{X_class0[i][0]:+.2f}, {X_class0[i][1]:+.2f}]  y = 0")
for i in [0, 1, 2]:
    print(f"  Class 1:  x = [{X_class1[i][0]:+.2f}, {X_class1[i][1]:+.2f}]  y = 1")
print()

# ============================================================
# STEP 2: SPLIT THE DATA
# ============================================================

print("=== STEP 2: SPLIT THE DATA ===")

indices = np.random.permutation(n_samples)
train_end = int(n_samples * 0.7)
val_end = int(n_samples * 0.85)

X_train, y_train = X[indices[:train_end]], y[indices[:train_end]]
X_val, y_val = X[indices[train_end:val_end]], y[indices[train_end:val_end]]
X_test, y_test = X[indices[val_end:]], y[indices[val_end:]]

print(f"  Train: {len(X_train)} samples")
print(f"  Val:   {len(X_val)} samples")
print(f"  Test:  {len(X_test)} samples")
print()

# ============================================================
# STEP 3: THE MODEL — PREDICT PROBABILITIES
# ============================================================

print("=== STEP 3: THE MODEL ===")
print()
print("Logistic regression does TWO things:")
print("  1. Compute a raw score (logit):  z = w·x + b")
print("  2. Squash it to a probability:   p = sigmoid(z)")
print()

w = np.zeros(2)  # two features → two weights
b = 0.0

# With w=0, b=0, sigmoid(0) = 0.5 for everything
z = X_train @ w + b
probs = sigmoid(z)

print(f"Starting weights: w = {w}, b = {b}")
print(f"With w=0, b=0 → sigmoid(0) = 0.5 for every sample")
print(f"  First 5 predictions: {probs[:5].round(4)}")
print(f"  The model is saying '50/50' for everything. Useless.")
print()

# ============================================================
# STEP 4: THE LOSS — BINARY CROSS-ENTROPY
# ============================================================

print("=== STEP 4: BINARY CROSS-ENTROPY LOSS ===")
print()
print("From Topic 0.5:")
print("  BCE = -(1/N) × Σ [ y·log(p) + (1-y)·log(1-p) ]")
print()
print("Remember: this is just -log(correct probability) averaged over all samples.")
print()

def binary_cross_entropy(probs, targets):
    eps = 1e-15
    probs = np.clip(probs, eps, 1 - eps)
    return -np.mean(targets * np.log(probs) + (1 - targets) * np.log(1 - probs))

loss = binary_cross_entropy(probs, y_train)
print(f"  Loss with w=0, b=0: {loss:.4f}")
print(f"  (For reference, -log(0.5) = {-np.log(0.5):.4f} — every sample contributes this)")
print()

# Baseline: always predict the most common class
majority_class = 1 if y_train.mean() > 0.5 else 0
baseline_acc = max(y_train.mean(), 1 - y_train.mean())
print(f"  Baseline (always predict class {majority_class}): accuracy = {baseline_acc:.1%}")
print()

# ============================================================
# STEP 5: THE GRADIENT
# ============================================================

print("=== STEP 5: THE GRADIENT ===")
print()
print("Here's where the 'beautiful gradient' from Topic 0.5 shows up.")
print()
print("The gradient of BCE w.r.t. weights is:")
print("  d(loss)/dw = (1/N) × X.T @ (predictions - targets)")
print("  d(loss)/db = (1/N) × sum(predictions - targets)")
print()
print("That's it. Same (predicted - actual) idea.")
print()

errors = probs - y_train  # predicted - actual
grad_w = (1 / len(X_train)) * X_train.T @ errors
grad_b = (1 / len(X_train)) * np.sum(errors)

print(f"  Predictions are all 0.5, targets are mix of 0s and 1s")
print(f"  Gradient for w: {grad_w.round(4)}")
print(f"  Gradient for b: {grad_b:.4f}")
print()

# ============================================================
# STEP 6: FULL TRAINING LOOP
# ============================================================

print("=== STEP 6: TRAINING LOOP ===")
print()

w = np.zeros(2)
b = 0.0
learning_rate = 0.5
n_epochs = 100

for epoch in range(n_epochs):
    # Predict
    z_train = X_train @ w + b
    probs_train = sigmoid(z_train)

    # Loss
    train_loss = binary_cross_entropy(probs_train, y_train)

    # Val loss
    probs_val = sigmoid(X_val @ w + b)
    val_loss = binary_cross_entropy(probs_val, y_val)

    # Gradient
    errors = probs_train - y_train
    grad_w = (1 / len(X_train)) * X_train.T @ errors
    grad_b = (1 / len(X_train)) * np.sum(errors)

    # Update
    w -= learning_rate * grad_w
    b -= learning_rate * grad_b

    if epoch % 20 == 0 or epoch == n_epochs - 1:
        preds_train = (probs_train > 0.5).astype(int)
        acc = np.mean(preds_train == y_train)
        print(f"  Epoch {epoch:3d}: loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
              f"train_acc={acc:.1%}  w={w.round(3)}")

print()
print(f"Learned weights: w = {w.round(4)}, b = {b:.4f}")
print()

# ============================================================
# STEP 7: THE DECISION BOUNDARY
# ============================================================

print("=== STEP 7: DECISION BOUNDARY ===")
print()
print("The model predicts class 1 when P > 0.5")
print("  P = 0.5 when sigmoid(z) = 0.5, which is when z = 0")
print("  z = w1·x1 + w2·x2 + b = 0")
print()
print(f"Our boundary: {w[0]:.3f}·x1 + {w[1]:.3f}·x2 + {b:.3f} = 0")

if abs(w[1]) > 0.001:
    print(f"  Solving for x2: x2 = ({-w[0]:.3f}·x1 + {-b:.3f}) / {w[1]:.3f}")
print()
print("Points above this line → class 1")
print("Points below this line → class 0")
print("The model learned a straight line that separates the two classes!")
print()

# ============================================================
# STEP 8: EVALUATE
# ============================================================

print("=== STEP 8: EVALUATE ===")
print()

def evaluate(X_set, y_set, w, b, set_name):
    probs = sigmoid(X_set @ w + b)
    preds = (probs > 0.5).astype(int)
    accuracy = np.mean(preds == y_set)
    loss = binary_cross_entropy(probs, y_set)
    print(f"  {set_name:5s}: accuracy = {accuracy:.1%}  loss = {loss:.4f}")
    return accuracy

train_acc = evaluate(X_train, y_train, w, b, "Train")
val_acc = evaluate(X_val, y_val, w, b, "Val")
test_acc = evaluate(X_test, y_test, w, b, "Test")
print()

print(f"  Baseline was: {baseline_acc:.1%}")
print(f"  We got:       {test_acc:.1%}")
print(f"  Beat baseline? {'YES' if test_acc > baseline_acc else 'NO'}")
print()

# Show some predictions
print("Sample predictions:")
probs_test = sigmoid(X_test @ w + b)
for i in range(min(8, len(X_test))):
    pred = int(probs_test[i] > 0.5)
    conf = probs_test[i] if pred == 1 else 1 - probs_test[i]
    correct = "✓" if pred == y_test[i] else "✗"
    print(f"  x=[{X_test[i][0]:+.2f}, {X_test[i][1]:+.2f}]  "
          f"P(class1)={probs_test[i]:.3f}  pred={pred}  actual={int(y_test[i])}  "
          f"{correct} ({conf:.0%} confident)")
print()

# ============================================================
# LINEAR REGRESSION vs LOGISTIC REGRESSION
# ============================================================

print("=== COMPARISON ===")
print()
print("  ┌──────────────────┬─────────────────────┬─────────────────────────┐")
print("  │                  │ Linear Regression    │ Logistic Regression     │")
print("  ├──────────────────┼─────────────────────┼─────────────────────────┤")
print("  │ Predicts         │ A number             │ A probability (0 to 1)  │")
print("  │ Output           │ w·x + b              │ sigmoid(w·x + b)        │")
print("  │ Loss function    │ MSE                  │ Binary cross-entropy    │")
print("  │ Use case         │ Regression           │ Classification          │")
print("  │ Training loop    │ Same                 │ Same                    │")
print("  └──────────────────┴─────────────────────┴─────────────────────────┘")
print()
print("The training loop (predict → loss → gradient → update) is IDENTICAL.")
print("The only differences: sigmoid on the output, and cross-entropy for the loss.")
