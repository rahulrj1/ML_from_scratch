"""
Topic 1.2: Linear Regression
===============================
Your first real model — from scratch. Gradient descent, cost functions,
and the full training loop, all demystified.

KEY TAKEAWAYS:
  - Linear regression predicts a number: y = w·x + b (weighted sum + bias).
  - Training = adjust w and b to minimize MSE loss.
  - Gradient tells us which direction to nudge each weight.
  - The update rule: w = w - learning_rate × gradient.
  - This is the foundation of ALL neural network training.
"""

import numpy as np

# ============================================================
# WHAT IS LINEAR REGRESSION?
# ============================================================

print("=== WHAT IS LINEAR REGRESSION? ===")
print()
print("You have data. You want to predict a number from some inputs.")
print()
print("  Example: predict house price from square footage.")
print("    input (x) = 1500 sq ft")
print("    output (y) = $300,000")
print()
print("Linear regression says: the relationship is a straight line.")
print("    y = w × x + b")
print()
print("  w (weight) = how much does price change per sq ft?")
print("  b (bias)   = what's the base price with 0 sq ft?")
print()
print("Training = finding the best w and b from the data.")
print()

# ============================================================
# STEP 1: CREATE SOME DATA
# ============================================================

print("=== STEP 1: CREATE DATA ===")

np.random.seed(42)

# True relationship: y = 3x + 7 (+ some noise)
true_w = 3.0
true_b = 7.0

n_samples = 100
X = np.random.uniform(0, 10, size=n_samples)  # 100 random x values between 0 and 10
noise = np.random.randn(n_samples) * 1.5       # some random noise
y = true_w * X + true_b + noise                 # the actual data

print(f"Generated {n_samples} data points from: y = {true_w}·x + {true_b} + noise")
print()
print("First 5 data points:")
for i in range(5):
    print(f"  x = {X[i]:.2f}  →  y = {y[i]:.2f}")
print()
print("Our job: figure out w ≈ 3.0 and b ≈ 7.0 from JUST the data.")
print("(In real life, we don't know the true w and b. That's the whole point.)")
print()

# ============================================================
# STEP 2: SPLIT THE DATA (from 1.1)
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
# STEP 3: THE MODEL — MAKE A PREDICTION
# ============================================================

print("=== STEP 3: THE MODEL ===")
print()

# Start with random (bad) weights
w = 0.0
b = 0.0

print(f"Our model: y_pred = w × x + b")
print(f"Starting with random weights: w = {w}, b = {b}")
print()

# Make predictions with our bad weights
y_pred = w * X_train + b

print("Predictions with w=0, b=0 (everything is predicted as 0):")
for i in range(3):
    print(f"  x = {X_train[i]:.2f}  →  pred = {y_pred[i]:.2f}  (actual = {y_train[i]:.2f})")
print()
print("Terrible! The model knows nothing yet. Let's fix that.")
print()

# ============================================================
# STEP 4: THE LOSS — HOW WRONG ARE WE?
# ============================================================

print("=== STEP 4: COMPUTE THE LOSS (MSE) ===")

def compute_mse(y_pred, y_actual):
    return np.mean((y_pred - y_actual) ** 2)

loss = compute_mse(y_pred, y_train)
print(f"  MSE loss with w=0, b=0: {loss:.4f}")
print()

# Baseline: always predict the mean
baseline_pred = np.mean(y_train)
baseline_loss = compute_mse(np.full_like(y_train, baseline_pred), y_train)
print(f"  Baseline (always predict mean={baseline_pred:.2f}): MSE = {baseline_loss:.4f}")
print()
print("Both are bad. We need to learn better weights.")
print()

# ============================================================
# STEP 5: THE GRADIENT — WHICH DIRECTION TO NUDGE?
# ============================================================

print("=== STEP 5: COMPUTE THE GRADIENT ===")
print()
print("The gradient answers: 'if I increase w (or b) slightly,")
print("does the loss go up or down, and by how much?'")
print()
print("For MSE loss with y_pred = w·x + b:")
print()
print("  d(loss)/dw = (2/N) × Σ (y_pred - y_actual) × x")
print("  d(loss)/db = (2/N) × Σ (y_pred - y_actual)")
print()
print("Let's compute them step by step:")
print()

errors = y_pred - y_train  # how wrong each prediction is

grad_w = (2 / len(X_train)) * np.sum(errors * X_train)
grad_b = (2 / len(X_train)) * np.sum(errors)

print(f"  errors (first 5): {errors[:5].round(2)}")
print(f"  All errors are negative because predictions (0) are below actual values")
print()
print(f"  Gradient for w: {grad_w:.4f}")
print(f"    Negative → increasing w will DECREASE the loss → we should go UP")
print()
print(f"  Gradient for b: {grad_b:.4f}")
print(f"    Negative → increasing b will DECREASE the loss → we should go UP")
print()
print("Makes sense! w=0 and b=0 are way too low. Both need to increase.")
print()

# ============================================================
# STEP 6: THE UPDATE — NUDGE THE WEIGHTS
# ============================================================

print("=== STEP 6: UPDATE THE WEIGHTS ===")
print()
print("The update rule:")
print("  w = w - learning_rate × gradient")
print()
print("Wait, why SUBTRACT? Because:")
print("  - Positive gradient → loss increases if w increases → go DOWN (subtract)")
print("  - Negative gradient → loss increases if w decreases → go UP (subtract negative = add)")
print()
print("It's always: walk OPPOSITE to the gradient. That's gradient DESCENT.")
print()

learning_rate = 0.01

w_new = w - learning_rate * grad_w
b_new = b - learning_rate * grad_b

print(f"  learning_rate = {learning_rate}")
print(f"  w: {w:.4f} → {w:.4f} - {learning_rate} × ({grad_w:.4f}) = {w_new:.4f}")
print(f"  b: {b:.4f} → {b:.4f} - {learning_rate} × ({grad_b:.4f}) = {b_new:.4f}")
print()

# Check: did the loss improve?
y_pred_new = w_new * X_train + b_new
loss_new = compute_mse(y_pred_new, y_train)
print(f"  Loss before update: {loss:.4f}")
print(f"  Loss after update:  {loss_new:.4f}")
print(f"  Loss went {'DOWN' if loss_new < loss else 'UP'}! One step of learning done.")
print()

# ============================================================
# STEP 7: THE TRAINING LOOP — DO IT 200 TIMES
# ============================================================

print("=== STEP 7: FULL TRAINING LOOP ===")
print()
print("Now we just repeat steps 3-6 many times (epochs).")
print()

w = 0.0
b = 0.0
learning_rate = 0.01
n_epochs = 200

train_losses = []
val_losses = []

for epoch in range(n_epochs):
    # --- PREDICT (on train set) ---
    y_pred_train = w * X_train + b

    # --- LOSS (on both sets) ---
    train_loss = compute_mse(y_pred_train, y_train)
    val_loss = compute_mse(w * X_val + b, y_val)

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    # --- GRADIENTS ---
    errors = y_pred_train - y_train
    grad_w = (2 / len(X_train)) * np.sum(errors * X_train)
    grad_b = (2 / len(X_train)) * np.sum(errors)

    # --- UPDATE ---
    w = w - learning_rate * grad_w
    b = b - learning_rate * grad_b

    if epoch % 25 == 0 or epoch == n_epochs - 1:
        print(f"  Epoch {epoch:3d}:  loss = {train_loss:.4f}  w = {w:.4f}  b = {b:.4f}")

print()
print(f"Learned: w = {w:.4f}, b = {b:.4f}")
print(f"Actual:  w = {true_w},  b = {true_b}")
print()

# ============================================================
# STEP 8: EVALUATE — HOW DID WE DO?
# ============================================================

print("=== STEP 8: EVALUATE ===")
print()

final_train_loss = train_losses[-1]
final_val_loss = val_losses[-1]

print(f"  Train loss: {final_train_loss:.4f}")
print(f"  Val loss:   {final_val_loss:.4f}")
print(f"  Baseline:   {baseline_loss:.4f}")
print()
print(f"  Beat baseline? {'YES' if final_val_loss < baseline_loss else 'NO'}")
print()

gap = abs(final_val_loss - final_train_loss)
if gap > final_train_loss * 0.5:
    print("  Diagnosis: Overfitting (val loss much higher than train)")
elif final_train_loss > baseline_loss * 0.5:
    print("  Diagnosis: Underfitting (still too high)")
else:
    print("  Diagnosis: Good fit! Train and val losses are low and close.")
print()

# Test set — final exam (only once)
test_loss = compute_mse(w * X_test + b, y_test)
print(f"  Test loss (final exam): {test_loss:.4f}")
print()

# Show some predictions
print("Sample predictions on test set:")
for i in range(5):
    pred = w * X_test[i] + b
    print(f"  x = {X_test[i]:.2f}  →  pred = {pred:.2f}  (actual = {y_test[i]:.2f})")
print()

# ============================================================
# TRAINING LOSS OVER TIME
# ============================================================

print("=== LOSS OVER TIME ===")
print()

checkpoints = [0, 10, 25, 50, 100, 150, 199]
for i in checkpoints:
    bar = "█" * int(train_losses[i] * 0.3)
    print(f"  Epoch {i:3d}: {train_losses[i]:7.2f}  {bar}")

print()
print("Loss drops fast at first, then slows down. This is normal.")
print("The model makes big corrections early, then fine-tunes.")
print()

# ============================================================
# MULTIPLE FEATURES
# ============================================================

print("=== BONUS: MULTIPLE FEATURES ===")
print()
print("Real problems have many inputs, not just one.")
print("  House price = w1·(sq ft) + w2·(bedrooms) + w3·(age) + b")
print()
print("The math is identical — just vectors instead of single numbers:")
print("  y_pred = X @ w + b         (matrix multiply)")
print("  grad_w = (2/N) × X.T @ errors")
print("  grad_b = (2/N) × sum(errors)")
print()

np.random.seed(42)

true_weights = np.array([3.0, -2.0, 0.5])
true_bias = 5.0

X_multi = np.random.randn(100, 3)
y_multi = X_multi @ true_weights + true_bias + np.random.randn(100) * 0.5

# Train with multiple features
w_multi = np.zeros(3)
b_multi = 0.0
lr = 0.01

for epoch in range(300):
    preds = X_multi @ w_multi + b_multi
    errors = preds - y_multi
    grad_w = (2 / len(X_multi)) * X_multi.T @ errors
    grad_b = (2 / len(X_multi)) * np.sum(errors)
    w_multi -= lr * grad_w
    b_multi -= lr * grad_b

print(f"Learned weights: {w_multi.round(3)}")
print(f"True weights:    {true_weights}")
print(f"Learned bias:    {b_multi:.3f}")
print(f"True bias:       {true_bias}")
print()
print("Same algorithm. Same gradient descent. Just more dimensions.")
print()

# ============================================================
# SUMMARY
# ============================================================

print("=== SUMMARY ===")
print()
print("Linear regression in 4 lines:")
print("  1. PREDICT:   y_pred = w * x + b")
print("  2. LOSS:      loss = mean((y_pred - y_actual)²)")
print("  3. GRADIENT:  grad_w = (2/N) × Σ (errors × x)")
print("                grad_b = (2/N) × Σ errors")
print("  4. UPDATE:    w = w - lr × grad_w")
print("                b = b - lr × grad_b")
print()
print("Repeat steps 1-4 for many epochs. That's training.")
print("Every neural network does the same thing — just with more layers.")
