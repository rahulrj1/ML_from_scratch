"""
Topic 0.5: Loss Functions (The Math)
======================================
MSE, cross-entropy, and why they exist. The loss function is the
single number that tells the model "how wrong you are." Training
is just minimizing this number.

KEY TAKEAWAYS:
  - Loss function = a score of how bad the model's prediction is.
  - MSE (Mean Squared Error) is for regression — penalizes big errors heavily.
  - Cross-entropy is for classification — penalizes confident wrong answers.
  - Why not MSE for classification? Gradients get tiny near 0 and 1 (sigmoid).
  - Cross-entropy + softmax has a beautifully simple gradient: (predicted - actual).
"""

import numpy as np

# ============================================================
# WHY LOSS FUNCTIONS EXIST
# ============================================================

print("=== WHY LOSS FUNCTIONS EXIST ===")
print()
print("A model makes a prediction. The loss function answers ONE question:")
print('  "How far off is this prediction from the truth?"')
print()
print("Training = find weights that MINIMIZE the loss.")
print("The derivative of the loss w.r.t. each weight tells us which way to nudge it.")
print()

# ============================================================
# MEAN SQUARED ERROR (MSE) — REGRESSION LOSS
# ============================================================

print("=== MEAN SQUARED ERROR (MSE) ===")

def mse_loss(predictions, targets):
    return np.mean((predictions - targets) ** 2)

targets = np.array([3.0, -0.5, 2.0, 7.0])

good_preds = np.array([2.8, -0.3, 2.1, 6.8])
bad_preds  = np.array([1.0,  2.0, 5.0, 3.0])

print(f"Targets:          {targets}")
print(f"Good predictions: {good_preds}")
print(f"Bad predictions:  {bad_preds}")
print()
print(f"MSE (good model): {mse_loss(good_preds, targets):.4f}")
print(f"MSE (bad model):  {mse_loss(bad_preds, targets):.4f}")
print()

# Step-by-step breakdown
errors = good_preds - targets
squared_errors = errors ** 2
print("Step-by-step for the good model:")
print(f"  Errors (pred - target):  {errors}")
print(f"  Squared errors:          {squared_errors}")
print(f"  Mean of squared errors:  {squared_errors.mean():.4f}")
print()
print("Why SQUARED?")
print("  1. Makes all errors positive (no cancellation of +/- errors)")
print("  2. Penalizes big errors MORE than small ones (2² = 4, but 4² = 16)")
print("  3. The derivative is smooth — great for gradient descent")
print()

# ============================================================
# MSE GRADIENT — HOW IT DRIVES LEARNING
# ============================================================

print("=== MSE GRADIENT ===")
print()
print("MSE = (1/N) × Σ (pred_i - target_i)²")
print("d(MSE)/d(pred_i) = (2/N) × (pred_i - target_i)")
print()

pred = 5.0
target = 3.0
grad = 2 * (pred - target)

print(f"Example: prediction = {pred}, target = {target}")
print(f"  Gradient = 2 × ({pred} - {target}) = {grad}")
print(f"  Gradient is POSITIVE → prediction is too HIGH → nudge it DOWN")
print()

pred = 1.0
target = 3.0
grad = 2 * (pred - target)

print(f"Example: prediction = {pred}, target = {target}")
print(f"  Gradient = 2 × ({pred} - {target}) = {grad}")
print(f"  Gradient is NEGATIVE → prediction is too LOW → nudge it UP")
print()
print("The gradient is proportional to the error — bigger mistake = stronger correction.")
print()

# ============================================================
# BINARY CROSS-ENTROPY — CLASSIFICATION LOSS (2 CLASSES)
# ============================================================

print("=== BINARY CROSS-ENTROPY (LOG LOSS) ===")

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def binary_cross_entropy(pred_prob, target):
    eps = 1e-15  # avoid log(0)
    pred_prob = np.clip(pred_prob, eps, 1 - eps)
    return -( target * np.log(pred_prob) + (1 - target) * np.log(1 - pred_prob) )

print("Binary classification: is this email spam (1) or not (0)?")
print()

target = 1.0  # it IS spam
for prob in [0.99, 0.8, 0.5, 0.2, 0.01]:
    loss = binary_cross_entropy(prob, target)
    bar = "█" * int(loss * 8)
    print(f"  P(spam) = {prob:.2f}  →  loss = {loss:.4f}  {bar}")

print()
print("When target = 1:")
print("  - Predicting 0.99 → tiny loss   (correct and confident)")
print("  - Predicting 0.01 → HUGE loss   (wrong and confident)")
print()

target = 0.0  # it is NOT spam
print("When target = 0:")
for prob in [0.01, 0.2, 0.5, 0.8, 0.99]:
    loss = binary_cross_entropy(prob, target)
    bar = "█" * int(loss * 8)
    print(f"  P(spam) = {prob:.2f}  →  loss = {loss:.4f}  {bar}")
print()
print("Cross-entropy PUNISHES confident wrong answers exponentially.")
print("Being 90% sure and wrong costs WAY more than being 60% sure and wrong.")
print()

# ============================================================
# WHY -log(p)? — THE INTUITION
# ============================================================

print("=== WHY -log(p)? ===")
print()

probs = np.array([0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99])
neg_logs = -np.log(probs)

print("If the correct class has probability p, cross-entropy = -log(p)")
print()
for p, nl in zip(probs, neg_logs):
    bar = "█" * int(nl * 5)
    print(f"  p = {p:.2f}  →  -log(p) = {nl:.4f}  {bar}")

print()
print("Properties of -log(p):")
print("  - p = 1.0 → loss = 0    (perfect prediction, no penalty)")
print("  - p = 0.5 → loss = 0.69 (uncertain, moderate penalty)")
print("  - p → 0   → loss → ∞    (confident AND wrong, massive penalty)")
print("  - Always non-negative")
print("  - Smooth and differentiable — perfect for gradient descent")
print()

# ============================================================
# CATEGORICAL CROSS-ENTROPY — MULTI-CLASS
# ============================================================

print("=== CATEGORICAL CROSS-ENTROPY (MULTI-CLASS) ===")

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def categorical_cross_entropy(probs, target_index):
    eps = 1e-15
    return -np.log(np.clip(probs[target_index], eps, 1.0))

labels = ["cat", "dog", "bird"]
true_class = 0  # it's a cat

print(f"True class: {labels[true_class]}")
print()

# Scenario 1: confident and correct
logits1 = np.array([4.0, 1.0, 0.5])
probs1 = softmax(logits1)
loss1 = categorical_cross_entropy(probs1, true_class)

print("Scenario 1 — Confident and CORRECT:")
for l, p in zip(labels, probs1):
    print(f"  P({l:4s}) = {p:.4f}")
print(f"  Loss = -log(P(cat)) = -log({probs1[true_class]:.4f}) = {loss1:.4f}")
print()

# Scenario 2: uncertain
logits2 = np.array([1.0, 0.9, 0.8])
probs2 = softmax(logits2)
loss2 = categorical_cross_entropy(probs2, true_class)

print("Scenario 2 — Uncertain:")
for l, p in zip(labels, probs2):
    print(f"  P({l:4s}) = {p:.4f}")
print(f"  Loss = -log(P(cat)) = -log({probs2[true_class]:.4f}) = {loss2:.4f}")
print()

# Scenario 3: confident and WRONG
logits3 = np.array([0.1, 5.0, 0.5])
probs3 = softmax(logits3)
loss3 = categorical_cross_entropy(probs3, true_class)

print("Scenario 3 — Confident and WRONG:")
for l, p in zip(labels, probs3):
    print(f"  P({l:4s}) = {p:.4f}")
print(f"  Loss = -log(P(cat)) = -log({probs3[true_class]:.4f}) = {loss3:.4f}")
print()

print(f"Loss comparison:  correct={loss1:.4f}  uncertain={loss2:.4f}  wrong={loss3:.4f}")
print("The worse the prediction, the higher the loss. That's the whole point.")
print()

# ============================================================
# WHY CROSS-ENTROPY, NOT MSE, FOR CLASSIFICATION?
# ============================================================

print("=== WHY NOT MSE FOR CLASSIFICATION? ===")

print()
print("Let's compare MSE vs cross-entropy when the model is confidently WRONG.")
print()

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

target = 1.0
logit = -5.0  # model is very confident it's class 0 (wrong!)
pred = sigmoid(logit)

print(f"Target: {target},  Logit: {logit},  Sigmoid output: {pred:.6f}")
print()

# MSE gradient through sigmoid
mse = (pred - target) ** 2
mse_grad_pred = 2 * (pred - target)
sigmoid_grad = pred * (1 - pred)  # sigmoid derivative
mse_grad_logit = mse_grad_pred * sigmoid_grad

print("MSE path:")
print(f"  MSE loss          = {mse:.6f}")
print(f"  d(MSE)/d(pred)    = {mse_grad_pred:.6f}")
print(f"  d(sigmoid)/d(logit) = sigmoid × (1 - sigmoid) = {sigmoid_grad:.8f}")
print(f"  d(MSE)/d(logit)   = {mse_grad_logit:.8f}  ← TINY! Almost no learning!")
print()

# Cross-entropy gradient through sigmoid
bce = binary_cross_entropy(pred, target)
ce_grad_logit = pred - target  # the beautiful simplification

print("Cross-entropy path:")
print(f"  CE loss           = {bce:.6f}")
print(f"  d(CE)/d(logit)    = pred - target = {pred:.6f} - {target} = {ce_grad_logit:.6f}")
print()

print(f"MSE gradient magnitude:           {abs(mse_grad_logit):.8f}")
print(f"Cross-entropy gradient magnitude: {abs(ce_grad_logit):.6f}")
print(f"Cross-entropy gradient is {abs(ce_grad_logit)/abs(mse_grad_logit):.0f}× stronger!")
print()
print("This is THE reason we use cross-entropy for classification:")
print("  MSE + sigmoid → gradient vanishes when the model is very wrong")
print("  CE + sigmoid  → gradient = (pred - target), strong and simple")
print("  The model recovers from bad predictions MUCH faster with cross-entropy.")
print()

# ============================================================
# THE BEAUTIFUL GRADIENT: SOFTMAX + CROSS-ENTROPY
# ============================================================

print("=== THE BEAUTIFUL GRADIENT: SOFTMAX + CROSS-ENTROPY ===")
print()

logits = np.array([2.0, 1.0, 0.1])
probs = softmax(logits)
true_class = 0

one_hot = np.zeros(3)
one_hot[true_class] = 1.0

gradient = probs - one_hot

print(f"Logits:       {logits}")
print(f"Softmax:      {probs.round(4)}")
print(f"True (1-hot): {one_hot}")
print(f"Gradient:     {gradient.round(4)}")
print()
print("The gradient of (softmax + cross-entropy) w.r.t. logits is simply:")
print("  gradient = predicted_probs - one_hot_target")
print()
print("For the correct class:  grad = P(class) - 1  (negative → push logit UP)")
print("For wrong classes:      grad = P(class) - 0  (positive → push logit DOWN)")
print()
print("This is arguably the most elegant result in all of deep learning.")
print("No matter how many classes, the gradient is always just: (predicted - actual).")
print()

# Verify numerically
print("Numerical verification:")
h = 1e-5
numerical_grads = np.zeros(3)
for i in range(3):
    logits_plus = logits.copy()
    logits_plus[i] += h
    logits_minus = logits.copy()
    logits_minus[i] -= h
    loss_plus = categorical_cross_entropy(softmax(logits_plus), true_class)
    loss_minus = categorical_cross_entropy(softmax(logits_minus), true_class)
    numerical_grads[i] = (loss_plus - loss_minus) / (2 * h)

print(f"  Analytical:  {gradient.round(6)}")
print(f"  Numerical:   {numerical_grads.round(6)}")
print(f"  Match: {np.allclose(gradient, numerical_grads, atol=1e-4)}")
print()

# ============================================================
# LOSS OVER A BATCH — WHAT TRAINING ACTUALLY COMPUTES
# ============================================================

print("=== LOSS OVER A BATCH ===")

np.random.seed(42)

n_samples = 5
n_classes = 3

logits_batch = np.random.randn(n_samples, n_classes)
true_classes = np.array([0, 2, 1, 0, 2])

print(f"Batch of {n_samples} samples, {n_classes} classes")
print(f"True classes: {true_classes}")
print()

total_loss = 0
for i in range(n_samples):
    probs = softmax(logits_batch[i])
    loss = categorical_cross_entropy(probs, true_classes[i])
    predicted = np.argmax(probs)
    correct = "✓" if predicted == true_classes[i] else "✗"
    print(f"  Sample {i}: probs={probs.round(3)}  true={true_classes[i]}  "
          f"pred={predicted} {correct}  loss={loss:.4f}")
    total_loss += loss

avg_loss = total_loss / n_samples
print()
print(f"  Average loss over batch: {avg_loss:.4f}")
print()
print("Training = run batches → compute avg loss → backprop → update weights → repeat.")
print("The loss should decrease over time. If it doesn't, something is wrong.")
print()

# ============================================================
# SUMMARY: WHICH LOSS WHEN?
# ============================================================

print("=== SUMMARY: WHICH LOSS FUNCTION WHEN? ===")
print()
print("  Problem Type              Loss Function          Output Layer")
print("  ─────────────────────     ────────────────────   ────────────")
print("  Regression                MSE (or MAE)           Linear (no activation)")
print("  Binary classification     Binary cross-entropy   Sigmoid")
print("  Multi-class (1 label)     Categorical CE         Softmax")
print()
print("The loss function and output activation are a PAIR — they go together.")
print("MSE + linear for regression.  Cross-entropy + softmax for classification.")
print("Use the wrong pair and training will be painfully slow or won't converge.")
