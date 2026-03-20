"""
Topic 1.5: Overfitting & Regularization
==========================================
The bias-variance tradeoff, L1/L2 regularization, and cross-validation.
The most important skill in ML: knowing when your model is memorizing
instead of learning.

KEY TAKEAWAYS:
  - Overfitting = model memorizes training data, fails on new data.
  - Underfitting = model is too simple, fails on everything.
  - Regularization = penalty on large weights, forces the model to stay simple.
  - L2 (Ridge) adds weight² penalty — shrinks weights toward zero.
  - L1 (Lasso) adds |weight| penalty — pushes some weights to exactly zero.
  - Cross-validation = smarter way to use limited data for evaluation.
"""

import numpy as np

# ============================================================
# OVERFITTING vs UNDERFITTING — THE CORE PROBLEM
# ============================================================

print("=== OVERFITTING vs UNDERFITTING ===")
print()
print("Imagine you're studying for an exam.")
print()
print("  UNDERFITTING (too little studying):")
print("    You barely read the textbook.")
print("    You fail the practice test AND the real exam.")
print("    → Model is too simple. Fails on everything.")
print()
print("  GOOD FIT (right amount):")
print("    You understand the concepts.")
print("    You do well on practice AND the real exam.")
print("    → Model learned the real patterns.")
print()
print("  OVERFITTING (wrong kind of studying):")
print("    You memorized every practice question word-for-word.")
print("    You ace the practice test but BOMB the real exam")
print("    because the questions are slightly different.")
print("    → Model memorized the training data, can't generalize.")
print()

# ============================================================
# SEE IT HAPPEN — POLYNOMIAL FITTING
# ============================================================

print("=== WATCH OVERFITTING HAPPEN ===")
print()

np.random.seed(42)

# True relationship: y = 2x + 1 (a simple line)
n_train = 10
n_test = 50
X_train = np.sort(np.random.uniform(0, 5, n_train))
y_train = 2 * X_train + 1 + np.random.randn(n_train) * 1.5

X_test = np.sort(np.random.uniform(0, 5, n_test))
y_test = 2 * X_test + 1 + np.random.randn(n_test) * 1.5

def compute_mse(y_pred, y_actual):
    return np.mean((y_pred - y_actual) ** 2)

def fit_polynomial(X_train, y_train, X_test, y_test, degree):
    """Fit a polynomial of given degree and return train/test MSE."""
    X_poly_train = np.column_stack([X_train ** d for d in range(degree + 1)])
    X_poly_test = np.column_stack([X_test ** d for d in range(degree + 1)])

    # Solve for weights using the normal equation (don't worry about this formula)
    weights = np.linalg.lstsq(X_poly_train, y_train, rcond=None)[0]

    train_pred = X_poly_train @ weights
    test_pred = X_poly_test @ weights

    return compute_mse(train_pred, y_train), compute_mse(test_pred, y_test), weights

print(f"Data: {n_train} training points from y = 2x + 1 + noise")
print(f"Testing on {n_test} new points from the same source")
print()
print(f"We'll fit polynomials of increasing complexity:")
print(f"  degree 1: y = ax + b              (a line)")
print(f"  degree 3: y = ax³ + bx² + cx + d  (a curve)")
print(f"  degree 9: y = ax⁹ + ... + j       (a wiggly mess)")
print()

print(f"  Degree   Train MSE    Test MSE     What's happening?")
print(f"  ──────   ─────────    ────────     ─────────────────")

for degree in [1, 2, 3, 5, 7, 9]:
    train_mse, test_mse, weights = fit_polynomial(X_train, y_train, X_test, y_test, degree)

    if degree <= 2:
        status = "Good fit"
    elif degree <= 5:
        status = "Starting to overfit..."
    else:
        status = "OVERFITTING!"

    t_bar = "█" * min(int(train_mse * 2), 30)
    e_bar = "░" * min(int(test_mse * 2), 30)
    print(f"    {degree}       {train_mse:7.3f}      {test_mse:7.3f}      {status}")
    print(f"           train {t_bar}")
    print(f"           test  {e_bar}")

print()
print("PATTERN:")
print("  Train MSE keeps going DOWN (model fits training data better and better)")
print("  Test MSE goes DOWN then back UP (model starts memorizing, not learning)")
print()
print("The gap between train and test error = overfitting.")
print("This is THE signal to watch for.")
print()

# ============================================================
# WHAT OVERFITTING LOOKS LIKE IN WEIGHTS
# ============================================================

print("=== WHAT HAPPENS TO THE WEIGHTS ===")
print()

for degree in [1, 3, 9]:
    _, _, weights = fit_polynomial(X_train, y_train, X_test, y_test, degree)
    max_w = np.max(np.abs(weights))
    print(f"  Degree {degree}: largest weight = {max_w:12.2f}  weights = {weights.round(2)}")

print()
print("As the model overfits, weights become HUGE.")
print("The model uses extreme weights to twist the curve through every training point.")
print("This is the clue: big weights = overfitting.")
print()
print("Solution? PENALIZE big weights. That's regularization.")
print()

# ============================================================
# L2 REGULARIZATION (RIDGE)
# ============================================================

print("=== L2 REGULARIZATION (RIDGE) ===")
print()
print("Add a penalty to the loss:")
print()
print("  New loss = MSE + λ × Σ(w²)")
print()
print("  λ (lambda) = how much to penalize big weights")
print("  λ = 0: no regularization (original model)")
print("  λ = big: heavy penalty, weights forced toward zero")
print()
print("The gradient now includes the penalty:")
print("  grad_w = original_grad + 2λ × w")
print("  (pulls weights toward zero every step)")
print()

def fit_polynomial_ridge(X_train, y_train, X_test, y_test, degree, lam):
    """Fit with L2 (Ridge) regularization."""
    X_poly_train = np.column_stack([X_train ** d for d in range(degree + 1)])
    X_poly_test = np.column_stack([X_test ** d for d in range(degree + 1)])

    n_features = degree + 1
    I = np.eye(n_features)
    I[0, 0] = 0  # don't penalize the bias term

    # Ridge closed-form solution (don't worry about this formula)
    weights = np.linalg.solve(X_poly_train.T @ X_poly_train + lam * I,
                              X_poly_train.T @ y_train)

    train_pred = X_poly_train @ weights
    test_pred = X_poly_test @ weights

    return compute_mse(train_pred, y_train), compute_mse(test_pred, y_test), weights

print("Degree 9 polynomial WITH regularization:")
print()
print(f"  Lambda     Train MSE   Test MSE    Largest Weight")
print(f"  ──────     ─────────   ────────    ──────────────")

for lam in [0, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]:
    train_mse, test_mse, weights = fit_polynomial_ridge(
        X_train, y_train, X_test, y_test, degree=9, lam=lam)
    max_w = np.max(np.abs(weights))
    print(f"  {lam:7.3f}     {train_mse:7.3f}     {test_mse:7.3f}     {max_w:10.2f}")

print()
print("Without regularization (λ=0): weights are enormous, test MSE is bad.")
print("With some regularization: weights shrink, test MSE improves!")
print("Too much regularization: model is forced to be TOO simple (underfitting).")
print()
print("There's a sweet spot. Finding it is part of the ML workflow.")
print()

# ============================================================
# L1 REGULARIZATION (LASSO) — FEATURE SELECTION
# ============================================================

print("=== L1 REGULARIZATION (LASSO) ===")
print()
print("L1 uses absolute values instead of squares:")
print()
print("  New loss = MSE + λ × Σ|w|")
print()
print("The key difference from L2:")
print("  L2 shrinks all weights toward zero (but rarely TO zero)")
print("  L1 pushes some weights to EXACTLY zero (kills them)")
print()
print("This means L1 does FEATURE SELECTION — it decides which features matter.")
print()

# Demonstrate: data with useful and useless features
np.random.seed(42)
n = 100

# Only features 0 and 1 matter. Features 2-4 are noise.
X_demo = np.random.randn(n, 5)
y_demo = 3 * X_demo[:, 0] + (-2) * X_demo[:, 1] + np.random.randn(n) * 0.5

# Train with L1 using coordinate descent (simplified)
def train_with_l1(X, y, lam, n_epochs=1000, lr=0.01):
    w = np.zeros(X.shape[1])
    b = 0.0
    for _ in range(n_epochs):
        preds = X @ w + b
        errors = preds - y
        grad_w = (2 / len(X)) * X.T @ errors
        grad_b = (2 / len(X)) * np.sum(errors)

        # L1 gradient: sign of weights
        l1_grad = lam * np.sign(w)
        w -= lr * (grad_w + l1_grad)
        b -= lr * grad_b
    return w, b

print("5 features, but only features 0 and 1 actually matter:")
print("  True weights: [3, -2, 0, 0, 0]")
print()

for lam in [0, 0.01, 0.1, 0.5]:
    w_l1, b_l1 = train_with_l1(X_demo, y_demo, lam=lam)
    print(f"  λ = {lam:.2f}:  weights = {w_l1.round(3)}")

print()
print("With L1 regularization, the useless features (2, 3, 4) get pushed to ~0.")
print("The model automatically figured out which features matter!")
print()

# ============================================================
# L1 vs L2 SUMMARY
# ============================================================

print("=== L1 vs L2 COMPARISON ===")
print()
print("  ┌──────────────┬──────────────────────┬──────────────────────┐")
print("  │              │ L2 (Ridge)           │ L1 (Lasso)           │")
print("  ├──────────────┼──────────────────────┼──────────────────────┤")
print("  │ Penalty      │ λ × Σ(w²)            │ λ × Σ|w|             │")
print("  │ Effect       │ Shrinks all weights   │ Some weights → zero  │")
print("  │ Use when     │ All features matter   │ Many useless features│")
print("  │ Also called  │ Ridge Regression      │ Lasso Regression     │")
print("  └──────────────┴──────────────────────┴──────────────────────┘")
print()

# ============================================================
# CROSS-VALIDATION — SMARTER DATA SPLITTING
# ============================================================

print("=== CROSS-VALIDATION ===")
print()
print("Problem: with small data, your val set might be too small")
print("to give a reliable score. You got lucky or unlucky with the split.")
print()
print("Solution: K-Fold Cross-Validation")
print("  Split data into K equal parts (folds)")
print("  Train K times, each time using a different fold as validation")
print("  Average the scores")
print()
print("Example with 5-fold CV on 100 samples:")
print()

np.random.seed(42)
n_cv = 100
X_cv = np.random.randn(n_cv, 1) * 3
y_cv = 2 * X_cv.squeeze() + 5 + np.random.randn(n_cv) * 2

k_folds = 5
fold_size = n_cv // k_folds
indices = np.random.permutation(n_cv)

scores = []
for fold in range(k_folds):
    val_start = fold * fold_size
    val_end = val_start + fold_size
    val_idx = indices[val_start:val_end]
    train_idx = np.concatenate([indices[:val_start], indices[val_end:]])

    X_tr, y_tr = X_cv[train_idx], y_cv[train_idx]
    X_vl, y_vl = X_cv[val_idx], y_cv[val_idx]

    # Simple linear regression
    w = 0.0
    b = 0.0
    for _ in range(200):
        preds = (X_tr.squeeze() * w + b)
        errors = preds - y_tr
        grad_w = (2 / len(X_tr)) * np.sum(errors * X_tr.squeeze())
        grad_b = (2 / len(X_tr)) * np.sum(errors)
        w -= 0.01 * grad_w
        b -= 0.01 * grad_b

    val_preds = X_vl.squeeze() * w + b
    val_mse = compute_mse(val_preds, y_vl)
    scores.append(val_mse)

    print(f"  Fold {fold + 1}: val MSE = {val_mse:.4f}  "
          f"(train on {len(train_idx)}, validate on {len(val_idx)})")

print()
print(f"  Average MSE: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
print()
print("Each fold gives a slightly different score.")
print("The average is MORE reliable than any single split.")
print("The ± tells you how much the score varies — lower is better.")
print()

# ============================================================
# HOW TO FIX OVERFITTING / UNDERFITTING
# ============================================================

print("=== FIXING OVERFITTING vs UNDERFITTING ===")
print()
print("  UNDERFITTING (train loss is high):")
print("    → Use a more complex model")
print("    → Add more features")
print("    → Train longer")
print("    → Reduce regularization")
print()
print("  OVERFITTING (train loss low, val loss high):")
print("    → Get more training data")
print("    → Add regularization (L1/L2)")
print("    → Use a simpler model")
print("    → Remove noisy features")
print("    → Early stopping (stop training before it overfits)")
print()
print("This is the iteration loop from Topic 1.1: diagnose → fix → repeat.")
