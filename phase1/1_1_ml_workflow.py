"""
Topic 1.1: The ML Workflow
============================
Problem framing, data splits, evaluation, and the iteration loop.
Before you build any model, you need to think like an ML engineer.

KEY TAKEAWAYS:
  - Frame the problem first: what's the input, what's the output, what type?
  - ALWAYS split data into train / validation / test. No exceptions.
  - Train set = learn. Validation set = tune. Test set = final grade.
  - The MOST important diagnostic in ML:
      Train HIGH, Val HIGH  → Underfitting (model too simple / not trained enough)
      Train LOW,  Val HIGH  → Overfitting  (memorized training data, doesn't generalize)
      Train LOW,  Val LOW   → Good fit     (this is what you want)
  - The ML workflow is a LOOP: try → measure → diagnose → fix → repeat.
  - Data leakage = test data influences training. Split FIRST, compute stats from train only.
"""

import numpy as np

# ============================================================
# STEP 1: FRAME THE PROBLEM
# ============================================================

print("=== STEP 1: FRAME THE PROBLEM ===")
print()
print("Before writing ANY code, answer these questions:")
print()
print("  1. What is the INPUT?")
print("     → An image? A row of numbers? A sentence? Audio?")
print()
print("  2. What is the OUTPUT?")
print("     → A number (regression)? A class (classification)?")
print()
print("  3. What TYPE of problem is it?")
print("     ┌──────────────────┬────────────────────────────────────┐")
print("     │ Type             │ Examples                           │")
print("     ├──────────────────┼────────────────────────────────────┤")
print("     │ Regression       │ Predict house price, temperature   │")
print("     │ Binary classif.  │ Spam or not, fraud or not          │")
print("     │ Multi-class      │ Cat/dog/bird, digit 0-9            │")
print("     │ Unsupervised     │ Group customers, find anomalies    │")
print("     └──────────────────┴────────────────────────────────────┘")
print()
print("  4. What METRIC will you use to judge success?")
print("     → Accuracy? Precision? RMSE? Depends on the problem.")
print()
print("  5. What's the BASELINE?")
print("     → Always have a dumb baseline to beat.")
print("     → 'Always predict the average' for regression.")
print("     → 'Always predict the most common class' for classification.")
print()

# ============================================================
# STEP 2: UNDERSTAND YOUR DATA
# ============================================================

print("=== STEP 2: LOOK AT YOUR DATA ===")

np.random.seed(42)

n_samples = 200
n_features = 3

X = np.random.randn(n_samples, n_features)
noise = np.random.randn(n_samples) * 0.5
y = 3 * X[:, 0] + (-2) * X[:, 1] + 0.5 * X[:, 2] + noise

print(f"Dataset: {n_samples} samples, {n_features} features")
print(f"X shape: {X.shape}  (rows=samples, cols=features)")
print(f"y shape: {y.shape}  (one target per sample)")
print()
print(f"First 3 samples:")
for i in range(3):
    print(f"  X[{i}] = {X[i].round(3)}  →  y[{i}] = {y[i]:.3f}")
print()

print("Quick stats:")
print(f"  y mean:  {y.mean():.3f}")
print(f"  y std:   {y.std():.3f}")
print(f"  y range: [{y.min():.2f}, {y.max():.2f}]")
print()
print("Always look at your data first. Shapes, ranges, missing values, outliers.")
print("Garbage in = garbage out. No model can fix bad data.")
print()

# ============================================================
# STEP 3: SPLIT YOUR DATA — TRAIN / VALIDATION / TEST
# ============================================================

print("=== STEP 3: SPLIT YOUR DATA ===")
print()
print("You MUST split into (at least) 3 sets:")
print()
print("  ┌─────────────────────────────────────────────────────────┐")
print("  │  TRAIN SET (60-80%)                                     │")
print("  │  The model LEARNS from this data.                       │")
print("  │  It sees these examples during training.                │")
print("  ├─────────────────────────────────────────────────────────┤")
print("  │  VALIDATION SET (10-20%)                                │")
print("  │  Used to TUNE the model (pick hyperparameters).         │")
print("  │  The model never trains on this, but you peek at it     │")
print("  │  repeatedly to decide what to change.                   │")
print("  ├─────────────────────────────────────────────────────────┤")
print("  │  TEST SET (10-20%)                                      │")
print("  │  The FINAL exam. Used ONCE at the very end.             │")
print("  │  If you tune based on test set, you're cheating.        │")
print("  └─────────────────────────────────────────────────────────┘")
print()

def train_val_test_split(X, y, train_ratio=0.7, val_ratio=0.15, seed=42):
    np.random.seed(seed)
    n = len(X)
    indices = np.random.permutation(n)

    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    return (X[train_idx], y[train_idx],
            X[val_idx], y[val_idx],
            X[test_idx], y[test_idx])

X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split(X, y)

print(f"Split results:")
print(f"  Train:      {X_train.shape[0]} samples ({X_train.shape[0]/n_samples:.0%})")
print(f"  Validation: {X_val.shape[0]} samples ({X_val.shape[0]/n_samples:.0%})")
print(f"  Test:       {X_test.shape[0]} samples ({X_test.shape[0]/n_samples:.0%})")
print()

# ============================================================
# WHY SHUFFLE? — ORDER MATTERS
# ============================================================

print("=== WHY SHUFFLE BEFORE SPLITTING? ===")
print()
print("Imagine your data is sorted by class:")
print("  [cat, cat, cat, ..., dog, dog, dog, ..., bird, bird, bird]")
print()
print("If you take the first 70% as train:")
print("  Train = all cats + some dogs")
print("  Test  = some dogs + all birds")
print("  The model has NEVER seen a bird during training!")
print()
print("Shuffling ensures each split has a representative mix.")
print()

# ============================================================
# STEP 4: ESTABLISH A BASELINE
# ============================================================

print("=== STEP 4: ESTABLISH A BASELINE ===")

baseline_pred = np.mean(y_train)

train_mse_baseline = np.mean((baseline_pred - y_train) ** 2)
val_mse_baseline = np.mean((baseline_pred - y_val) ** 2)

print(f"Dumbest possible model: always predict the training set mean = {baseline_pred:.3f}")
print()
print(f"  Baseline train MSE: {train_mse_baseline:.4f}")
print(f"  Baseline val MSE:   {val_mse_baseline:.4f}")
print()
print("Any real model MUST beat this. If it doesn't, something is broken.")
print()

# ============================================================
# STEP 5: TRAIN → EVALUATE
# ============================================================

print("=== STEP 5: TRAIN AND EVALUATE ===")
print()
print("(We'll learn HOW training works in Topic 1.2 — Linear Regression.)")
print("For now, just understand WHAT happens at a high level:")
print()
print("  1. The model makes predictions on the TRAIN set")
print("  2. We compute the loss (how wrong it is)")
print("  3. The model adjusts its weights to reduce the loss")
print("  4. Repeat for many rounds (called 'epochs')")
print()
print("After each epoch, we ALSO compute loss on the VALIDATION set")
print("(without training on it!) to see if the model generalizes.")
print()

# Simulate what training loss looks like over time
# (Don't worry about HOW — that's Topic 1.2)
train_losses = [11.29, 9.56, 8.02, 6.82, 5.74, 4.88, 4.15, 3.56, 3.04, 2.61, 2.31]
val_losses   = [15.07, 12.85, 10.95, 9.34, 8.02, 6.90, 5.94, 5.13, 4.43, 3.87, 3.44]

print("Here's what it looks like — loss decreasing over time:")
print()
for i, (tl, vl) in enumerate(zip(train_losses, val_losses)):
    epoch = i * 5
    t_bar = "█" * int(tl * 2)
    v_bar = "░" * int(vl * 2)
    print(f"  Epoch {epoch:2d}:  train={tl:5.2f} {t_bar}")
    print(f"           val  ={vl:5.2f} {v_bar}")
print()
print("Both losses going DOWN = model is learning and generalizing. Good!")
print("(We'll build this ourselves from scratch in 1.2.)")
print()

final_train = train_losses[-1]
final_val = val_losses[-1]

# ============================================================
# STEP 6: DIAGNOSE — OVERFITTING vs UNDERFITTING
# ============================================================

print("=== STEP 6: DIAGNOSE — OVERFITTING vs UNDERFITTING ===")
print()
print("Compare train loss vs validation loss:")
print()

final_train = train_losses[-1]
final_val = val_losses[-1]

print(f"  Our model:  train={final_train:.4f}  val={final_val:.4f}")

gap = abs(final_val - final_train)
print()

if final_train > 5:
    print("  DIAGNOSIS: Underfitting")
    print("  Both losses are high. Model is too simple or needs more training.")
elif gap > final_train * 0.5:
    print("  DIAGNOSIS: Overfitting")
    print("  Train loss is low but val loss is much higher.")
    print("  Model memorized training data instead of learning patterns.")
else:
    print("  DIAGNOSIS: Good fit")
    print("  Train and val losses are both low and close together.")

print()
print("  ┌────────────────────────────────────────────────────────────┐")
print("  │ Scenario        │ Train Loss │ Val Loss   │ Diagnosis     │")
print("  ├────────────────────────────────────────────────────────────┤")
print("  │ Both high        │ HIGH       │ HIGH       │ Underfitting  │")
print("  │ Train low,       │ LOW        │ HIGH       │ Overfitting   │")
print("  │  val high        │            │            │               │")
print("  │ Both low &       │ LOW        │ LOW        │ Good fit!     │")
print("  │  close           │            │            │               │")
print("  └────────────────────────────────────────────────────────────┘")
print()
print("This is the MOST important diagnostic in all of ML.")
print("Always compare train vs validation loss. Always.")
print()

# ============================================================
# STEP 7: THE ITERATION LOOP
# ============================================================

print("=== STEP 7: THE ML ITERATION LOOP ===")
print()
print("ML is NOT: build model → ship it.")
print("ML IS:     build model → measure → diagnose → fix → repeat.")
print()
print("  ┌──────────────────────────────────────────────┐")
print("  │                                              │")
print("  │   Frame problem                              │")
print("  │       ↓                                      │")
print("  │   Get & split data                           │")
print("  │       ↓                                      │")
print("  │   Establish baseline                         │")
print("  │       ↓                                      │")
print("  │   ┌─→ Train model                            │")
print("  │   │       ↓                                  │")
print("  │   │   Evaluate on validation set             │")
print("  │   │       ↓                                  │")
print("  │   │   Diagnose (overfit? underfit?)          │")
print("  │   │       ↓                                  │")
print("  │   │   Fix (more data? simpler model?         │")
print("  │   │        regularization? features?)        │")
print("  │   │       ↓                                  │")
print("  │   └── Repeat until good                      │")
print("  │       ↓                                      │")
print("  │   Final evaluation on TEST set (once!)       │")
print("  │       ↓                                      │")
print("  │   Ship it                                    │")
print("  │                                              │")
print("  └──────────────────────────────────────────────┘")
print()

# ============================================================
# DATA LEAKAGE — THE SILENT KILLER
# ============================================================

print("=== COMMON MISTAKE: DATA LEAKAGE ===")
print()

y_all = y.copy()
all_mean = np.mean(y_all)
all_std = np.std(y_all)
y_normalized_wrong = (y_all - all_mean) / all_std

print("DATA LEAKAGE = test data influences training.")
print()
print("Example — wrong way to normalize:")
print(f"  1. Compute mean/std on ALL data (including test): mean={all_mean:.3f}, std={all_std:.3f}")
print(f"  2. Normalize everything using those stats")
print(f"  3. THEN split into train/test")
print(f"  PROBLEM: The test set's statistics leaked into the training normalization!")
print()

train_mean = np.mean(y_train)
train_std = np.std(y_train)

print("Correct way:")
print(f"  1. Split FIRST")
print(f"  2. Compute mean/std on TRAIN set only: mean={train_mean:.3f}, std={train_std:.3f}")
print(f"  3. Use train stats to normalize ALL sets (train, val, test)")
print(f"  The test set stays truly unseen.")
print()
print("Rule: Anything computed from data (mean, std, vocabulary, etc.)")
print("      must come from the TRAINING set only.")
