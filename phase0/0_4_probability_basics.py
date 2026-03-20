"""
Topic 0.4: Probability Basics
===============================
Distributions, Bayes' theorem, and how probability connects to ML.

KEY TAKEAWAYS:
  - Model outputs are probability distributions (must sum to 1).
  - Softmax converts raw scores into probabilities.
  - Normal distribution is everywhere — weight init, data, noise.
  - Bayes' theorem: accuracy is misleading on imbalanced data.
  - Expected value = average outcome = what "loss" really means.
"""

import numpy as np

# ============================================================
# PROBABILITY DISTRIBUTION — WHAT A CLASSIFIER OUTPUTS
# ============================================================

print("=== CLASSIFIER OUTPUT IS A PROBABILITY DISTRIBUTION ===")

raw_scores = np.array([2.0, 1.0, 0.1])  # raw output from last layer (called "logits")

# These are NOT probabilities — they don't sum to 1, and can be negative.
print(f"Raw scores (logits): {raw_scores}")
print(f"Sum: {raw_scores.sum():.1f}  (not 1 — these aren't probabilities yet)")
print()

# Softmax converts logits → probabilities
def softmax(x):
    e_x = np.exp(x - np.max(x))  # subtract max for numerical stability
    return e_x / e_x.sum()

probs = softmax(raw_scores)
labels = ["cat", "dog", "bird"]

print("After softmax:")
for label, prob in zip(labels, probs):
    bar = "█" * int(prob * 40)
    print(f"  P({label:4s}) = {prob:.4f}  {bar}")
print(f"  Sum = {probs.sum():.4f}  (always 1.0)")
print()
print("Softmax does two things:")
print("  1. Makes all values positive (via exp)")
print("  2. Makes them sum to 1 (via dividing by total)")
print()

# ============================================================
# ARGMAX vs PROBABILITIES
# ============================================================

print("=== PREDICTION = ARGMAX OF PROBABILITIES ===")
predicted_class = np.argmax(probs)
print(f"Probabilities: {probs.round(4)}")
print(f"Predicted class: {predicted_class} ({labels[predicted_class]})")
print(f"Confidence: {probs[predicted_class]:.1%}")
print()

# What if the model is uncertain?
uncertain_logits = np.array([1.0, 0.9, 0.8])
uncertain_probs = softmax(uncertain_logits)
print(f"Uncertain model logits: {uncertain_logits}")
print(f"Uncertain model probs:  {uncertain_probs.round(4)}")
print(f"Almost uniform — the model has no idea!")
print()

# ============================================================
# NORMAL (GAUSSIAN) DISTRIBUTION
# ============================================================

print("=== NORMAL DISTRIBUTION ===")

# Generate samples from a normal distribution
np.random.seed(42)
samples = np.random.normal(loc=0.0, scale=1.0, size=10000)  # mean=0, std=1

print(f"10,000 samples from N(0, 1):")
print(f"  Mean:   {samples.mean():.4f}   (should be close to 0)")
print(f"  Std:    {samples.std():.4f}    (should be close to 1)")
print(f"  Min:    {samples.min():.4f}")
print(f"  Max:    {samples.max():.4f}")
print()

# The 68-95-99.7 rule
within_1std = np.sum(np.abs(samples) < 1) / len(samples) * 100
within_2std = np.sum(np.abs(samples) < 2) / len(samples) * 100
within_3std = np.sum(np.abs(samples) < 3) / len(samples) * 100

print("The 68-95-99.7 rule:")
print(f"  Within 1 std: {within_1std:.1f}%  (expected ~68%)")
print(f"  Within 2 std: {within_2std:.1f}%  (expected ~95%)")
print(f"  Within 3 std: {within_3std:.1f}%  (expected ~99.7%)")
print()

# How neural network weights are initialized
print("Neural network weights are typically initialized from N(0, small_number):")
weight_init = np.random.normal(0, 0.01, size=(3, 3))
print(f"  Example weight matrix (initialized from N(0, 0.01)):\n  {weight_init.round(4)}")
print(f"  All values are tiny and centered around 0 — this is how training starts.")
print()

# ============================================================
# BAYES' THEOREM — WHY ACCURACY IS MISLEADING
# ============================================================

print("=== BAYES' THEOREM — THE MEDICAL TEST EXAMPLE ===")

p_disease = 0.01        # 1% of people have it
p_pos_given_disease = 0.90   # test catches 90% of cases
p_pos_given_healthy = 0.05   # 5% false positive rate

p_healthy = 1 - p_disease
p_positive = (p_pos_given_disease * p_disease) + (p_pos_given_healthy * p_healthy)
p_disease_given_pos = (p_pos_given_disease * p_disease) / p_positive

print(f"P(disease)              = {p_disease:.2%}")
print(f"P(positive | disease)   = {p_pos_given_disease:.2%}")
print(f"P(positive | healthy)   = {p_pos_given_healthy:.2%}")
print(f"P(positive)             = {p_positive:.4f}")
print()
print(f"P(disease | positive)   = {p_disease_given_pos:.2%}")
print(f"Even with a positive test, only {p_disease_given_pos:.1%} chance of disease!")
print()

# ============================================================
# ML CONNECTION: IMBALANCED DATASETS
# ============================================================

print("=== WHY THIS MATTERS IN ML — IMBALANCED DATA ===")

np.random.seed(42)
n_samples = 1000
n_spam = 10  # only 1% spam

# A "dumb" model that always predicts "not spam"
always_not_spam_accuracy = (n_samples - n_spam) / n_samples

print(f"Email dataset: {n_samples} emails, {n_spam} spam ({n_spam/n_samples:.1%})")
print(f"Model that always says 'not spam': accuracy = {always_not_spam_accuracy:.1%}")
print(f"Sounds great! But it catches 0 spam emails — completely useless.")
print()
print("This is why we need precision, recall, F1 — not just accuracy.")
print("We'll cover these properly in Topic 1.4.")
print()

# ============================================================
# EXPECTED VALUE — WHAT "AVERAGE LOSS" MEANS
# ============================================================

print("=== EXPECTED VALUE ===")

# Fair die
die_values = np.array([1, 2, 3, 4, 5, 6])
die_probs = np.array([1/6] * 6)
expected_value = np.sum(die_values * die_probs)

print(f"Fair die:")
print(f"  E[X] = sum(value × probability)")
print(f"       = {' + '.join(f'{v}×{p:.3f}' for v, p in zip(die_values, die_probs))}")
print(f"       = {expected_value:.4f}")
print()

# Simulate to verify
rolls = np.random.choice(die_values, size=100000)
print(f"  Simulated average of 100,000 rolls: {rolls.mean():.4f}")
print(f"  (Close to theoretical {expected_value:.4f})")
print()

# Connection to ML: loss is an expected value
print("In ML, training loss = average loss over all samples in a batch:")
print("  L = (1/N) × Σ loss(prediction_i, target_i)")
print("  This IS an expected value — the average 'wrongness' of your model.")
print("  Gradient descent minimizes this expected value.")
