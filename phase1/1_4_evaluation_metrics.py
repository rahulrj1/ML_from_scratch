"""
Topic 1.4: Evaluation Metrics
================================
Accuracy is not enough. Precision, recall, F1, AUC-ROC, and
confusion matrices — the tools to REALLY understand your model.

KEY TAKEAWAYS:
  - Accuracy can be misleading (99% accuracy on imbalanced data = useless).
  - Confusion matrix: TP, FP, TN, FN — the foundation of all metrics.
  - Precision: "of all things I predicted positive, how many were right?"
  - Recall: "of all actual positives, how many did I catch?"
  - F1 score: the balance between precision and recall.
  - Choose your metric based on what MISTAKES cost more.
"""

import numpy as np

# ============================================================
# WHY ACCURACY IS NOT ENOUGH
# ============================================================

print("=== WHY ACCURACY IS NOT ENOUGH ===")
print()

n_total = 10000
n_fraud = 50  # 0.5% fraud rate

print(f"Credit card fraud detection:")
print(f"  {n_total} transactions, only {n_fraud} are fraud ({n_fraud/n_total:.1%})")
print()

# "Model" that always predicts "not fraud"
accuracy = (n_total - n_fraud) / n_total
print(f"  Model that ALWAYS says 'not fraud':")
print(f"  Accuracy = {accuracy:.2%}")
print(f"  Sounds amazing! But it catches ZERO fraud.")
print(f"  This model is completely useless.")
print()
print("Accuracy tells you 'what % did I get right overall'")
print("but ignores WHERE the mistakes are. We need better tools.")
print()

# ============================================================
# THE CONFUSION MATRIX — WHERE MISTAKES HAPPEN
# ============================================================

print("=== THE CONFUSION MATRIX ===")
print()
print("Every prediction falls into one of 4 boxes:")
print()
print("                          PREDICTED")
print("                     Positive    Negative")
print("               ┌────────────┬────────────┐")
print("  ACTUAL  Pos  │     TP     │     FN     │")
print("               │ (correct!) │  (missed!) │")
print("               ├────────────┼────────────┤")
print("          Neg  │     FP     │     TN     │")
print("               │  (false    │ (correct!) │")
print("               │   alarm!)  │            │")
print("               └────────────┴────────────┘")
print()
print("  TP = True Positive:   predicted YES, actually YES  (correct)")
print("  TN = True Negative:   predicted NO,  actually NO   (correct)")
print("  FP = False Positive:  predicted YES, actually NO   (false alarm)")
print("  FN = False Negative:  predicted NO,  actually YES  (missed it)")
print()

# ============================================================
# CONCRETE EXAMPLE
# ============================================================

print("=== EXAMPLE: EMAIL SPAM DETECTOR ===")
print()

np.random.seed(42)

# Simulate a model's predictions
actual    = np.array([1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
predicted = np.array([1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0])

labels = {(1,1): "TP", (0,0): "TN", (0,1): "FP", (1,0): "FN"}

print(f"20 emails: 8 actually spam, 12 actually not spam")
print()
print("  actual:    {0}".format("".join(str(x) for x in actual)))
print("  predicted: {0}".format("".join(str(x) for x in predicted)))
print()

TP = np.sum((predicted == 1) & (actual == 1))
TN = np.sum((predicted == 0) & (actual == 0))
FP = np.sum((predicted == 1) & (actual == 0))
FN = np.sum((predicted == 0) & (actual == 1))

print(f"  TP (spam, caught):        {TP}")
print(f"  TN (not spam, correct):   {TN}")
print(f"  FP (not spam, flagged):   {FP}  ← false alarms")
print(f"  FN (spam, missed):        {FN}  ← spam got through")
print()

accuracy = (TP + TN) / (TP + TN + FP + FN)
print(f"  Accuracy = (TP + TN) / total = ({TP} + {TN}) / {len(actual)} = {accuracy:.0%}")
print()

# ============================================================
# PRECISION — "HOW TRUSTWORTHY ARE MY POSITIVE PREDICTIONS?"
# ============================================================

print("=== PRECISION ===")
print()
print("  Precision = TP / (TP + FP)")
print()
print('  Question: "Of everything I flagged as spam, how much really WAS spam?"')
print()

precision = TP / (TP + FP) if (TP + FP) > 0 else 0
print(f"  Precision = {TP} / ({TP} + {FP}) = {TP} / {TP + FP} = {precision:.2%}")
print()
print(f"  {precision:.0%} of what we flagged as spam was actually spam.")
print(f"  {FP} false alarms — legitimate emails wrongly sent to spam folder.")
print()
print("  HIGH precision = few false alarms")
print("  When precision matters: spam filter (you don't want real emails in spam)")
print()

# ============================================================
# RECALL — "HOW MANY ACTUAL POSITIVES DID I CATCH?"
# ============================================================

print("=== RECALL (aka SENSITIVITY) ===")
print()
print("  Recall = TP / (TP + FN)")
print()
print('  Question: "Of all the ACTUAL spam, how much did I catch?"')
print()

recall = TP / (TP + FN) if (TP + FN) > 0 else 0
print(f"  Recall = {TP} / ({TP} + {FN}) = {TP} / {TP + FN} = {recall:.2%}")
print()
print(f"  We caught {recall:.0%} of all spam. {FN} spam emails got through.")
print()
print("  HIGH recall = few missed positives")
print("  When recall matters: cancer detection (you don't want to miss a real case)")
print()

# ============================================================
# THE TRADEOFF
# ============================================================

print("=== PRECISION vs RECALL TRADEOFF ===")
print()
print("You can't maximize BOTH. There's always a tension:")
print()
print("  Flag MORE things as positive:")
print("    → Recall goes UP  (you catch more real positives)")
print("    → Precision goes DOWN (more false alarms)")
print()
print("  Flag FEWER things as positive:")
print("    → Precision goes UP (what you flag is more likely correct)")
print("    → Recall goes DOWN (you miss more real positives)")
print()

# Demonstrate with different thresholds
print("Example with different thresholds:")
print()

np.random.seed(42)
n = 100
actual_labels = np.array([1]*20 + [0]*80)
scores = np.where(actual_labels == 1,
                  np.random.uniform(0.3, 0.9, n),
                  np.random.uniform(0.1, 0.6, n))

print(f"  Threshold  Precision  Recall   Flagged")
print(f"  ─────────  ─────────  ──────   ───────")
for thresh in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
    preds = (scores >= thresh).astype(int)
    tp = np.sum((preds == 1) & (actual_labels == 1))
    fp = np.sum((preds == 1) & (actual_labels == 0))
    fn = np.sum((preds == 0) & (actual_labels == 1))
    p = tp / (tp + fp) if (tp + fp) > 0 else 0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0
    flagged = np.sum(preds)
    print(f"    {thresh:.1f}       {p:5.1%}      {r:5.1%}     {flagged:3d}")

print()
print("As threshold goes UP: precision improves but recall drops.")
print("As threshold goes DOWN: recall improves but precision drops.")
print()

# ============================================================
# F1 SCORE — THE BALANCE
# ============================================================

print("=== F1 SCORE ===")
print()
print("  F1 = 2 × (precision × recall) / (precision + recall)")
print()
print("F1 is the harmonic mean of precision and recall.")
print("It's high ONLY when BOTH precision and recall are high.")
print()

f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
print(f"  Our model:")
print(f"    Precision = {precision:.2%}")
print(f"    Recall    = {recall:.2%}")
print(f"    F1        = {f1:.2%}")
print()

# Show why harmonic mean
print("Why harmonic mean, not regular average?")
print()
print("  If precision=1.0, recall=0.01:")
print(f"    Regular average = {(1.0 + 0.01) / 2:.2f}  (looks okay — but it's terrible!)")
f1_bad = 2 * (1.0 * 0.01) / (1.0 + 0.01)
print(f"    F1 (harmonic)   = {f1_bad:.2f}  (correctly says this is bad)")
print()
print("F1 punishes imbalance. You can't game it by being good at only one thing.")
print()

# ============================================================
# WHICH METRIC WHEN?
# ============================================================

print("=== WHICH METRIC TO USE? ===")
print()
print("It depends on what mistakes COST more:")
print()
print("  ┌─────────────────────────┬────────────────┬──────────────────────────┐")
print("  │ Scenario                │ Optimize for   │ Why                      │")
print("  ├─────────────────────────┼────────────────┼──────────────────────────┤")
print("  │ Spam filter             │ Precision      │ Don't lose real emails   │")
print("  │ Cancer screening        │ Recall         │ Don't miss any cancer    │")
print("  │ Search engine results   │ Precision      │ Top results must be good │")
print("  │ Fraud detection         │ Recall         │ Catch all fraud          │")
print("  │ General / balanced      │ F1             │ Balance both             │")
print("  └─────────────────────────┴────────────────┴──────────────────────────┘")
print()
print("There's no universal 'best metric.' It depends on your problem.")
print()

# ============================================================
# CONFUSION MATRIX FOR MULTI-CLASS
# ============================================================

print("=== MULTI-CLASS: IT STILL WORKS ===")
print()

classes = ["cat", "dog", "bird"]
actual_multi    = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 1, 2])
predicted_multi = np.array([0, 0, 2, 1, 1, 0, 2, 2, 1, 0, 0, 2])

n_classes = len(classes)
conf_matrix = np.zeros((n_classes, n_classes), dtype=int)
for a, p in zip(actual_multi, predicted_multi):
    conf_matrix[a][p] += 1

print("Confusion matrix (rows = actual, cols = predicted):")
print()
print(f"              predicted")
print(f"              {'  '.join(f'{c:>4s}' for c in classes)}")
for i, cls in enumerate(classes):
    row = "  ".join(f"{conf_matrix[i][j]:4d}" for j in range(n_classes))
    print(f"  actual {cls:4s}  {row}")
print()
print("Read it as: row=what it IS, column=what we PREDICTED")
print("  Diagonal = correct predictions")
print("  Off-diagonal = mistakes")
print()

total_correct = np.trace(conf_matrix)
total = len(actual_multi)
print(f"Overall accuracy: {total_correct}/{total} = {total_correct/total:.1%}")
print()

# Per-class metrics
print("Per-class precision and recall:")
for i, cls in enumerate(classes):
    tp = conf_matrix[i][i]
    fp = conf_matrix[:, i].sum() - tp  # others predicted as this class
    fn = conf_matrix[i, :].sum() - tp  # this class predicted as others
    p = tp / (tp + fp) if (tp + fp) > 0 else 0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0
    print(f"  {cls:4s}: precision={p:.2f}  recall={r:.2f}")
print()

# ============================================================
# SUMMARY
# ============================================================

print("=== SUMMARY ===")
print()
print("  Accuracy  = (TP + TN) / total         → overall correctness")
print("  Precision = TP / (TP + FP)             → 'of my YES predictions, how many right?'")
print("  Recall    = TP / (TP + FN)             → 'of actual YES, how many did I find?'")
print("  F1        = harmonic mean of P and R   → balanced score")
print()
print("  Always look at the CONFUSION MATRIX first. Then pick the right metric.")
print("  Accuracy alone is almost NEVER enough.")
