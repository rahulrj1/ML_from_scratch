"""
Topic 0.2: Matrix Multiplication
=================================
Going deeper: full matrix-matrix multiply, shape rules, transpose patterns,
broadcasting, and why GPUs are perfect for this.

KEY TAKEAWAYS:
  - (m, n) @ (n, p) → (m, p). Inner dims must match, outer dims survive.
  - Matrix multiply = all rows dotted with all columns. Each result element is independent.
  - Neural net layer on a batch: (batch_size, in_features) @ (in_features, out_features) → (batch_size, out_features)
  - GPUs are fast because every element in the result can be computed in parallel.
  - When shapes don't match, check: do I need a .T? a .reshape? did I swap the order?
"""

import numpy as np

# ============================================================
# MATRIX × MATRIX — THE FULL OPERATION
# ============================================================

print("=== MATRIX × MATRIX ===")

A = np.array([
    [1, 2, 3],
    [4, 5, 6],
])  # (2, 3)

B = np.array([
    [7, 10],
    [8, 11],
    [9, 12],
])  # (3, 2)

C = A @ B  # (2, 3) @ (3, 2) → (2, 2)

print(f"A (2×3):\n{A}")
print(f"B (3×2):\n{B}")
print(f"A @ B (2×2):\n{C}")
print()
print("How each element was computed:")
print(f"  C[0,0] = dot(A[row 0], B[col 0]) = dot({A[0]}, {B[:, 0]}) = {C[0, 0]}")
print(f"  C[0,1] = dot(A[row 0], B[col 1]) = dot({A[0]}, {B[:, 1]}) = {C[0, 1]}")
print(f"  C[1,0] = dot(A[row 1], B[col 0]) = dot({A[1]}, {B[:, 0]}) = {C[1, 0]}")
print(f"  C[1,1] = dot(A[row 1], B[col 1]) = dot({A[1]}, {B[:, 1]}) = {C[1, 1]}")
print()

# ============================================================
# THE SHAPE RULE — YOUR #1 DEBUGGING TOOL
# ============================================================

print("=== THE SHAPE RULE ===")
print("(m, n) @ (n, p) → (m, p)")
print("       ^^^  ^^^")
print("       must match — they get consumed")
print()

shapes_demo = [
    ((2, 3), (3, 4), "✓"),   # inner 3 == 3
    ((5, 10), (10, 2), "✓"), # inner 10 == 10
    ((32, 784), (784, 128), "✓"),  # a real NN layer: batch of 32, 784→128
    ((2, 3), (4, 5), "✗"),   # inner 3 ≠ 4
]

for shape_a, shape_b, valid in shapes_demo:
    inner_match = shape_a[1] == shape_b[0]
    if inner_match:
        result = f"({shape_a[0]}, {shape_b[1]})"
    else:
        result = "ERROR"
    print(f"  {shape_a} @ {shape_b} → {result}  {valid}")
print()

# ============================================================
# NEURAL NETWORK LAYER = ONE MATRIX MULTIPLY
# ============================================================

print("=== NEURAL NETWORK LAYER ON A BATCH ===")

np.random.seed(42)

batch_size = 4
in_features = 3
out_features = 2

X = np.random.randn(batch_size, in_features).round(2)   # (4, 3) — 4 samples
W = np.random.randn(in_features, out_features).round(2)  # (3, 2) — weights
b = np.random.randn(out_features).round(2)                # (2,)   — bias

output = X @ W + b   # (4, 3) @ (3, 2) + (2,) → (4, 2)

print(f"Input X  shape: {X.shape}  — {batch_size} samples, {in_features} features each")
print(f"Weights  shape: {W.shape}  — transforms {in_features} features → {out_features} outputs")
print(f"Bias     shape: {b.shape}")
print(f"Output   shape: {output.shape}  — {batch_size} samples, {out_features} outputs each")
print()
print(f"Input X:\n{X}")
print(f"Weights W:\n{W}")
print(f"Bias b: {b}")
print(f"Output (X @ W + b):\n{output}")
print()

# Verify: the first row of the output should be X[0] dotted with each column of W, plus bias
row0_manual = np.array([np.dot(X[0], W[:, 0]) + b[0], np.dot(X[0], W[:, 1]) + b[1]])
print(f"Manual check for sample 0: {row0_manual.round(4)}")
print(f"Matrix result for sample 0: {output[0].round(4)}")
print(f"Match: {np.allclose(row0_manual, output[0])}")
print()

# ============================================================
# TRANSPOSE — FLIPPING ROWS AND COLUMNS
# ============================================================

print("=== TRANSPOSE ===")

M = np.array([
    [1, 2, 3],
    [4, 5, 6],
])  # (2, 3)

print(f"M (2×3):\n{M}")
print(f"M.T (3×2):\n{M.T}")
print()

# Common pattern: weights stored as (out, in) but you need (in, out)
# In PyTorch, nn.Linear stores weights as (out_features, in_features)
# so you often see X @ W.T

W_pytorch_style = np.array([
    [0.1, 0.2, 0.3],   # neuron 0's weights for each input
    [0.4, 0.5, 0.6],   # neuron 1's weights for each input
])  # (2, 3) — 2 output neurons, 3 inputs — stored as (out, in)

x_single = np.array([1.0, 2.0, 3.0])  # (3,) — one sample

# To make shapes work: (3,) needs to meet (?, 3) on the left
# W is (2, 3) — wrong order. W.T is (3, 2) — now (3,) @ ??? doesn't work either
# The right way: W @ x gives (2, 3) @ (3,) → (2,) ✓
# OR for batches: X_batch @ W.T gives (batch, 3) @ (3, 2) → (batch, 2) ✓

print("PyTorch-style weights (out_features, in_features):")
print(f"  W shape: {W_pytorch_style.shape}")
print(f"  Single input:  W @ x     → {W_pytorch_style.shape} @ {x_single.shape} → result shape {(W_pytorch_style @ x_single).shape}")
print(f"  Batch input:   X @ W.T   → {X.shape} @ {W_pytorch_style.T.shape} → result shape {(X @ W_pytorch_style.T).shape}")
print()

# ============================================================
# ORDER MATTERS — A @ B ≠ B @ A
# ============================================================

print("=== ORDER MATTERS ===")

P = np.array([[1, 2], [3, 4]])  # (2, 2)
Q = np.array([[5, 6], [7, 8]])  # (2, 2)

print(f"P @ Q:\n{P @ Q}")
print(f"Q @ P:\n{Q @ P}")
print(f"Are they equal? {np.array_equal(P @ Q, Q @ P)}")
print()

# With non-square matrices, swapping might not even be possible
R = np.array([[1, 2, 3]])  # (1, 3)
S = np.array([[4], [5], [6]])  # (3, 1)

print(f"R @ S = {R.shape} @ {S.shape} → {(R @ S).shape}:\n{R @ S}")  # (1, 3) @ (3, 1) → (1, 1)
print(f"S @ R = {S.shape} @ {R.shape} → {(S @ R).shape}:\n{S @ R}")  # (3, 1) @ (1, 3) → (3, 3)
print("Same inputs, completely different results and shapes!")
print()

# ============================================================
# BROADCASTING — HOW BIAS ADDITION WORKS
# ============================================================

print("=== BROADCASTING (how bias addition works) ===")

# When we write X @ W + b, the shapes are:
#   (4, 2) + (2,)
# These shapes don't match! So how does it work?
# NumPy "broadcasts" the smaller array to match the larger one.

result_no_bias = X @ W  # (4, 2)
print(f"X @ W shape: {result_no_bias.shape}")  # (4, 2)
print(f"bias shape:  {b.shape}")                # (2,)
print()
print("NumPy broadcasts (2,) to match (4, 2):")
print("  It treats b as if it were repeated 4 times:")
print(f"  [[{b[0]}, {b[1]}],")
print(f"   [{b[0]}, {b[1]}],")
print(f"   [{b[0]}, {b[1]}],")
print(f"   [{b[0]}, {b[1]}]]")
print()
print("Broadcasting rules:")
print("  1. Align shapes from the RIGHT")
print("  2. Dimensions match if they're equal OR one of them is 1 (or missing)")
print("  3. The smaller array is 'stretched' to match")
print()
print("  (4, 2) + (2,)  → (4, 2) + (1, 2) → (4, 2)  ✓  bias added to every row")
print("  (4, 2) + (4,)  → (4, 2) + (1, 4) → ERROR     ✗  can't broadcast 4 into 2")
print()

# ============================================================
# REAL-WORLD SCALE
# ============================================================

print("=== REAL-WORLD SCALE ===")

# GPT-2 small: hidden size 768, vocab 50257
# A single attention layer does multiple matmuls of shape ~(batch, seq_len, 768) @ (768, 768)

batch, seq_len, hidden = 8, 512, 768
X_big = np.random.randn(batch, seq_len, hidden)  # 8 sequences, 512 tokens, 768-dim
W_big = np.random.randn(hidden, hidden)           # one projection matrix

import time
start = time.time()
result_big = X_big @ W_big
elapsed = time.time() - start

total_dots = batch * seq_len * hidden  # each element in result is one dot product
print(f"Shape: ({batch}, {seq_len}, {hidden}) @ ({hidden}, {hidden}) → {result_big.shape}")
print(f"Total dot products computed: {total_dots:,}")
print(f"Time on CPU: {elapsed*1000:.1f}ms")
print(f"On a GPU, this would be ~100x faster due to massive parallelism")
