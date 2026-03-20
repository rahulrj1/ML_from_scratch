"""
Topic 0.1: Vectors, Matrices & Dot Products
============================================
Run this file and read the output. Modify things. Break things. Learn.

KEY TAKEAWAYS:
  - A vector is a list of numbers. Everything in ML (data, embeddings, pixels) is a vector.
  - A matrix is a grid of numbers. Datasets and model weights are matrices.
  - The dot product is what a single neuron computes: dot(inputs, weights) + bias.
  - Matrix multiplication = many dot products at once. This is why ML needs GPUs.
  - Dot product measures similarity — this is how RAG/vector search finds relevant docs.

NUMPY SHAPE GOTCHAS (discussed in detail):
  - 1D array shape (4,) is NOT the same as 2D row (1, 4) or column (4, 1).
  - .T on a 1D array does NOTHING — there are no rows/columns to swap.
  - matrix @ 1D_vector works because NumPy auto-treats 1D as a column vector.
  - matrix (3,4) @ 2D_row (1,4) FAILS — inner dimensions 4 ≠ 1 don't match.
  - Use .reshape(-1, 1) or .reshape(1, -1) to force 1D into 2D when needed.
"""

import numpy as np

# ============================================================
# VECTORS
# ============================================================

# A vector is just a list of numbers
house = np.array([3, 1500, 2])  # 3 bedrooms, 1500 sqft, 2 bathrooms
pixel = np.array([255, 128, 0])  # an orange pixel (RGB)

print("=== VECTORS ===")
print(f"House features: {house}")
print(f"Shape: {house.shape}")  # (3,) means 1D array with 3 elements
print(f"Dimension: {house.ndim}, Size: {house.size}")
print()

# Vector arithmetic — element-wise operations
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

print(f"a = {a}")
print(f"b = {b}")
print(f"a + b = {a + b}")      # [5, 7, 9]
print(f"a * b = {a * b}")      # [4, 10, 18]  (element-wise, NOT dot product)
print(f"a * 3 = {a * 3}")      # [3, 6, 9]   (scalar multiplication)
print()

# ============================================================
# DOT PRODUCT
# ============================================================

print("=== DOT PRODUCT ===")
dot_result = np.dot(a, b)  # (1*4) + (2*5) + (3*6) = 32
print(f"dot(a, b) = {dot_result}")
print(f"Manual check: {1*4} + {2*5} + {3*6} = {1*4 + 2*5 + 3*6}")
print()

# The neuron analogy
inputs = np.array([0.5, 0.8, 0.2])   # some input data
weights = np.array([0.4, -0.3, 0.9])  # learned weights
bias = 0.1

neuron_output = np.dot(inputs, weights) + bias
print("=== A SINGLE NEURON ===")
print(f"Inputs:  {inputs}")
print(f"Weights: {weights}")
print(f"Bias:    {bias}")
print(f"Output:  dot({inputs}, {weights}) + {bias}")
print(f"       = ({inputs[0]}×{weights[0]}) + ({inputs[1]}×{weights[1]}) + ({inputs[2]}×{weights[2]}) + {bias}")
print(f"       = {inputs[0]*weights[0]:.2f} + {inputs[1]*weights[1]:.2f} + {inputs[2]*weights[2]:.2f} + {bias}")
print(f"       = {neuron_output:.2f}")
print()

# Dot product as similarity measure
print("=== DOT PRODUCT AS SIMILARITY ===")
cat_embedding = np.array([0.9, 0.1, 0.8, 0.2])
dog_embedding = np.array([0.85, 0.15, 0.75, 0.25])
car_embedding = np.array([0.1, 0.9, 0.2, 0.8])

print(f"cat · dog = {np.dot(cat_embedding, dog_embedding):.3f}  (similar — both animals)")
print(f"cat · car = {np.dot(cat_embedding, car_embedding):.3f}  (different — animal vs vehicle)")
print()

# ============================================================
# MATRICES
# ============================================================

print("=== MATRICES ===")
# A dataset: 4 houses, each with 3 features (bedrooms, sqft, bathrooms)
dataset = np.array([
    [3, 1500, 2],
    [4, 2000, 3],
    [2, 800, 1],
    [5, 3000, 4],
])
print(f"Dataset (4 houses × 3 features):\n{dataset}")
print(f"Shape: {dataset.shape}")  # (4, 3)
print(f"Row 0 (first house): {dataset[0]}")
print(f"Column 1 (all sqft): {dataset[:, 1]}")
print()

# ============================================================
# MATRIX-VECTOR MULTIPLICATION — A NEURAL NETWORK LAYER
# ============================================================

print("=== MATRIX × VECTOR = NEURAL NETWORK LAYER ===")
# This is literally what a single layer does:
# Take a 3-element input, produce a 2-element output

x = np.array([1.0, 2.0, 3.0])  # input vector (3 features)

W = np.array([
    [0.2, 0.4, 0.1],   # weights for output neuron 0
    [0.5, -0.3, 0.8],  # weights for output neuron 1
])  # shape: (2, 3) — 2 output neurons, each looking at 3 inputs

b = np.array([0.1, -0.2])  # bias for each output neuron

output = np.dot(W, x) + b  # matrix-vector multiply + bias

print(f"Input x (3 features):  {x}")
print(f"Weight matrix W (2×3):\n{W}")
print(f"Bias b: {b}")
print(f"Output = W·x + b = {output}")
print()
print("What happened:")
print(f"  Neuron 0: dot({W[0]}, {x}) + {b[0]} = {np.dot(W[0], x):.1f} + {b[0]} = {np.dot(W[0], x) + b[0]:.1f}")
print(f"  Neuron 1: dot({W[1]}, {x}) + {b[1]} = {np.dot(W[1], x):.1f} + {b[1]} = {np.dot(W[1], x) + b[1]:.1f}")
print()
print("We just transformed a 3D input into a 2D output. That's a layer!")
print()

# ============================================================
# MATRIX-MATRIX MULTIPLICATION — PROCESSING A BATCH
# ============================================================

print("=== MATRIX × MATRIX = PROCESSING A BATCH ===")
# In real training, you don't process one input at a time.
# You process a BATCH of inputs simultaneously.

X_batch = np.array([
    [1.0, 2.0, 3.0],   # input 0
    [0.5, 1.5, 2.5],   # input 1
    [2.0, 0.0, 1.0],   # input 2
])  # shape: (3 inputs, 3 features)

# X_batch @ W.T does the layer computation for ALL inputs at once
# (.T means transpose — flip rows and columns)
output_batch = X_batch @ W.T + b

print(f"Batch input (3 samples × 3 features):\n{X_batch}")
print(f"Batch output (3 samples × 2 neurons):\n{output_batch}")
print()
print("Each row of the output is one sample processed through the layer.")
print("This is why GPUs are fast — all 3 samples computed in parallel!")

print("=== DOT PRODUCT AS SIMILARITY MEASURE ===")

query     = np.array([0.9, 0.1, 0.3, 0.7])
doc_a     = np.array([0.8, 0.2, 0.4, 0.6])   # "how to train a dog"
doc_b     = np.array([0.1, 0.9, 0.8, 0.2])   # "car engine repair"
doc_c     = np.array([0.85, 0.15, 0.35, 0.65]) # "puppy training tips"

doc_matrix = np.array([doc_a, doc_b, doc_c])  # stack into a 3×4 matrix
scores = doc_matrix @ query                     # one operation, all 3 scores
# This works because: (3, 4) @ (4,) → NumPy treats (4,) as column → (3, 4) @ (4, 1) → (3,)
# If query was shape (1, 4) instead, this would FAIL. You'd need: doc_matrix @ query.T
print(scores)                                   # [1.280, 0.560, 1.340]
best_idx = np.argmax(scores)                    # index of highest score — used everywhere in ML for predictions
print(f"Most relevant: doc {best_idx} with score {scores[best_idx]:.3f}")

# ============================================================
# SHAPE CHEAT SHEET (reference)
# ============================================================
#
#   np.array([1, 2, 3])            → shape (3,)    — 1D, no rows/columns
#   np.array([[1, 2, 3]])          → shape (1, 3)  — 2D row vector
#   np.array([[1], [2], [3]])      → shape (3, 1)  — 2D column vector
#
#   v = np.array([1, 2, 3])
#   v.T                            → shape (3,)    — SAME! transpose is no-op on 1D
#   v.reshape(1, -1)               → shape (1, 3)  — force into row
#   v.reshape(-1, 1)               → shape (3, 1)  — force into column
#
#   max(items, key=fn)             → finds item where fn(item) is largest
#   np.argmax(array)               → index of the max value in array
