"""
Topic 0.3: Derivatives & Chain Rule
=====================================
The engine behind backpropagation. Every weight update in every neural
network ever trained relies on exactly these concepts.

KEY TAKEAWAYS:
  - Derivative = "if I nudge the input, how much does the output change?"
  - You can estimate any derivative numerically: (f(x+h) - f(x-h)) / (2h)
  - Chain rule: derivative of f(g(x)) = f'(g(x)) * g'(x)
  - Backpropagation IS the chain rule, applied from loss back to each weight.
  - PyTorch autograd does this automatically — but you should know what it's doing.
"""

import numpy as np

# ============================================================
# WHAT IS A DERIVATIVE? — NUMERICAL INTUITION
# ============================================================

print("=== WHAT IS A DERIVATIVE? ===")

def f(x):
    return x ** 2

x = 3.0
h = 0.0001  # tiny nudge

numerical_derivative = (f(x + h) - f(x - h)) / (2 * h)
analytical_derivative = 2 * x  # d/dx of x² = 2x

print(f"f(x) = x²")
print(f"At x = {x}:")
print(f"  Numerical derivative:  {numerical_derivative:.6f}")
print(f"  Analytical derivative: {analytical_derivative:.6f}")
print(f"  (They match!)")
print()

# ============================================================
# DERIVATIVE = SLOPE = SENSITIVITY
# ============================================================

print("=== DERIVATIVE AT DIFFERENT POINTS ===")
print("f(x) = x²")
print()
for x_val in [-3, -1, 0, 1, 3, 5]:
    deriv = 2 * x_val
    direction = "increasing" if deriv > 0 else "decreasing" if deriv < 0 else "flat (minimum!)"
    print(f"  x = {x_val:2d}  →  f(x) = {x_val**2:2d}  →  f'(x) = {deriv:3d}  →  {direction}")

print()
print("When derivative is positive → function is going UP → increase x, output increases")
print("When derivative is negative → function is going DOWN → increase x, output decreases")
print("When derivative is zero → you're at a minimum (or maximum) → this is what training seeks!")
print()

# ============================================================
# DERIVATIVES OF COMMON ML OPERATIONS
# ============================================================

print("=== DERIVATIVES OF COMMON ML OPERATIONS ===")

def numerical_grad(func, x, h=1e-5):
    """Estimate derivative of any function numerically."""
    return (func(x + h) - func(x - h)) / (2 * h)

# ReLU: f(x) = max(0, x)
def relu(x):
    return np.maximum(0, x)

# Sigmoid: f(x) = 1 / (1 + e^(-x))
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

print("ReLU:    f(x) = max(0, x)")
print(f"  f'(-2) = {numerical_grad(relu, -2.0):.1f}   (input negative → gradient is 0, neuron is 'dead')")
print(f"  f'( 2) = {numerical_grad(relu, 2.0):.1f}   (input positive → gradient is 1, passes straight through)")
print()

print("Sigmoid: f(x) = 1 / (1 + e^(-x))")
print(f"  f'(-3) = {numerical_grad(sigmoid, -3.0):.4f}  (far from 0 → tiny gradient)")
print(f"  f'( 0) = {numerical_grad(sigmoid, 0.0):.4f}  (at center → maximum gradient of 0.25)")
print(f"  f'( 3) = {numerical_grad(sigmoid, 3.0):.4f}  (far from 0 → tiny gradient)")
print("  This is the 'vanishing gradient' problem — sigmoid squashes gradients at extremes!")
print()

# ============================================================
# THE CHAIN RULE
# ============================================================

print("=== THE CHAIN RULE ===")
print("If y = f(g(x)), then dy/dx = f'(g(x)) × g'(x)")
print()

# Example: y = (3x + 2)²
# outer: f(u) = u²    → f'(u) = 2u
# inner: g(x) = 3x+2  → g'(x) = 3
# chain: dy/dx = 2(3x+2) × 3 = 6(3x+2)

def composed(x):
    return (3 * x + 2) ** 2

x = 1.0
numerical = numerical_grad(composed, x)
analytical = 6 * (3 * x + 2)  # chain rule result

print(f"y = (3x + 2)²")
print(f"At x = {x}:")
print(f"  inner g(x)  = 3({x}) + 2 = {3*x + 2}")
print(f"  outer f(g)  = {3*x + 2}² = {(3*x + 2)**2}")
print(f"  dy/dx = 2·(3x+2) · 3 = 6·(3x+2) = 6·{3*x+2} = {analytical}")
print(f"  Numerical check: {numerical:.4f}")
print()

# ============================================================
# CHAIN RULE IN A MINI NEURAL NETWORK
# ============================================================

print("=== CHAIN RULE IN A NEURAL NETWORK ===")
print("Forward pass: x → [×w] → [+b] → [relu] → [×w2] → loss")
print()

# A single-neuron, single-input "network"
x = 2.0       # input
w1 = 0.5      # weight layer 1
b1 = 0.1      # bias layer 1
w2 = -0.3     # weight layer 2
target = 1.0  # what we want the output to be

# Forward pass — step by step
z1 = x * w1           # step 1: multiply by weight
z2 = z1 + b1          # step 2: add bias
a1 = max(0, z2)       # step 3: ReLU activation
z3 = a1 * w2          # step 4: second layer
loss = (z3 - target) ** 2  # step 5: squared error loss

print("FORWARD PASS:")
print(f"  z1 = x * w1           = {x} × {w1} = {z1}")
print(f"  z2 = z1 + b1          = {z1} + {b1} = {z2}")
print(f"  a1 = relu(z2)         = relu({z2}) = {a1}")
print(f"  z3 = a1 * w2          = {a1} × {w2} = {z3}")
print(f"  loss = (z3 - target)² = ({z3} - {target})² = {loss}")
print()

# Backward pass — chain rule step by step
dloss_dz3 = 2 * (z3 - target)     # d/dz3 of (z3 - target)²
dz3_da1   = w2                     # d/da1 of (a1 * w2)
da1_dz2   = 1.0 if z2 > 0 else 0  # d/dz2 of relu(z2)
dz2_dz1   = 1.0                    # d/dz1 of (z1 + b1)
dz1_dw1   = x                      # d/dw1 of (x * w1)

# Chain them all together
dloss_dw1 = dloss_dz3 * dz3_da1 * da1_dz2 * dz2_dz1 * dz1_dw1

print("BACKWARD PASS (chain rule):")
print(f"  dloss/dz3 = 2(z3 - target)       = 2({z3} - {target}) = {dloss_dz3}")
print(f"  dz3/da1   = w2                    = {dz3_da1}")
print(f"  da1/dz2   = 1 if z2>0 else 0      = {da1_dz2}")
print(f"  dz2/dz1   = 1                     = {dz2_dz1}")
print(f"  dz1/dw1   = x                     = {dz1_dw1}")
print()
print(f"  dloss/dw1 = {dloss_dz3} × {dz3_da1} × {da1_dz2} × {dz2_dz1} × {dz1_dw1}")
print(f"            = {dloss_dw1}")
print()

# Verify numerically
def network_loss(w1_val):
    z1 = x * w1_val
    z2 = z1 + b1
    a1 = max(0, z2)
    z3 = a1 * w2
    return (z3 - target) ** 2

numerical_dloss_dw1 = numerical_grad(network_loss, w1)
print(f"  Numerical verification: {numerical_dloss_dw1:.6f}")
print(f"  Analytical (chain rule): {dloss_dw1:.6f}")
print(f"  Match: {abs(numerical_dloss_dw1 - dloss_dw1) < 1e-4}")
print()

# ============================================================
# GRADIENT DESCENT — USING THE DERIVATIVE TO LEARN
# ============================================================

print("=== GRADIENT DESCENT — PUTTING IT ALL TOGETHER ===")

x = 2.0
w1 = 0.5
b1 = 0.1
w2 = -0.3
target = 1.0
learning_rate = 0.01

print(f"Goal: adjust w1 so the network output gets closer to {target}")
print(f"Starting w1 = {w1}")
print()

for step in range(10):
    # Forward
    z1 = x * w1
    z2 = z1 + b1
    a1 = max(0, z2)
    z3 = a1 * w2
    loss = (z3 - target) ** 2

    # Backward (chain rule)
    dloss_dz3 = 2 * (z3 - target)
    dz3_da1 = w2
    da1_dz2 = 1.0 if z2 > 0 else 0.0
    dz2_dz1 = 1.0
    dz1_dw1 = x
    grad = dloss_dz3 * dz3_da1 * da1_dz2 * dz2_dz1 * dz1_dw1

    # Update
    w1 = w1 - learning_rate * grad

    print(f"  Step {step}: loss = {loss:.4f}, grad = {grad:.4f}, w1 = {w1:.4f}")

print()
print(f"After 10 steps, loss went from the initial value down — w1 is learning!")
print("This is the ENTIRE training algorithm: forward → loss → backward → update → repeat.")
