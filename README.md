# ML From Scratch

Implementing machine learning algorithms from the ground up -- starting with pure NumPy, eventually building up to PyTorch and HuggingFace.

The goal: build deep intuition for how ML actually works, from linear regression through transformers.

## What's here

Each file is a self-contained, runnable script that explains a concept, implements it from scratch, and includes exercises.

```
phase1/          # Classical ML
  1_1  ML Workflow        — problem framing, data splits, evaluation
  1_2  Linear Regression  — gradient descent, cost functions
  1_3  Logistic Regression — sigmoid, decision boundaries
  1_4  Evaluation Metrics  — precision, recall, F1, AUC-ROC
  1_5  Overfitting        — bias-variance, L1/L2 regularization
  ...
```

## Running

```bash
python -m venv myvenv
source myvenv/bin/activate
pip install -r requirements.txt
python phase1/1_2_linear_regression.py
```

## Roadmap

See [ROADMAP.md](ROADMAP.md) for the full 8-phase plan (math foundations through production ML).

| Phase | Topics |
|-------|--------|
| 0. Math Intuition | Vectors, matrices, calculus, probability, loss functions |
| 1. Classical ML | Linear/logistic regression, trees, evaluation, feature engineering |
| 2. Deep Learning | PyTorch, perceptrons, backprop, training loops |
| 3. CNNs | Convolutions, transfer learning, image classification |
| 4. Sequence Models | RNNs, LSTMs, attention |
| 5. Transformers | Self-attention, positional encoding, mini GPT |
| 6. Fine-Tuning | HuggingFace, LoRA, instruction tuning |
| 7. Scale & Production | Distributed training, quantization, serving |
