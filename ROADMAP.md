# ML Model Training — Learning Roadmap

> A structured, hands-on roadmap for going from "I know backend & applied AI" to "I can train, fine-tune, and ship ML models."

---

## Phase 0: Mathematical Intuition (Foundation)

> You don't need to become a mathematician. You need enough intuition to understand *why* things work, not just *how* to call the API.


| #   | Topic                                                                      | Status |
| --- | -------------------------------------------------------------------------- | ------ |
| 0.1 | **Vectors, Matrices & Dot Products** — what a neural net actually computes | [x]    |
| 0.2 | **Matrix Multiplication** — batched operations, why GPUs love matrices     | [x]    |
| 0.3 | **Derivatives & Chain Rule** — the engine behind backpropagation           | [x]    |
| 0.4 | **Probability Basics** — distributions, Bayes' theorem, expectation        | [x]    |
| 0.5 | **Loss Functions (the math)** — MSE, cross-entropy, why they exist         | [x]    |


---

## Phase 1: Classical ML — Learn to Think About Data

> Classical ML teaches you the *discipline* of ML — how to frame problems, avoid pitfalls, and evaluate models. Skip this and you'll build broken deep learning models later.


| #   | Topic                                                                               | Status |
| --- | ----------------------------------------------------------------------------------- | ------ |
| 1.1 | **The ML Workflow** — problem framing, data splits, evaluation, iteration           | [x]    |
| 1.2 | **Linear Regression** — gradient descent from scratch, cost functions               | [x]    |
| 1.3 | **Logistic Regression** — binary classification, sigmoid, decision boundaries       | [x]    |
| 1.4 | **Evaluation Metrics** — accuracy, precision, recall, F1, AUC-ROC, confusion matrix | [x]    |
| 1.5 | **Overfitting & Regularization** — bias-variance tradeoff, L1/L2, cross-validation  | [ ]    |
| 1.6 | **Decision Trees & Random Forests** — splits, information gain, ensembles           | [ ]    |
| 1.7 | **Feature Engineering** — scaling, encoding, missing values, feature selection      | [ ]    |
| 1.8 | **Unsupervised Learning Basics** — K-Means clustering, PCA                          | [ ]    |
| 1.9 | **Mini Project: Tabular Classification** — end-to-end with scikit-learn             | [ ]    |


---

## Phase 2: Deep Learning Fundamentals — Neurons to Networks

> This is where you go from "I use models" to "I understand models." PyTorch is your primary tool from here on.


| #   | Topic                                                                              | Status |
| --- | ---------------------------------------------------------------------------------- | ------ |
| 2.1 | **PyTorch Essentials** — tensors, autograd, device management                      | [ ]    |
| 2.2 | **The Perceptron & Activation Functions** — ReLU, sigmoid, tanh, softmax           | [ ]    |
| 2.3 | **Building a Neural Network from Scratch** — forward pass, loss, backward pass     | [ ]    |
| 2.4 | **Training Loop Anatomy** — epochs, batches, optimizers (SGD, Adam), learning rate | [ ]    |
| 2.5 | **Backpropagation Deep Dive** — chain rule in action, computational graphs         | [ ]    |
| 2.6 | **Data Loading** — Dataset, DataLoader, transforms, batching                       | [ ]    |
| 2.7 | **Regularization in Practice** — dropout, batch norm, early stopping               | [ ]    |
| 2.8 | **Debugging Training** — loss curves, gradient issues, learning rate tuning        | [ ]    |
| 2.9 | **Mini Project: MNIST Digit Classifier** — full pipeline in PyTorch                | [ ]    |


---

## Phase 3: Convolutional Neural Networks (CNNs) — Vision

> CNNs teach you *spatial feature extraction* — a concept that carries over even to modern vision transformers.


| #   | Topic                                                                                 | Status |
| --- | ------------------------------------------------------------------------------------- | ------ |
| 3.1 | **Convolution Operation** — filters, stride, padding, feature maps                    | [ ]    |
| 3.2 | **Pooling & Architecture Patterns** — MaxPool, common architectures (LeNet, VGG idea) | [ ]    |
| 3.3 | **Transfer Learning** — using pretrained models, freezing layers, fine-tuning         | [ ]    |
| 3.4 | **Data Augmentation** — transforms for robustness                                     | [ ]    |
| 3.5 | **Mini Project: Image Classifier** — build, train, evaluate on a real dataset         | [ ]    |


---

## Phase 4: Sequence Models — Text & Time

> The full landscape from RNNs to the attention mechanism that replaced them.


| #   | Topic                                                                               | Status |
| --- | ----------------------------------------------------------------------------------- | ------ |
| 4.1 | **Recurrent Neural Networks (RNNs)** — hidden state, unrolling, vanishing gradients | [ ]    |
| 4.2 | **LSTMs & GRUs** — gating mechanisms, why they solve vanishing gradients            | [ ]    |
| 4.3 | **Word Embeddings** — Word2Vec, GloVe, learned embeddings                           | [ ]    |
| 4.4 | **Sequence-to-Sequence Models** — encoder-decoder, teacher forcing                  | [ ]    |
| 4.5 | **Attention Mechanism** — the key insight that changed everything                   | [ ]    |
| 4.6 | **Mini Project: Text Classifier or Sequence Predictor**                             | [ ]    |


---

## Phase 5: Transformers — The Architecture That Runs the World

> The architecture behind GPT, BERT, LLaMA, and most modern AI. Understanding it deeply is non-negotiable.


| #   | Topic                                                                   | Status |
| --- | ----------------------------------------------------------------------- | ------ |
| 5.1 | **Self-Attention from Scratch** — Q, K, V, scaled dot-product attention | [ ]    |
| 5.2 | **Multi-Head Attention** — why multiple heads, what each learns         | [ ]    |
| 5.3 | **Positional Encoding** — how transformers know word order              | [ ]    |
| 5.4 | **The Transformer Block** — LayerNorm, residual connections, FFN        | [ ]    |
| 5.5 | **Encoder vs Decoder Transformers** — BERT-style vs GPT-style           | [ ]    |
| 5.6 | **Tokenization** — BPE, WordPiece, SentencePiece                        | [ ]    |
| 5.7 | **Building a Mini GPT** — implement a small transformer from scratch    | [ ]    |
| 5.8 | **Mini Project: Train a character-level language model**                | [ ]    |


---

## Phase 6: Fine-Tuning & the HuggingFace Ecosystem

> Go from "I call APIs" to "I adapt models to my data."


| #   | Topic                                                                            | Status |
| --- | -------------------------------------------------------------------------------- | ------ |
| 6.1 | **HuggingFace Transformers Library** — models, tokenizers, pipelines             | [ ]    |
| 6.2 | **HuggingFace Datasets & Trainers** — loading, preprocessing, training API       | [ ]    |
| 6.3 | **Full Fine-Tuning** — fine-tune BERT for classification                         | [ ]    |
| 6.4 | **Parameter-Efficient Fine-Tuning (PEFT)** — LoRA, QLoRA, adapters               | [ ]    |
| 6.5 | **Fine-Tuning an LLM** — instruction tuning a small model (Llama 3.2 1B / Phi-3) | [ ]    |
| 6.6 | **Evaluation & Benchmarking** — perplexity, BLEU, ROUGE, human eval              | [ ]    |
| 6.7 | **Mini Project: Fine-tune a model on your own dataset**                          | [ ]    |


---

## Phase 7: Training at Scale & Production

> Where ML meets infrastructure — distributed training, quantization, and serving models in production.


| #   | Topic                                                                  | Status |
| --- | ---------------------------------------------------------------------- | ------ |
| 7.1 | **GPU Fundamentals** — CUDA cores, memory hierarchy, utilization       | [ ]    |
| 7.2 | **Mixed Precision Training** — FP16, BF16, loss scaling                | [ ]    |
| 7.3 | **Distributed Training** — DataParallel, DistributedDataParallel, FSDP | [ ]    |
| 7.4 | **Quantization** — INT8, INT4, GPTQ, AWQ, GGUF                         | [ ]    |
| 7.5 | **Experiment Tracking** — Weights & Biases, MLflow                     | [ ]    |
| 7.6 | **Model Serving** — TorchServe, vLLM, Triton, ONNX Runtime             | [ ]    |
| 7.7 | **MLOps Basics** — model registries, CI/CD for ML, monitoring          | [ ]    |


---

## Progress Tracker


| Phase                     | Topics | Completed | Progress              |
| ------------------------- | ------ | --------- | --------------------- |
| Phase 0 — Math Intuition  | 5      | 5         | ██████████ 100%       |
| Phase 1 — Classical ML    | 9      | 4         | ████░░░░░░ 44%        |
| Phase 2 — Deep Learning   | 9      | 0         | ░░░░░░░░░░ 0%         |
| Phase 3 — CNNs            | 5      | 0         | ░░░░░░░░░░ 0%         |
| Phase 4 — Sequence Models | 6      | 0         | ░░░░░░░░░░ 0%         |
| Phase 5 — Transformers    | 8      | 0         | ░░░░░░░░░░ 0%         |
| Phase 6 — Fine-Tuning     | 7      | 0         | ░░░░░░░░░░ 0%         |
| Phase 7 — Scale & Prod    | 7      | 0         | ░░░░░░░░░░ 0%         |
| **Total**                 | **56** | **9**     | █░░░░░░░░░ **16%**    |


