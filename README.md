# Sparse autoencoders & feature Superposition

A PyTorch implementation and experimental study of **Sparse Autoencoders (SAEs)** reproducing core methodologies from Anthropic's *Towards Monosemanticity* (Bricken et al., 2023).

Developed as a project for the **Machine Learning** course (MSc in Computer Science, University of Padua).

---

## Overview

In neural networks and Large Language Models, models often represent more concepts than they have dimensions, a phenomenon known as **superposition**, which leads to **polysemantic neurons**. This repository implements an overcomplete Sparse Autoencoder trained on controlled synthetic superposition benchmarks (N = 20 ground-truth features compressed into M = 5 activation dimensions) to disentangle polysemantic activations into monosemantic latent directions.

### Core Architectural Features
* **Untied Encoder-Decoder Architecture:** Independent linear encoder (with ReLU activation) and decoder dictionaries to prevent interference.
* **Geometric Integrity (Unit-Norm Constraint):** Enforces unit norm on decoder dictionary columns via three-step constrained optimization (tangent gradient projection, Adam update, and explicit L2 normalization) to prevent scaling-freedom exploits.
* **Geometric Median Bias Initialization:** Centers input activations using the geometric median of the activation distribution[cite: 171].
* **Dead-Neuron Resampling:** Periodically detects inactive latent units and reinitializes their encoder/decoder weights toward high-loss residual directions while resetting their Adam optimizer states[cite: 171].

---

## Experiments & Key Findings

Feature recovery quality is evaluated using **Mean Maximum Cosine Similarity (MMCS)** against ground-truth directions alongside **sparsity (L0)** and **decoder norm**[cite: 171]:

1. **Constrained vs. Unconstrained Training (*Scaling Freedom*):** Without unit-norm constraints, the unconstrained SAE shrinks decoder norms (~0.80) to artificially lower the L1 penalty while degrading sparsity (L0 = 4.0 vs. 3.6)[cite: 171].
2. **Linear vs. ReLU Activations:** Removing ReLU non-linearity yields nearly identical feature alignment (\text{MMCS} \approx 0.82\text{--}0.83) but increases L0 (5.8 vs. 3.5), confirming that superposition geometry—rather than activation non-linearity—is the primary bottleneck[cite: 171].
3. **Aggressive Non-Linearity (GELU + Quadratic Interactions) & L1 Tuning:** Under warped activations with inter-neuron quadratic interactions, L0 spikes to 7.6; increasing the regularization penalty (\lambda = 0.005 \rightarrow 0.1) restores sparsity (L0 = 3.33) without degrading MMCS[cite: 171].
4. **Long-Horizon Training Dynamics (300 \rightarrow 10,000 epochs):** Extended training improves dictionary alignment (\text{MMCS}: 0.83 \rightarrow 0.88) at the cost of denser representations (L0: 3.58 \rightarrow 5.36), highlighting the need to co-scale regularization over long horizons[cite: 171].

---

## Repository Structure

```text
├── sparse_autoencoders_superposition.ipynb   # Main notebook with code, plots, and analysis
└── README.md                                 # Project overview and experimental summary
```

## Quickstart

### Requirements

- Python 3.9+
- `torch`[cite: 171]
- `geom-median`[cite: 171]
- `numpy`[cite: 171]
- `matplotlib`[cite: 171]

### Running Locally

Bash

# 

```
git clone https://github.com/JackLoat/sparse-autoencoders-superposition.git
cd sparse-autoencoders-superposition
pip install torch geom-median numpy matplotlib
jupyter notebook sparse_autoencoders_superposition.ipynb
```

## References

- Bricken, T., Templeton, A., Batson, J., Chen, B., Jermyn, A., Conerly, T., Turner, N., Anil, C., Denison, C., Askell, A., Lasenby, R., Wu, Y., Kravec, S., Schiefer, N., Maxwell, T., Joseph, N., Hatfield-Dodds, Z., Tamkin, A., Nguyen, K., McLean, B., Burke, J. E., Hume, T., Carter, S., Henighan, T., & Olah, C. (2023). *Towards Monosemanticity: Decomposing Language Models With Dictionary Learning*. Transformer Circuits Thread[cite: 171].
