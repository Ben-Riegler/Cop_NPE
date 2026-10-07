# Copula Neural Posterior Estimation

This repository contains the main building blocks of the Copula NPE paper.

- `tasks.py` — the simulation tasks (data-generating processes, exact densities where available, the dependency-graph masks) and the shared train/test data protocol.
- `copnpe.py` — Copula NPE: marginal MDNs, the masked Gaussian-mixture copula, the training pipeline of Algorithm 1, LPP and the conditional mutual information diagnostic.
- `models.py` — the baseline models: MDN-NPE, NSF-NPE and FMPE via `sbi`, and a PyTorch Simformer with graph-structured attention masks.
- `example.py` — fits Copula NPE on one task and prints LPP next to the prior and true-posterior anchors.
- `requirements.txt` — Python dependencies.
