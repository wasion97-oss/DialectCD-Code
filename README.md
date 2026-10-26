# DialectCD: Explainable Dialectical Causal Discovery via Dual-Source Arguments and Meta-Arbitration



> Anonymous Repository for The Web Conference (WWW) Submission.  

> This repository contains the complete implementation and reproduction scripts for DialectCD.



---



## 📌 Overview



*\*DialectCD\*\* is a dialectical causal discovery framework designed for web-scale systems and complex observational datasets. It introduces:

- \*\*Dual-Source Evidence Formulation\*\*: Transforms statistical CI tests and LLM prior beliefs into formal argumentation arguments.

- \*\*Sample-Adaptive Evidentiary Handover\*\*: Smooth Sigmoid continuous transition balancing empirical evidence against prior beliefs.

- \*\*Epistemic Prior Shielding \& Sublinear Aggregation\*\*: Mitigates finite-sample noise and prevents cumulative attack degradation.

- \*\*Dialectical Audit Trace\*\*: Produces human-interpretable justifications and counter-arguments for each retained or pruned causal edge.



---



## ⚙️ Environment Setup



### 1. Python Environment

We recommend using Conda with Python 3.9+:



conda create -n dialectcd python=3.9 -y

conda activate dialectcd

pip install -r requirements.txt



2\. Optional: R Packages (for standard SHD / SID causal metrics)

If you wish to calculate exact structural intervention distance (SID) via R's pcalg library:

code



# In R terminal:

install.packages(c("pcalg", "SID", "bnlearn"))

(Note: If R is not installed, the engine will automatically fallback to high-precision Python implementations of Skeleton-F1, Arrow-F1, and SHD).

🚀 Quick Reproduction Guide

A. Web Platform Benchmark \& Downstream Causal Debiasing (Tables A \& B, Figures 3 \& 4)

Run the web-scale synthetic platform suite (heavy-tail + zero-inflation) with ground-truth causal evaluation:


python dialectcd.py --run\_web\_benchmark --out\_dir ./results

Outputs generated:

results/web\_scm\_benchmark\_summary.csv: Structure learning performance (F1, Arrow-F1, SHD).

results/web\_ate\_debiasing\_summary.csv: Downstream recommendation ATE estimation errors.

results/web\_causal\_graph.png: Layered publication-ready causal DAG visualization.

results/web\_audit\_trace.md: Dialectical dispute audit log for case studies.

