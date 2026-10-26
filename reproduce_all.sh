#!/usr/bin/env bash
# ==============================================================================
# One-Click Reproduction Script for DialectCD (Anonymous WWW Evaluation)
# ==============================================================================

set -e

echo "=========================================================="
echo " Starting Full DialectCD Reproduction Pipeline"
echo "=========================================================="

# 1. Create output directory
mkdir -p results

# 2. Run Web-Native Benchmark & Causal Debiasing (Tables A, B & Figures)
echo "[1/3] Running Web SCM Benchmark & Downstream Debiasing..."
python dialectcd.py --run_web_benchmark --out_dir ./results

# 3. Run Ablation Studies & Sensitivity Sweeps (Section 4.3)
echo "[2/3] Running Component Ablation Studies..."
python dialectcd.py --run_ablation_sensitivity --out_dir ./results

# 4. Run Multi-Dataset Benchmark (Table 1)
echo "[3/3] Running Standard Multi-Dataset Benchmark (Child & Insurance)..."
python dialectcd.py --benchmark \
    --benchmark_datasets child insurance \
    --samples 2000 \
    --repeats 20 \
    --out_dir ./results

echo "=========================================================="
echo " [✓] All Experiments Completed Successfully!"
echo " Check the './results' directory for generated figures & tables."
echo "=========================================================="