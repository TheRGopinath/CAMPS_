#!/bin/bash
# Optional: submit the full scan to a SLURM cluster with  sbatch run_scan.slurm
#SBATCH --job-name=mipt_scan
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=mipt_scan_%j.log

cd "$SLURM_SUBMIT_DIR"            # submit from the MIPT_CODE folder
python -m examples.mipt_scan --workers "$SLURM_CPUS_PER_TASK"
python analysis/collapse_signed.py examples/output/mipt_scan/entanglement_vs_p.csv 0.16 1.3 examples/output/mipt_scan/collapse_signed.png
python analysis/s_of_p.py         examples/output/mipt_scan/entanglement_vs_p.csv examples/output/mipt_scan/s_of_p.png
python analysis/loglog_S_vs_N.py  examples/output/mipt_scan/entanglement_vs_p.csv examples/output/mipt_scan/loglog_S_vs_N.png
