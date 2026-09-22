#!/usr/bin/env python3
"""
ParkinsonXAI — Master Pipeline Runner
Runs all phases in order and logs results.
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"

PHASES = [
    ("Phase 0: Environment Setup",     "phase0_setup.py"),
    ("Phase 1+2: Dataset Audit",       "phase1_dataset_audit.py"),
    ("Phase 3+4: Leakage-Safe Splits", "phase3_splitting.py"),
    ("Phase 5: Preprocessing",         "phase5_preprocessing.py"),
    ("Phase 6: Audio Features",        "phase6_audio_features.py"),
    ("Phase 7-9: Models",              "phase7_8_9_models.py"),
    ("Phase 10-13: Optuna+Ensemble",   "phase10_13_optuna_ensemble.py"),
    ("Phase 15: Severity Regression",  "phase15_severity_model.py"),
    ("Phase 17: SHAP/XAI",             "phase17_shap_analysis.py"),
    ("Phase 16+18+19+20: Finalize",    "phase16_18_19_20_finalize.py"),
]

print("="*70)
print("ParkinsonXAI — Full Pipeline Execution")
print("="*70)

for phase_name, script_name in PHASES:
    script_path = SCRIPTS / script_name
    print(f"\n{'='*70}")
    print(f"  STARTING: {phase_name}")
    print(f"  Script: {script_name}")
    print(f"{'='*70}")
    start = time.time()

    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(ROOT),
    )
    elapsed = time.time() - start

    if result.returncode == 0:
        print(f"\n  ✓ {phase_name} COMPLETE in {elapsed:.1f}s")
    else:
        print(f"\n  ✗ {phase_name} FAILED (exit code {result.returncode}) after {elapsed:.1f}s")
        print(f"  Continuing to next phase...")

print("\n" + "="*70)
print("PIPELINE EXECUTION COMPLETE")
print("="*70)
