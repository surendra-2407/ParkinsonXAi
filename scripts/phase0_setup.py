#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
"""
Phase 0: Environment Verification & Requirements
ParkinsonXAI Project
"""
import subprocess
import sys
import importlib

REQUIRED_PACKAGES = [
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("scipy", "scipy"),
    ("matplotlib", "matplotlib"),
    ("seaborn", "seaborn"),
    ("sklearn", "scikit-learn"),
    ("xgboost", "xgboost"),
    ("lightgbm", "lightgbm"),
    ("catboost", "catboost"),
    ("optuna", "optuna"),
    ("imblearn", "imbalanced-learn"),
    ("shap", "shap"),
    ("librosa", "librosa"),
    ("opensmile", "opensmile"),
    ("parselmouth", "praat-parselmouth"),
    ("soundfile", "soundfile"),
    ("pydub", "pydub"),
    ("resampy", "resampy"),
    ("statsmodels", "statsmodels"),
    ("joblib", "joblib"),
]

def check_packages():
    print("="*60)
    print("ParkinsonXAI — Phase 0: Environment Verification")
    print("="*60)
    missing = []
    installed = []
    for module_name, pkg_name in REQUIRED_PACKAGES:
        try:
            mod = importlib.import_module(module_name)
            version = getattr(mod, "__version__", "unknown")
            installed.append((pkg_name, version))
            print(f"  [OK] {pkg_name:<30} {version}")
        except ImportError:
            missing.append(pkg_name)
            print(f"  [!!] {pkg_name:<30} NOT INSTALLED")

    # Deep learning (optional)
    for mod_name, pkg_name in [("torch", "torch"), ("tensorflow", "tensorflow")]:
        try:
            mod = importlib.import_module(mod_name)
            version = getattr(mod, "__version__", "unknown")
            installed.append((pkg_name, version))
            print(f"  [OK] {pkg_name:<30} {version} (optional)")
        except ImportError:
            print(f"  [ ] {pkg_name:<30} not installed (optional)")

    print()
    if missing:
        print(f"MISSING PACKAGES: {missing}")
        print("Run: pip install " + " ".join(missing))
    else:
        print("All required packages are installed.")

    # Write requirements.txt
    result = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        capture_output=True, text=True
    )
    with open("requirements.txt", "w") as f:
        f.write(result.stdout)
    print("\nrequirements.txt written.")
    return len(missing) == 0

if __name__ == "__main__":
    ok = check_packages()
    sys.exit(0 if ok else 1)
