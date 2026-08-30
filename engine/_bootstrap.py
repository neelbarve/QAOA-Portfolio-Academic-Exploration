"""
Path bootstrap for the numbered pipeline-stage folders.
=========================================================
Folders under engine/ are prefixed 01_.. through 10_.. so the directory
listing itself shows the project's data flow (data -> preprocessing -> EDA ->
classical formulation -> QUBO -> quantum solvers -> classical benchmark ->
benchmarking/scaling -> statistics -> Fama-French Hamiltonian derivation).

A directory name starting with a digit is not a legal Python package name
for a plain `import numbered.pkg` statement, so these folders are NOT
packages (no __init__.py inside them). Instead, this module adds each one
to sys.path once, and every stage file inside is a flat module imported by
its own plain name, e.g. `from qubo_builder import build_qubo`, exactly the
same style as the original portfolio_qaoa.py / data_adapters.py scripts.

Any entry point (pipeline.py, the Streamlit dashboard, tests, scripts) just
needs `import engine` once before importing stage modules by name.
"""

import sys
from pathlib import Path

ENGINE_DIR = Path(__file__).resolve().parent

for _sub in sorted(ENGINE_DIR.iterdir()):
    if _sub.is_dir() and _sub.name[:2].isdigit():
        p = str(_sub)
        if p not in sys.path:
            sys.path.insert(0, p)
