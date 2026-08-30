r"""
Dashboard entry point: two switchable pages, per the project instructions
("Single dashboard (streamlit) with two pages that can be switched between
- academic and portfolio pages").

Run with:
    streamlit run dashboard/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ENGINE_DIR = REPO_ROOT / "engine"
sys.path.insert(0, str(ENGINE_DIR))
sys.path.insert(0, str(REPO_ROOT / "dashboard"))
import _bootstrap  # noqa: E402  (engine's numbered-folder sys.path setup)

import streamlit as st  # noqa: E402

st.set_page_config(page_title="QAOA Portfolio Optimization", layout="wide", page_icon="⚛")

PAGES = {
    "Academic": "academic_page",
    "Portfolio Dashboard": "portfolio_page",
}

st.sidebar.title("QAOA Portfolio Optimization")
st.sidebar.caption("Quantum Approximate Optimization Algorithm for constrained portfolio selection")
choice = st.sidebar.radio("Page", list(PAGES.keys()), label_visibility="collapsed")

st.sidebar.divider()
st.sidebar.caption(
    "Part 1 (academic): cardinality-constrained Markowitz portfolio selection, "
    "reformulated as a QUBO and solved with QAOA, benchmarked against classical "
    "solvers on real crypto and equity data."
)

import importlib
page_module = importlib.import_module(PAGES[choice])
page_module.render()
