r"""
Page 2/2 - Portfolio dashboard (extended project, Part 2).
================================================================
Deliberately left blank per the project instructions ("Blank page - will
be modified in future"). Kept as its own module so it can be built out
later without touching the academic page or app.py's router at all - the
router only calls render(), nothing here is coupled to how the academic
page works internally.
"""

from __future__ import annotations

import streamlit as st


def render():
    st.title("Portfolio Dashboard")
    st.info(
        "This page is intentionally blank for now - it's the reserved slot for "
        "the portfolio-management extension (Part 2 of the project). "
        "The academic page (Part 1) is fully built out; switch to it from the sidebar.",
        icon="🚧",
    )
