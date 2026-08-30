"""
Path bootstrap for portfolio_engine's own numbered stage folders.
========================================================================
Sister module to engine/_bootstrap.py, same mechanism, deliberately
duplicated rather than shared: portfolio_engine is its OWN pipeline flow
(universe selection -> weighting -> risk metrics -> constraint
experiments) built ON TOP OF the academic engine (imported directly, not
copied - see 01_universe/data_sources.py), not a continuation of its
numbering. Keeping the two bootstraps independent means either folder
tree can be read, tested, or moved on its own.

NAMED DIFFERENTLY from engine/_bootstrap.py on purpose, not just by
sister-folder convention: `import _bootstrap` from two different
directories added to sys.path resolves to whichever one Python imported
FIRST (Python caches modules by name in sys.modules, not by path) - the
second `import _bootstrap` silently returns the FIRST one's already-
executed module instead of re-running this file, so portfolio_engine's
own numbered folders never actually get added to sys.path. Hit this
directly while wiring up the dashboard; renaming this file is the fix.
"""

import sys
from pathlib import Path

PORTFOLIO_ENGINE_DIR = Path(__file__).resolve().parent

for _sub in sorted(PORTFOLIO_ENGINE_DIR.iterdir()):
    if _sub.is_dir() and _sub.name[:2].isdigit():
        p = str(_sub)
        if p not in sys.path:
            sys.path.insert(0, p)
