# Contributing
1. `pip install -e ".[api,tracking,dev]"` 2. `pytest` must pass 3. keep type hints + docstrings 4. one logical change per commit
using conventional messages (`feat:`, `fix:`, `test:`, `docs:`, `exp:` for experiment changes). Any change to
`simulate.py`, `failures.py` or `monitor.py` invalidates `results/cache`; regenerate with `experiments/run_benchmark.py`
and re-run `experiments/analyze.py` so that RESULTS.md always matches the code.
