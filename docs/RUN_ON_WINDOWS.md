# Running SentinelML from Windows Command Prompt (cmd)

Tested on Linux only; commands below are the Windows equivalents. Needs Python 3.10+ (https://www.python.org/downloads/ - tick "Add python.exe to PATH").

1. `cd` to the unzipped folder:  `cd C:\Users\<you>\Downloads\sentinelml`
2. Create + activate a virtual environment:  `python -m venv .venv`  then  `.venv\Scripts\activate.bat`
3. Install:  `python -m pip install --upgrade pip`  then  `pip install -e ".[api,tracking,dev]"`
4. Run tests (~1 min):  `python -m pytest`
5. Start the API (terminal 1):  `set SECRET_KEY=any-long-random-string-at-least-32-chars`  then
   `set FAILURE_PREDICTOR_PATH=artifacts\failure_predictor.joblib`  `set FAILURE_TYPE_PATH=artifacts\failure_type_clf.joblib`  then
   `uvicorn sentinelml.api.main:app --port 8000`   (docs UI: http://localhost:8000/docs)
6. Run the guided demo (terminal 2, same folder, venv activated):  `python scripts\demo_api_flow.py`
7. Reproduce the experiments:  `python experiments\analyze.py` (uses the bundled cache, a few minutes) -> `results\RESULTS.md`
   Regenerate the benchmark from scratch (slow, ~20-40 min): `del /q results\cache\*` then `python experiments\run_benchmark.py`
8. Optional: `python scripts\train_failure_predictor.py`, `python scripts\train_baselines.py --dataset breast_cancer`
9. Optional dashboard: `pip install streamlit requests` then `streamlit run dashboard\app.py`
Troubleshooting: `'python' is not recognized` -> use `py` or reinstall with PATH ticked; `No module named sentinelml` -> run step 3 inside the activated venv;
port busy -> `--port 8001` (and pass the URL to the demo script); `set` variables last only for that cmd window.

## Error: `ValueError: On field "rows" the following field constraints are set but not enforced` / pydantic `ModelField`
You have Pydantic 1.x installed; SentinelML needs Pydantic 2.x. Fix: `pip install -U "pydantic>=2" "fastapi>=0.100" "sqlalchemy>=2"` then `pip show pydantic` (Version must start with 2).
Tip: avoid putting the project or `.venv` inside OneDrive (sync can lock files and slow installs); `C:\dev\sentinelml` is safer.
