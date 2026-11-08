"""Minimal Streamlit dashboard over the SentinelML API (prototype UI; the React frontend is NOT implemented).

    pip install streamlit requests
    API_URL=http://localhost:8000 streamlit run dashboard/app.py
NOTE: this file has been syntax-checked only; it was not run against a browser in the build environment.
"""
import os

import pandas as pd
import requests
import streamlit as st

API = os.getenv("API_URL", "http://localhost:8000") + "/api/v1"
st.set_page_config(page_title="SentinelML", layout="wide")
st.title("SentinelML - model reliability monitor")

with st.sidebar:
    email = st.text_input("Email"); pw = st.text_input("Password", type="password")
    if st.button("Login"):
        r = requests.post(f"{API}/auth/login", json={"email": email, "password": pw}, timeout=10)
        if r.ok:
            st.session_state["h"] = {"Authorization": f"Bearer {r.json()['access_token']}"}
        else:
            st.error(r.json().get("detail", "login failed"))

h = st.session_state.get("h")
if not h:
    st.info("Log in to continue."); st.stop()


def get(path, **params):
    r = requests.get(f"{API}{path}", headers=h, params=params, timeout=30)
    return r.json() if r.ok else None


def post(path, body):
    r = requests.post(f"{API}{path}", headers=h, json=body, timeout=300)
    return r.status_code, r.json()


ov = get("/monitoring/overview") or {}
c = st.columns(5)
for col, k in zip(c, ["total_models", "healthy", "at_risk", "degraded", "open_alerts"]):
    col.metric(k.replace("_", " "), ov.get(k, 0))
models = get("/models") or []
if not models:
    st.warning("No models yet - upload a dataset and train via the API."); st.stop()
m = st.selectbox("Model", models, format_func=lambda x: f"#{x['id']} {x['name']}")
tab1, tab2, tab3, tab4 = st.tabs(["Monitoring", "Failure prediction", "Diagnosis", "Recovery"])
with tab1:
    d = get("/monitoring/drift", model_id=m["id"])
    if d:
        hist = pd.DataFrame(d["history"]).set_index("window_index")
        st.line_chart(hist[["psi_max", "ks_max", "missing_max"]])
        st.dataframe(pd.DataFrame(d["latest_feature_drift"]).T)
    else:
        st.info("No snapshots yet.")
    st.dataframe(pd.DataFrame(get("/monitoring/alerts", model_id=m["id"]) or []))
with tab2:
    if st.button("Predict failure"):
        code, body = post("/failure-predictions", {"model_id": m["id"]}); st.json(body)
with tab3:
    if st.button("Run diagnosis"):
        code, body = post("/diagnosis", {"model_id": m["id"]}); st.json(body)
with tab4:
    if st.button("Simulate recovery"):
        code, body = post("/recovery/simulate", {"model_id": m["id"]})
        st.json(body)
    sid = st.number_input("Simulation id", min_value=1, step=1)
    if st.button("Validate"):
        st.json(post("/recovery/validate", {"simulation_id": int(sid)})[1])
    if st.button("Approve & deploy (human decision)"):
        st.json(post("/recovery/approve", {"simulation_id": int(sid)})[1])
    if st.button("Roll back"):
        st.json(post("/recovery/rollback", {"simulation_id": int(sid)})[1])
