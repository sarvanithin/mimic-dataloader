"""
MIMIC DataLoader — Health Universe deployment
Calls the live Render API.
"""

import requests
import streamlit as st

API = "https://mimic-dataloader.onrender.com"
HEADERS = {"Content-Type": "application/json"}

TASKS = ["mortality", "readmission", "los", "sepsis", "phenotyping", "decompensation"]

TASK_DESCRIPTIONS = {
    "mortality": "In-hospital mortality prediction (48-hour window)",
    "readmission": "30-day readmission prediction",
    "los": "Length of stay prediction (regression, days)",
    "sepsis": "Sepsis onset prediction (6-hour early warning)",
    "phenotyping": "25-class ICD phenotype classification",
    "decompensation": "Physiologic decompensation prediction (hourly)",
}

st.set_page_config(
    page_title="MIMIC DataLoader",
    page_icon="🏥",
    layout="wide",
)

st.title("🏥 MIMIC DataLoader")
st.caption("MIMIC-IV clinical dataset explorer — schemas, statistics & sample previews for 6 prediction tasks")

try:
    h = requests.get(f"{API}/health", timeout=8).json()
    st.success(f"Service online · **{h.get('service', 'mimic-dataloader-api')}** v{h.get('version', '—')}")
except Exception:
    st.warning("⚠️ Service may be waking up (free tier). Refresh in 30s.")

st.divider()

tab1, tab2, tab3, tab4 = st.tabs(["📋 Browse Tasks", "🔬 Schema", "📊 Statistics", "👁️ Preview"])

# ── Tab 1: Browse Tasks ────────────────────────────────────────────────────────
with tab1:
    st.subheader("Available MIMIC-IV Prediction Tasks")
    st.caption("Full dataset access requires credentialed PhysioNet MIMIC-IV access.")

    if st.button("🔄 Load Tasks", type="primary"):
        with st.spinner("Loading…"):
            try:
                resp = requests.get(f"{API}/v1/tasks", timeout=10)
                if resp.status_code == 200:
                    data = resp.json()
                    tasks = data.get("tasks", TASKS)
                    st.info(data.get("note", ""))
                    cols = st.columns(3)
                    for i, task in enumerate(tasks):
                        with cols[i % 3]:
                            st.metric(task.upper(), TASK_DESCRIPTIONS.get(task, "—"))
                else:
                    st.error(f"Error {resp.status_code}: {resp.text[:300]}")
            except Exception as ex:
                st.error(f"Error: {ex}")
    else:
        cols = st.columns(3)
        for i, (task, desc) in enumerate(TASK_DESCRIPTIONS.items()):
            with cols[i % 3]:
                with st.container(border=True):
                    st.markdown(f"**{task.upper()}**")
                    st.caption(desc)

# ── Tab 2: Schema ──────────────────────────────────────────────────────────────
with tab2:
    st.subheader("Dataset Schema")
    st.caption("Column schema and metadata for each prediction task. ($0.01 per request)")

    col1, col2 = st.columns([2, 1])
    with col1:
        task_schema = st.selectbox("Task", TASKS, key="schema_task")
    with col2:
        split_schema = st.selectbox("Split", ["train", "val", "test"], key="schema_split")

    if st.button("Get Schema", type="primary", key="btn_schema"):
        with st.spinner("Fetching schema…"):
            try:
                resp = requests.post(
                    f"{API}/v1/schema",
                    json={"task": task_schema, "split": split_schema},
                    headers=HEADERS,
                    timeout=15,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    schema = data.get("schema", {})
                    st.success(f"Schema for **{task_schema}** ({split_schema})")

                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.markdown(f"**Description:** {schema.get('description', '—')}")
                        st.markdown(f"**Label:** `{schema.get('label', '—')}`")
                        if schema.get("seq_length"):
                            st.markdown(f"**Sequence length:** {schema['seq_length']} hours")
                    with col_b:
                        st.markdown(f"**Train rows:** {schema.get('rows_train', '—'):,}")
                        st.markdown(f"**Val rows:** {schema.get('rows_val', '—'):,}")
                        st.markdown(f"**Test rows:** {schema.get('rows_test', '—'):,}")

                    features = schema.get("features", [])
                    if features:
                        st.markdown("**Features:**")
                        for f in features:
                            st.markdown(f"- `{f}`")

                elif resp.status_code == 402:
                    st.error("402 Payment Required — API token needed for programmatic access.")
                else:
                    st.error(f"Error {resp.status_code}: {resp.text[:300]}")
            except Exception as ex:
                st.error(f"Error: {ex}")

# ── Tab 3: Statistics ──────────────────────────────────────────────────────────
with tab3:
    st.subheader("Dataset Statistics")
    st.caption("Class balance and feature statistics for each task. ($0.01 per request)")

    col1, col2 = st.columns([2, 1])
    with col1:
        task_stats = st.selectbox("Task", TASKS, key="stats_task")
    with col2:
        split_stats = st.selectbox("Split", ["train", "val", "test"], key="stats_split")

    if st.button("Get Statistics", type="primary", key="btn_stats"):
        with st.spinner("Fetching statistics…"):
            try:
                resp = requests.post(
                    f"{API}/v1/stats",
                    json={"task": task_stats, "split": split_stats},
                    headers=HEADERS,
                    timeout=15,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    st.success(f"Statistics for **{task_stats}** ({split_stats})")

                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Total Rows", f"{data.get('total_rows', 0):,}" if isinstance(data.get('total_rows'), int) else data.get('total_rows', '—'))
                    c2.metric("Features", data.get("feature_count", "—"))
                    c3.metric("Label Type", data.get("label_type", "—").split(" ")[0])
                    seq = data.get("seq_length")
                    c4.metric("Seq Length", f"{seq}h" if seq else "Variable")

                    st.markdown(f"**Label:** `{data.get('label_type', '—')}`")

                elif resp.status_code == 402:
                    st.error("402 Payment Required — API token needed for programmatic access.")
                else:
                    st.error(f"Error {resp.status_code}: {resp.text[:300]}")
            except Exception as ex:
                st.error(f"Error: {ex}")

# ── Tab 4: Preview ─────────────────────────────────────────────────────────────
with tab4:
    st.subheader("Sample Record Preview")
    st.caption(
        "Preview 5 anonymized schema-level records for any task. "
        "**Note:** Real patient data requires PhysioNet MIMIC-IV credentials. ($0.02 per request)"
    )

    col1, col2 = st.columns([2, 1])
    with col1:
        task_preview = st.selectbox("Task", TASKS, key="preview_task")
    with col2:
        split_preview = st.selectbox("Split", ["train", "val", "test"], key="preview_split")

    if st.button("Get Preview", type="primary", key="btn_preview"):
        with st.spinner("Fetching preview…"):
            try:
                resp = requests.post(
                    f"{API}/v1/preview",
                    json={"task": task_preview, "split": split_preview},
                    headers=HEADERS,
                    timeout=15,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    st.info(data.get("note", ""))
                    samples = data.get("samples", [])
                    if samples:
                        import pandas as pd
                        st.dataframe(pd.DataFrame(samples), use_container_width=True)
                elif resp.status_code == 402:
                    st.error("402 Payment Required — API token needed for programmatic access.")
                else:
                    st.error(f"Error {resp.status_code}: {resp.text[:300]}")
            except Exception as ex:
                st.error(f"Error: {ex}")

st.divider()
st.caption("Powered by [MIMIC DataLoader](https://mimic-dataloader.onrender.com/docs) · x402/MPP payment protocol · Requires PhysioNet MIMIC-IV credentials for full access")
