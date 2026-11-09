from collections import Counter

import pandas as pd
import streamlit as st
from api_client import ApiUnavailable, get_json
from charts import fleet_grid


@st.fragment(run_every="2s")
def fleet_view() -> None:
    try:
        items = get_json("/fleet")
    except ApiUnavailable:
        st.warning("Start it with `uv run pmt demo`.", title="The twin API is not reachable")
        return
    if not items:
        st.info("Waiting for the first sensor readings.")
        return

    fleet = pd.DataFrame(items)
    counts = Counter(item["status"] for item in items)
    with st.container(horizontal=True):
        st.metric(":material/check_circle: Healthy", counts["HEALTHY"], border=True)
        st.metric(":material/warning: Watch", counts["WATCH"], border=True)
        st.metric(":material/error: Critical", counts["CRITICAL"], border=True)
        st.metric(
            ":material/notifications_active: Alarms",
            sum(item["alarm"] for item in items),
            border=True,
        )

    st.altair_chart(
        fleet_grid(fleet), alt=f"Status of {len(fleet)} engines, one cell per engine", key="grid"
    )
    st.dataframe(
        fleet.sort_values("rul")[["engine_id", "status", "alarm", "rul", "cycle"]],
        column_config={
            "engine_id": st.column_config.NumberColumn("Engine"),
            "status": st.column_config.TextColumn("Status"),
            "alarm": st.column_config.CheckboxColumn("Alarm"),
            "rul": st.column_config.NumberColumn("RUL (cycles)", format="%.0f"),
            "cycle": st.column_config.NumberColumn("Cycle"),
        },
        hide_index=True,
        height=280,
        alt="Engines sorted by remaining cycles, lowest first",
        key="table",
    )


st.caption("Each cell is one engine. RUL is the model's estimate of cycles left.")
fleet_view()
