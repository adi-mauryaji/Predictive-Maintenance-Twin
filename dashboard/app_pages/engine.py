import pandas as pd
import streamlit as st
from api_client import ApiUnavailable, get_json
from charts import sensor_trend

ALERTS = {"HEALTHY": st.success, "WATCH": st.warning, "CRITICAL": st.error}
ICONS = {
    "HEALTHY": ":material/check_circle:",
    "WATCH": ":material/warning:",
    "CRITICAL": ":material/error:",
}


@st.fragment(run_every="2s")
def engine_view(engine_id: int) -> None:
    try:
        snapshot = get_json(f"/engines/{engine_id}")
        history = get_json(f"/engines/{engine_id}/history")
    except ApiUnavailable:
        st.warning("Start it with `uv run pmt demo`.", title="The twin API is not reachable")
        return

    status = snapshot["status"]
    with st.container(horizontal=True):
        st.metric("RUL (cycles)", f"{snapshot['rul']['value']:.0f}", border=True)
        st.metric("Cycle", snapshot["cycle"], border=True)
        st.metric("Life", snapshot["life"], border=True)
    ALERTS[status](snapshot["recommendation"], title=status.title(), icon=ICONS[status])

    points = pd.DataFrame(
        [{"cycle": point["cycle"], "rul": point["rul"], **point["sensors"]} for point in history]
    )
    st.subheader("Predicted RUL")
    st.line_chart(
        points,
        x="cycle",
        y="rul",
        x_label="Cycle",
        y_label="Cycles left",
        alt="Predicted remaining cycles over the engine's life",
    )
    st.subheader("Two sensors that drift with wear")
    left, right = st.columns(2)
    with left:
        st.altair_chart(sensor_trend(points, "s11"), alt="Sensor s11 over cycles")
    with right:
        st.altair_chart(sensor_trend(points, "s12"), alt="Sensor s12 over cycles")


try:
    fleet = get_json("/fleet")
except ApiUnavailable:
    st.warning("Start it with `uv run pmt demo`.", title="The twin API is not reachable")
    st.stop()

if not fleet:
    st.info("Waiting for the first sensor readings.")
    st.stop()

closest_to_failure = min(range(len(fleet)), key=lambda index: fleet[index]["rul"])
chosen = st.selectbox(
    "Engine", [item["engine_id"] for item in fleet], index=closest_to_failure, key="engine_id"
)
if chosen is not None:
    engine_view(chosen)
