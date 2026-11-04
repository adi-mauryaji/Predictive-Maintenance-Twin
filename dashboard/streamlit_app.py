import streamlit as st

st.set_page_config(page_title="Predictive Maintenance Twin", layout="wide")

page = st.navigation(
    [
        st.Page("app_pages/fleet.py", title="Fleet", icon=":material/grid_view:"),
        st.Page("app_pages/engine.py", title="Engine", icon=":material/monitoring:"),
    ],
    position="top",
)
st.title(page.title, icon=page.icon)
page.run()
