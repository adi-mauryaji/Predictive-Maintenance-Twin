"""Fleet chart. Status always shows as colour plus a mark in the cell, never colour alone."""

import altair as alt
import pandas as pd

ENGINES_PER_ROW = 10
STATUS_ORDER = ["HEALTHY", "WATCH", "CRITICAL"]
STATUS_COLORS = ["#0ca30c", "#fab219", "#d03b3b"]
STATUS_MARKS = {"HEALTHY": "", "WATCH": " ▲", "CRITICAL": " !"}
DARK_INK_ON = "WATCH"  # white text is unreadable on the warning colour


def fleet_grid(fleet: pd.DataFrame) -> alt.LayerChart | alt.FacetChart:
    cells = fleet.assign(
        column=(fleet["engine_id"] - 1) % ENGINES_PER_ROW,
        row=(fleet["engine_id"] - 1) // ENGINES_PER_ROW,
        label=fleet["engine_id"].astype(str)
        + fleet["status"].map(lambda status: STATUS_MARKS[status]),
        rul_rounded=fleet["rul"].round(0),
    )
    rows = max(cells["row"].tolist()) + 1
    base = alt.Chart(cells).encode(
        x=alt.X("column:O", axis=None, scale=alt.Scale(paddingInner=0.06)),
        y=alt.Y("row:O", axis=None, scale=alt.Scale(paddingInner=0.06)),
        tooltip=[
            alt.Tooltip("engine_id:Q", title="Engine"),
            alt.Tooltip("status:N", title="Status"),
            alt.Tooltip("rul_rounded:Q", title="RUL (cycles)"),
            alt.Tooltip("cycle:Q", title="Cycle"),
        ],
    )
    fill = base.mark_rect(cornerRadius=4).encode(
        color=alt.Color(
            "status:N",
            scale=alt.Scale(domain=STATUS_ORDER, range=STATUS_COLORS),
            legend=alt.Legend(title=None, orient="top"),
        )
    )
    text = base.mark_text(fontSize=12).encode(
        text="label:N",
        color=alt.condition(
            alt.datum.status == DARK_INK_ON, alt.value("#0b0b0b"), alt.value("#ffffff")
        ),
    )
    return alt.layer(fill, text, height=rows * 42)


def sensor_trend(points: pd.DataFrame, sensor: str) -> alt.Chart:
    """A sensor drifts by about 1% of its value, so the axis follows the data, not zero."""
    return (
        alt.Chart(points)
        .mark_line(strokeWidth=2)
        .encode(
            x=alt.X("cycle:Q", title="Cycle"),
            y=alt.Y(f"{sensor}:Q", title=sensor, scale=alt.Scale(zero=False)),
            tooltip=["cycle:Q", f"{sensor}:Q"],
        )
    )
