"""Utilities for computing out-of-range (OOR) events."""

from typing import Dict, List, Optional

import pandas as pd


# Minimum duration in minutes to assign to a single-sample OOR event.
OOR_MIN_SINGLE_SAMPLE_MINUTES = 1.0


def _is_oor_strict(value: float, lo: Optional[float], hi: Optional[float]) -> bool:
    """Return True when the value is strictly outside the provided bounds."""

    if pd.isna(value):
        return False
    if lo is not None and value < lo:
        return True
    if hi is not None and value > hi:
        return True
    return False


def _compute_oor_events(
    df: pd.DataFrame,
    channel: str,
    lo: Optional[float],
    hi: Optional[float],
    default_source: str,
) -> pd.DataFrame:
    """Return a DataFrame describing out-of-range runs for a channel."""

    columns = ["Start", "End", "Duration(min)", "Source"]
    if df.empty or channel not in df.columns:
        return pd.DataFrame(columns=columns)

    work_cols = ["DateTime", channel]
    if "Source" in df.columns:
        work_cols.append("Source")

    work = df.loc[:, work_cols].dropna(subset=["DateTime", channel]).copy()
    if work.empty:
        return pd.DataFrame(columns=columns)

    work["DateTime"] = pd.to_datetime(work["DateTime"], errors="coerce")
    work = work.dropna(subset=["DateTime"])
    work = work.sort_values("DateTime")
    work = work.drop_duplicates(subset=["DateTime"], keep="first")

    if "Source" not in work.columns:
        work["Source"] = default_source
    else:
        work["Source"] = work["Source"].fillna(default_source)

    oor_mask = work[channel].apply(lambda value: _is_oor_strict(value, lo, hi))
    if not oor_mask.any():
        return pd.DataFrame(columns=columns)

    groups = (oor_mask != oor_mask.shift(fill_value=False)).cumsum()
    events: List[Dict[str, object]] = []

    for _, block in work[oor_mask].groupby(groups[oor_mask]):
        times = block["DateTime"].reset_index(drop=True)
        start = times.iloc[0]
        end = times.iloc[-1]
        if len(times) >= 2:
            gaps = times.diff().dropna()
            duration_min = gaps.dt.total_seconds().sum() / 60.0
            duration_min = max(duration_min, 0.0)
        else:
            duration_min = float(OOR_MIN_SINGLE_SAMPLE_MINUTES)

        mode_source = block["Source"].mode(dropna=True)
        if not mode_source.empty:
            source_label = mode_source.iloc[0]
        else:
            non_na_sources = block["Source"].dropna()
            source_label = (
                non_na_sources.iloc[0] if not non_na_sources.empty else default_source
            )

        events.append(
            {
                "Start": start,
                "End": end,
                "Duration(min)": float(duration_min),
                "Source": source_label,
            }
        )

    return pd.DataFrame(events, columns=columns)
