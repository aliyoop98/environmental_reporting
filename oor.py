"""Utilities for computing out-of-range (OOR) events."""

from typing import Dict, List, Optional

import numpy as np
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

    work = df.loc[:, work_cols].copy()
    work = work.dropna(subset=["DateTime", channel])
    if work.empty:
        return pd.DataFrame(columns=columns)

    work["DateTime"] = pd.to_datetime(work["DateTime"], errors="coerce")
    work = work.dropna(subset=["DateTime"])
    work = work.sort_values("DateTime")
    work = work.drop_duplicates(subset=["DateTime"], keep="last")

    work[channel] = pd.to_numeric(work[channel], errors="coerce")
    work = work.dropna(subset=[channel])
    if work.empty:
        return pd.DataFrame(columns=columns)

    if "Source" not in work.columns:
        work["Source"] = default_source
    else:
        work["Source"] = work["Source"].fillna(default_source)

    values = work[channel].to_numpy(dtype=float, copy=False)
    times = work["DateTime"].to_numpy(dtype="datetime64[ns]", copy=False)
    n = len(values)
    if n == 0:
        return pd.DataFrame(columns=columns)

    oor_mask = np.zeros(n, dtype=bool)
    if lo is not None:
        oor_mask |= values < lo
    if hi is not None:
        oor_mask |= values > hi
    if not np.any(oor_mask):
        return pd.DataFrame(columns=columns)

    # Determine sampling cadence (minutes). Default to 1 minute when unknown.
    diffs = np.diff(times).astype("timedelta64[s]").astype(float) / 60.0
    cadence_min = float(np.median(diffs)) if diffs.size else 1.0
    cadence_min = max(cadence_min, 1.0)

    oor_indices = np.flatnonzero(oor_mask)
    split_points = np.where(np.diff(oor_indices) > 1)[0] + 1
    runs = np.split(oor_indices, split_points)

    events: List[Dict[str, object]] = []
    for run in runs:
        start_i = int(run[0])
        end_i = int(run[-1])

        start_ts = pd.Timestamp(times[start_i])
        next_i = end_i + 1
        if next_i < n and not oor_mask[next_i]:
            end_ts = pd.Timestamp(times[next_i])
        else:
            end_ts = pd.Timestamp(times[end_i]) + pd.to_timedelta(cadence_min, unit="m")

        duration_min = max(
            (end_ts - start_ts).total_seconds() / 60.0,
            float(OOR_MIN_SINGLE_SAMPLE_MINUTES),
        )

        block_sources = work.iloc[run]["Source"]
        mode_source = block_sources.mode(dropna=True)
        if not mode_source.empty:
            source_label = mode_source.iloc[0]
        else:
            non_na_sources = block_sources.dropna()
            source_label = (
                non_na_sources.iloc[0] if not non_na_sources.empty else default_source
            )

        events.append(
            {
                "Start": start_ts,
                "End": end_ts,
                "Duration(min)": round(float(duration_min), 2),
                "Source": source_label,
            }
        )

    return pd.DataFrame(events, columns=columns)
