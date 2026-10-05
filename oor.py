from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

MIN_SINGLE_SAMPLE_MINUTES = 1.0


def compute_oor_events(
    df: pd.DataFrame,
    channel: str,
    low: Optional[float],
    high: Optional[float],
    default_source: str = "Primary",
) -> pd.DataFrame:
    """Return contiguous out-of-range runs for one measurement channel.

    Threshold values themselves are in range.  A run ends at the first following
    in-range sample.  A final/single-sample run uses the median sampling cadence,
    with a one-minute minimum.
    """
    columns = ["Start", "End", "Duration(min)", "Source"]
    if df is None or df.empty or channel not in df.columns or "DateTime" not in df.columns:
        return pd.DataFrame(columns=columns)

    keep = ["DateTime", channel] + (["Source"] if "Source" in df.columns else [])
    work = df[keep].copy()
    work["DateTime"] = pd.to_datetime(work["DateTime"], errors="coerce")
    work[channel] = pd.to_numeric(work[channel], errors="coerce")
    work = work.dropna(subset=["DateTime", channel]).sort_values("DateTime")
    if work.empty:
        return pd.DataFrame(columns=columns)

    if "Source" not in work.columns:
        work["Source"] = default_source
    else:
        work["Source"] = work["Source"].fillna(default_source)

    # Keep one value per timestamp for this channel.
    work = work.drop_duplicates(subset=["DateTime"], keep="last").reset_index(drop=True)
    values = work[channel].to_numpy(dtype=float)
    times = work["DateTime"].to_numpy(dtype="datetime64[ns]")

    mask = np.zeros(len(work), dtype=bool)
    if low is not None:
        mask |= values < low
    if high is not None:
        mask |= values > high
    if not mask.any():
        return pd.DataFrame(columns=columns)

    diffs = np.diff(times).astype("timedelta64[s]").astype(float) / 60.0
    positive = diffs[diffs > 0]
    cadence = float(np.median(positive)) if positive.size else MIN_SINGLE_SAMPLE_MINUTES
    cadence = max(cadence, MIN_SINGLE_SAMPLE_MINUTES)

    indices = np.flatnonzero(mask)
    runs = np.split(indices, np.where(np.diff(indices) > 1)[0] + 1)
    events = []
    for run in runs:
        start_i, end_i = int(run[0]), int(run[-1])
        start = pd.Timestamp(times[start_i])
        if end_i + 1 < len(work) and not mask[end_i + 1]:
            end = pd.Timestamp(times[end_i + 1])
        else:
            end = pd.Timestamp(times[end_i]) + pd.to_timedelta(cadence, unit="m")
        duration = max((end - start).total_seconds() / 60.0, MIN_SINGLE_SAMPLE_MINUTES)
        sources = work.iloc[run]["Source"].dropna()
        source = sources.mode().iloc[0] if not sources.empty else default_source
        events.append(
            {
                "Start": start,
                "End": end,
                "Duration(min)": round(float(duration), 2),
                "Source": source,
            }
        )

    return pd.DataFrame(events, columns=columns)


# Backward-compatible name for older imports.
_compute_oor_events = compute_oor_events

__all__ = ["compute_oor_events", "_compute_oor_events"]
