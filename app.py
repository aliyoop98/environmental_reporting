from typing import Optional

import streamlit as st
import pandas as pd
import io
import calendar
import altair as alt
from datetime import datetime, timedelta


def _read_csv_flexible(text: str) -> Optional[pd.DataFrame]:
    """Read CSV content supporting multiple delimiters.

    Tries automatic delimiter detection first (e.g., tab separated files) and
    falls back to the default comma behaviour if necessary.
    """

    for kwargs in ({"sep": None, "engine": "python"}, {}):
        try:
            df = pd.read_csv(io.StringIO(text), on_bad_lines="skip", **kwargs)
        except Exception:
            continue
        return df.rename(columns=lambda c: c.strip() if isinstance(c, str) else c)
    return None

st.set_page_config(page_title="Environmental Reporting", layout="wide", page_icon="⛅")
st.markdown(
    """
    <style>
        .stApp { background-color: white; }
        [data-testid="stSidebar"] { background-color: white; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.sidebar.header("🌦️ Data Upload & Configuration")
probe_files = st.sidebar.file_uploader(
    "Upload Probe CSV files",
    accept_multiple_files=True,
    key="probe_uploader"
)
if not probe_files:
    st.sidebar.info("Upload Probe CSV files to begin.")
    st.stop()

tempstick_files = st.sidebar.file_uploader(
    "Upload Tempstick CSV files (optional)",
    accept_multiple_files=True,
    key="tempstick_uploads"
)

# Parse Tempstick CSVs
tempdfs = {}
if tempstick_files:
    for f in tempstick_files:
        raw = f.read().decode('utf-8', errors='ignore').splitlines(True)
        try:
            idx = max(i for i, line in enumerate(raw) if 'timestamp' in line.lower())
        except ValueError:
            continue
        csv_text = ''.join(raw[idx:])
        df_ts = _read_csv_flexible(csv_text)
        if df_ts is None:
            continue
        ts_col = next((c for c in df_ts.columns if 'timestamp' in c.lower()), None)
        if not ts_col:
            continue
        df_ts = df_ts.rename(columns={ts_col: 'DateTime'})
        df_ts['DateTime'] = pd.to_datetime(df_ts['DateTime'], infer_datetime_format=True, errors='coerce')
        temp_col = next((c for c in df_ts.columns if 'temp' in c.lower()), None)
        if temp_col:
            df_ts['Temperature'] = pd.to_numeric(df_ts[temp_col].astype(str).str.replace('+',''), errors='coerce')
        hum_col = next((c for c in df_ts.columns if 'hum' in c.lower()), None)
        if hum_col:
            df_ts['Humidity'] = pd.to_numeric(df_ts[hum_col].astype(str).str.replace('+',''), errors='coerce')
        cols = ['DateTime']
        if 'Temperature' in df_ts.columns:
            cols.append('Temperature')
        if 'Humidity' in df_ts.columns:
            cols.append('Humidity')
        tempdfs[f.name] = df_ts[cols]


def _match_tempstick_channel(ts_df: pd.DataFrame, channel: str) -> Optional[str]:
    """Return the column in a tempstick dataframe that matches a probe channel."""

    if channel in ts_df.columns:
        return channel
    ch_lower = channel.lower()
    if 'temp' in ch_lower and 'Temperature' in ts_df.columns:
        return 'Temperature'
    if 'hum' in ch_lower and 'Humidity' in ts_df.columns:
        return 'Humidity'
    return None

# Parse Probe CSVs
dfs = {}
ranges = {}
for f in probe_files:
    name = f.name
    raw = f.read().decode('utf-8', errors='ignore').splitlines(True)
    try:
        idx = max(i for i, line in enumerate(raw) if 'ch1' in line.lower() or 'p1' in line.lower())
    except ValueError:
        continue
    content = ''.join(raw[idx:])
    df = _read_csv_flexible(content)
    if df is None:
        continue
    lname = name.lower()
    has_fridge = any(term in lname for term in ('fridge', 'refrigerator'))
    has_freezer = 'freezer' in lname
    if has_fridge and has_freezer:
        mapping = {'Fridge Temp': 'P1', 'Freezer Temp': 'P2'}
        ranges[name] = {'Fridge Temp': (2, 8), 'Freezer Temp': (-35, -5)}
    elif has_fridge:
        mapping = {'Temperature': 'P1'}
        ranges[name] = {'Temperature': (2, 8)}
    elif has_freezer:
        mapping = {'Temperature': 'P1'}
        ranges[name] = {'Temperature': (-35, -5)}
    else:
        cols_lower = [c.lower() for c in df.columns]
        if any('ch4' in c for c in cols_lower):
            mapping = {'Humidity': 'CH3', 'Temperature': 'CH4'}
        else:
            mapping = {'Humidity': 'CH1', 'Temperature': 'CH2'}
        temp_range = (15, 28) if 'olympus' in lname else (15, 25)
        ranges[name] = {'Temperature': temp_range, 'Humidity': (0, 60)}
    mapping.update({'Date': 'Date', 'Time': 'Time'})
    col_map = {}
    for new, key in mapping.items():
        col = next((c for c in df.columns if key.lower() in c.lower()), None)
        if col:
            col_map[col] = new
    df = df[list(col_map)].rename(columns=col_map)
    df['Date'] = pd.to_datetime(df['Date'], infer_datetime_format=True, errors='coerce')
    df['DateTime'] = pd.to_datetime(
        df['Date'].dt.strftime('%Y-%m-%d ') + df['Time'].astype(str),
        errors='coerce'
    )
    for c in df.columns.difference(['Date', 'Time', 'DateTime']):
        df[c] = pd.to_numeric(df[c].astype(str).str.replace('+',''), errors='coerce')
    dfs[name] = df.reset_index(drop=True)

# Sidebar legend label configuration
probe_labels = {}
if dfs:
    st.sidebar.subheader("Legend Labels")
    st.sidebar.caption("Probe files")
    for name in dfs:
        probe_labels[name] = st.sidebar.text_input(
            f"Label for {name}", value=name, key=f"label_probe_{name}"
        )

tempstick_labels = {}
if tempdfs:
    if not dfs:
        st.sidebar.subheader("Legend Labels")
    st.sidebar.caption("Tempstick files")
    for name in tempdfs:
        tempstick_labels[name] = st.sidebar.text_input(
            f"Label for {name}", value=name, key=f"label_tempstick_{name}"
        )

# Year & Month selection
years = sorted({dt.year for df in dfs.values() for dt in df['Date'].dropna()})
months = sorted({dt.month for df in dfs.values() for dt in df['Date'].dropna()})
year = st.sidebar.selectbox("Year", years)
month = st.sidebar.selectbox("Month", months, format_func=lambda m: calendar.month_name[m])

for name, df in dfs.items():
    st.header(name)
    selected_tempsticks = (
        st.multiselect(
            f"Match Tempsticks for {name}",
            options=list(tempdfs.keys()),
            format_func=lambda opt: tempstick_labels.get(opt, opt),
            key=f"ts_{name}"
        )
        if tempdfs
        else []
    )
    comparison_options = [p for p in dfs.keys() if p != name]
    comparison_probes = st.multiselect(
        "Additional probe files to overlay",
        options=comparison_options,
        format_func=lambda opt: probe_labels.get(opt, opt),
        key=f"compare_{name}"
    )
    title = st.text_input(
        "Chart Title",
        value=f"{name} - {calendar.month_name[month]} {year}",
        key=f"title_{name}"
    )
    materials = st.text_input("Materials List", key=f"materials_{name}")
    probe_id = st.text_input("Probe ID", key=f"probe_{name}")
    equip_id = st.text_input("Equipment ID", key=f"equip_{name}")
    channel_keys = list(ranges[name].keys())
    channels = st.multiselect(
        "Channels to plot",
        options=channel_keys,
        default=channel_keys,
        key=f"channels_{name}"
    )

    if not st.button(f"Generate {name}", key=f"btn_{name}"):
        continue

    sel = df[(df['Date'].dt.year == year) & (df['Date'].dt.month == month)].sort_values('DateTime').reset_index(drop=True)
    if sel.empty:
        st.warning("No data for selected period.")
        continue

    start_date = datetime(year, month, 1)
    end_date = start_date + timedelta(days=calendar.monthrange(year, month)[1] - 1)

    for ch in channels:
        series_frames = []
        probe_legends = []
        tempstick_legends = []

        probe_label = probe_labels.get(name, name)
        probe_sub = sel[['DateTime', ch]].rename(columns={ch: 'Value'})
        probe_sub['Legend'] = probe_label
        series_frames.append(probe_sub)
        probe_legends.append(probe_label)

        for comp_name in comparison_probes:
            comp_df = dfs[comp_name]
            comp_sel = comp_df[
                (comp_df['Date'].dt.year == year)
                & (comp_df['Date'].dt.month == month)
            ].sort_values('DateTime').reset_index(drop=True)
            if ch not in comp_sel.columns:
                continue
            comp_label = probe_labels.get(comp_name, comp_name)
            comp_sub = comp_sel[['DateTime', ch]].rename(columns={ch: 'Value'})
            comp_sub['Legend'] = comp_label
            series_frames.append(comp_sub)
            probe_legends.append(comp_label)

        for ts_name in selected_tempsticks:
            ts_df = tempdfs.get(ts_name)
            if ts_df is None:
                continue
            ts_filtered = ts_df[
                (ts_df['DateTime'].dt.year == year)
                & (ts_df['DateTime'].dt.month == month)
            ].copy()
            ts_column = _match_tempstick_channel(ts_filtered, ch)
            if not ts_column:
                continue
            ts_label = tempstick_labels.get(ts_name, ts_name)
            ts_sub = ts_filtered[['DateTime', ts_column]].rename(columns={ts_column: 'Value'})
            ts_sub['Legend'] = ts_label
            series_frames.append(ts_sub)
            tempstick_legends.append(ts_label)

        if not series_frames:
            st.warning(f"No data available to chart for channel {ch}.")
            continue

        df_chart = pd.concat(series_frames, ignore_index=True).dropna(subset=['Value'])
        data_min = df_chart['Value'].min()
        data_max = df_chart['Value'].max()
        lo, hi = ranges[name].get(ch, (data_min, data_max))
        raw_min, raw_max = min(data_min, lo), max(data_max, hi)
        span = (raw_max - raw_min) or 1
        pad = span * 0.1
        ymin, ymax = raw_min - pad, raw_max + pad
        probe_palette = ['#1f77b4', '#9467bd', '#17becf', '#7f7f7f']
        tempstick_palette = ['#ff7f0e', '#bcbd22', '#8c564b', '#e377c2']
        color_map = {}
        for idx, legend in enumerate(probe_legends):
            color_map[legend] = probe_palette[idx % len(probe_palette)]
        for idx, legend in enumerate(tempstick_legends):
            color_map[legend] = tempstick_palette[idx % len(tempstick_palette)]
        color_map.update({'Lower Limit': '#2ca02c', 'Upper Limit': '#d62728'})
        legend_entries = probe_legends + tempstick_legends
        has_limits = ch in ranges[name]
        if has_limits:
            legend_entries.extend(['Lower Limit', 'Upper Limit'])
        color_domain = [entry for entry in color_map if entry in legend_entries]
        color_scale = alt.Scale(domain=color_domain, range=[color_map[e] for e in color_domain])
        base = alt.Chart(df_chart).encode(
            x=alt.X('DateTime:T', title='Date/Time', scale=alt.Scale(domain=[start_date, end_date])),
            y=alt.Y('Value:Q', title=f"{ch} ({'°C' if 'Temp' in ch else '%RH'})", scale=alt.Scale(domain=[ymin, ymax], nice=False)),
            color=alt.Color(
                'Legend:N',
                scale=color_scale,
                legend=alt.Legend(title='Series', labelLimit=0)
            )
        )
        line = base.mark_line()
        layers = [line]
        if has_limits:
            lo, hi = ranges[name][ch]
            limits_df = pd.DataFrame({'y': [lo, hi], 'Legend': ['Lower Limit', 'Upper Limit']})
            layers.append(
                alt.Chart(limits_df)
                .mark_rule(strokeDash=[4, 4])
                .encode(y='y:Q', color=alt.Color('Legend:N', scale=color_scale, legend=None))
            )
        title_lines = [
            f"{title} - {ch}",
            f"Materials: {materials} | Probe: {probe_id} | Equipment: {equip_id}"
        ]
        chart = (
            alt.layer(*layers)
            .properties(title={"text": title_lines, "anchor": "start"})
            .configure_title(fontSize=14, lineHeight=20, offset=10)
            .configure(background="white", view=alt.ViewConfig(fill="white", stroke=None))
        )
        st.altair_chart(chart, use_container_width=True)

    st.subheader("Out-of-Range Events")
    col_objs = st.columns(len(channels))
    for i, ch in enumerate(channels):
        ch_lo, ch_hi = ranges[name].get(ch, (None, None))
        if ch_lo is None or ch not in sel.columns:
            continue
        df_ch = sel[['DateTime', ch]].copy()
        df_ch['OOR'] = df_ch[ch].apply(lambda v: pd.notna(v) and (v < ch_lo or v > ch_hi))
        df_ch['Group'] = (df_ch['OOR'] != df_ch['OOR'].shift(fill_value=False)).cumsum()
        events = []
        for gid, grp in df_ch.groupby('Group'):
            if not grp['OOR'].iloc[0]:
                continue
            start = grp['DateTime'].iloc[0]
            last_idx = grp.index[-1]
            if last_idx + 1 < len(sel) and not df_ch.loc[last_idx+1, 'OOR']:
                end = sel.loc[last_idx+1, 'DateTime']
            else:
                end = grp['DateTime'].iloc[-1]
            duration = max((end - start).total_seconds() / 60, 0)
            events.append({'Start': start, 'End': end, 'Duration(min)': duration})
        with col_objs[i]:
            st.markdown(f"### {ch} OOR Events")
            if events:
                ev_df = pd.DataFrame(events)
                total = ev_df['Duration(min)'].sum()
                incident = total >= 60 or any(ev_df['Duration(min)'] > 60)
                st.table(ev_df)
                st.write(f"**Total OOR minutes:** {total:.1f}")
                st.write(f"**Incident:** {'YES' if incident else 'No'}")
            else:
                st.info("No out-of-range events detected.")
