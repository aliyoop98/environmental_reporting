import streamlit as st
import pandas as pd
import io
import calendar
import altair as alt
from datetime import datetime, timedelta

st.set_page_config(page_title="Environmental Reporting", layout="wide", page_icon="📈")

# Sidebar configuration
st.sidebar.header("Data Upload & Configuration")
probe_files = st.sidebar.file_uploader(
    "Upload Probe CSV files", type=["csv"], accept_multiple_files=True
)
if not probe_files:
    st.sidebar.info("Upload Probe CSV files to begin.")
    st.stop()

tempstick_files = st.sidebar.file_uploader(
    "Upload Tempstick CSV files (optional)",
    type=["csv"], accept_multiple_files=True, key="tempstick_uploads"
)

# Initialize Tempstick data dictionary
tempdfs = {}

# Parse Tempstick CSVs
# ... [parsing logic unchanged] ...

# Parse Probe CSVs
parsed_probes = {}
for f in probe_files:
    raw_bytes = f.read()
    text = raw_bytes.decode('utf-8', errors='ignore')
    lines = text.splitlines(True)
    try:
        header_idx = max(i for i, line in enumerate(lines)
                         if 'ch1' in line.lower() or 'p1' in line.lower())
    except ValueError:
        st.warning(f"No valid header row found in {f.name}, skipping.")
        continue
    csv_text = ''.join(lines[header_idx:])
    try:
        df = pd.read_csv(io.StringIO(csv_text), on_bad_lines='skip', skip_blank_lines=True)
    except Exception as e:
        st.error(f"Failed to parse {f.name}: {e}")
        continue
    df.columns = [c.strip() for c in df.columns]
    col_map = {}
    for c in df.columns:
        lc = c.lower()
        if 'date' in lc:
            col_map[c] = 'Date'
        elif 'time' in lc:
            col_map[c] = 'Time'
        elif 'p1' in lc:
            col_map[c] = 'P1'
        elif 'p2' in lc:
            col_map[c] = 'P2'
        elif 'ch3' in lc:
            col_map[c] = 'CH3'
        elif 'ch4' in lc:
            col_map[c] = 'CH4'
        elif 'ch1' in lc:
            col_map[c] = 'CH3'
        elif 'ch2' in lc:
            col_map[c] = 'CH4'
    df = df.rename(columns=col_map)
    parsed_probes[f.name] = df

# After parsing probe CSVs, convert Date and build DateTime
for name, df in parsed_probes.items():
    df['Date'] = pd.to_datetime(df['Date'], infer_datetime_format=True, errors='coerce')
    df['DateTime'] = pd.to_datetime(
        df['Date'].dt.strftime('%Y-%m-%d') + ' ' + df['Time'],
        infer_datetime_format=True,
        errors='coerce'
    )
# Collect into dfs
dfs = parsed_probes

# Define default threshold ranges for each file based on type
ranges = {}
for name, df in dfs.items():
    lname = name.lower()
    has_fridge = 'fridge' in lname
    has_freezer = 'freezer' in lname
    is_combo = has_fridge and has_freezer
    is_room = not (has_fridge or has_freezer)
    is_olympus = 'olympus' in lname
    r = {}
    if is_combo:
        if 'P1' in df.columns:
            r['P1'] = (2, 8)
        if 'P2' in df.columns:
            r['P2'] = (-35, -5)
    elif has_fridge and not has_freezer:
        if 'P1' in df.columns:
            r['P1'] = (2, 8)
    elif has_freezer and not has_fridge:
        if 'P1' in df.columns:
            r['P1'] = (-35, -5)
    elif is_room:
        if 'CH3' in df.columns:
            r['CH3'] = (0, 60)
        if 'CH4' in df.columns:
            r['CH4'] = (15, 28 if is_olympus else 25)
    ranges[name] = r

# Year & Month selection
# Collect available years across all probes
all_years = set()
for df in dfs.values():
    if 'Date' in df.columns:
        all_years.update(df['Date'].dt.year.dropna().astype(int).unique())
all_years = sorted(all_years)
if not all_years:
    st.error("No valid dates found in uploaded data.")
    st.stop()
year = st.sidebar.selectbox("Select Year", all_years)
# Month selection
month = st.sidebar.selectbox(
    "Select Month",
    list(range(1, 13)),
    format_func=lambda m: calendar.month_name[m]
)

# Main visualization loop
for name, df in dfs.items():
    st.header(name)
    # ... [metadata inputs unchanged] ...
    sel = df[(df['Date'].dt.year == year) & (df['Date'].dt.month == month)].copy().reset_index(drop=True)
    if sel.empty:
        st.warning("No data for selected period.")
        continue
    ts_df = None  # Tempstick overlay not available

    # Determine which channels to plot based on ranges defined
    channels = list(ranges[name].keys())
    # Plot each channel
    for ch in channels:
        df_chart = sel[['DateTime', ch]].rename(columns={ch: 'Value'})
        df_chart['Source'] = 'Probe'
        if ts_df is not None and ch in ts_df.columns:
            ts_sub = ts_df[['DateTime', ch]].rename(columns={ch: 'Value'})
            ts_sub['Source'] = 'Tempstick'
            df_chart = pd.concat([df_chart, ts_sub], ignore_index=True)
        # Compute domain including thresholds
        low_val, high_val = ranges[name][ch]
        data_min = df_chart['Value'].min()
        data_max = df_chart['Value'].max()
        domain_min = min(data_min, low_val)
        domain_max = max(data_max, high_val)
        # Build chart
        line = alt.Chart(df_chart).mark_line().encode(
            x=alt.X('DateTime:T', title='Date/Time', scale=alt.Scale(domain=[
                datetime(year, month, 1),
                datetime(year, month, calendar.monthrange(year, month)[1])
            ])),
            y=alt.Y('Value:Q', title=f"{ch}", scale=alt.Scale(domain=[domain_min, domain_max])),
            color='Source:N'
        )
        # Add threshold rules
        rules = alt.Chart(pd.DataFrame({'threshold': [low_val, high_val]})).mark_rule(color='red', strokeDash=[5,5]).encode(y='threshold:Q')
        chart = (line + rules).properties(
            title=f"{title} - {ch} | Materials: {materials} | Probe: {probe_id} | Equipment: {equip_id}"
        )
        st.altair_chart(chart, use_container_width=True)

    # Flag and summarize OOR events
    sel['OOR'] = sel.apply(
        lambda r: any(r[c] < lo or r[c] > hi for c, (lo, hi) in ranges[name].items()), axis=1
    )
    sel['Group'] = (sel['OOR'] != sel['OOR'].shift(fill_value=False)).cumsum()
    events = []
    for gid, grp in sel.groupby('Group'):
        if not grp['OOR'].iloc[0]: continue
        start = grp['DateTime'].iloc[0]
        last_idx = grp.index[-1]
        if last_idx + 1 < len(sel) and not sel.loc[last_idx+1, 'OOR']:
            end = sel.loc[last_idx+1, 'DateTime']
        else:
            end = grp['DateTime'].iloc[-1]
        duration = (end - start).total_seconds() / 60
        events.append({'Start': start, 'End': end, 'Duration(min)': max(duration, 0)})
    if events:
        ev_df = pd.DataFrame(events)
        total = ev_df['Duration(min)'].sum()
        incident = total >= 60 or any(ev_df['Duration(min)'] > 60)
        st.subheader("Out-of-Range Events")
        st.table(ev_df)
        st.write(f"Total OOR minutes: {total:.1f} | Incident: {'YES' if incident else 'No'}")
