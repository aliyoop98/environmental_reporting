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

# Parse Tempstick CSVs
# ... [parsing logic unchanged] ...

# Parse Probe CSVs
# ... [parsing logic unchanged] ...

# Year & Month selection
# ... [selection logic unchanged] ...

# Main visualization loop
for name, df in dfs.items():
    st.header(name)
    # ... [metadata inputs unchanged] ...
    sel = df[(df['Date']...)]  # filtered by year/month
    if sel.empty:
        st.warning("No data for selected period.")
        continue
    ts_df = tempdfs.get(ts_choice) if ts_choice else None

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
