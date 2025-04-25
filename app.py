import streamlit as st
import pandas as pd
import io
import calendar
import altair as alt
from datetime import datetime, timedelta

st.set_page_config(page_title="Environmental Reporting", layout="wide", page_icon="📈")

st.sidebar.header("Data Upload & Configuration")
probe_files = st.sidebar.file_uploader("Upload Probe CSV files", type=["csv"], accept_multiple_files=True)
if not probe_files:
    st.sidebar.info("Upload Probe CSV files to begin.")
    st.stop()

tempstick_files = st.sidebar.file_uploader("Upload Tempstick CSV files (optional)", type=["csv"], accept_multiple_files=True, key="tempsticks")

# Parse Tempstick CSVs
tempdfs = {}
if tempstick_files:
    for f in tempstick_files:
        raw = f.read().decode('utf-8', errors='ignore').splitlines(True)
        try:
            idx = max(i for i, line in enumerate(raw) if 'timestamp' in line.lower())
        except ValueError:
            st.warning(f"No valid header row found in Tempstick {f.name}, skipping.")
            continue
        csv_text = ''.join(raw[idx:])
        df_ts = pd.read_csv(io.StringIO(csv_text), on_bad_lines='skip').rename(columns=str.strip)
        ts_col = next((c for c in df_ts.columns if 'timestamp' in c.lower()), None)
        if not ts_col:
            st.warning(f"No Timestamp column in Tempstick {f.name}, skipping.")
            continue
        df_ts = df_ts.rename(columns={ts_col: 'DateTime'})
        df_ts['DateTime'] = pd.to_datetime(df_ts['DateTime'], infer_datetime_format=True, errors='coerce')
        temp_col = next((c for c in df_ts.columns if 'temp' in c.lower()), None)
        if temp_col:
            df_ts['Temperature'] = pd.to_numeric(df_ts[temp_col], errors='coerce')
        hum_col = next((c for c in df_ts.columns if 'hum' in c.lower()), None)
        if hum_col:
            df_ts['Humidity'] = pd.to_numeric(df_ts[hum_col], errors='coerce')
        cols = ['DateTime']
        if 'Temperature' in df_ts.columns:
            cols.append('Temperature')
        if 'Humidity' in df_ts.columns:
            cols.append('Humidity')
        tempdfs[f.name] = df_ts[cols]

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
    df = pd.read_csv(io.StringIO(content), on_bad_lines='skip').rename(columns=str.strip)
    lname = name.lower()
    # Determine mapping and ranges
    has_fridge = 'fridge' in lname
    has_freezer = 'freezer' in lname
    if has_fridge and has_freezer:
        mapping = {'Fridge Temp':'P1','Freezer Temp':'P2'}
        ranges[name] = {'Fridge Temp':(2,8),'Freezer Temp':(-35,-5)}
    elif has_fridge:
        mapping = {'Temperature':'P1'}
        ranges[name] = {'Temperature':(2,8)}
    elif has_freezer:
        mapping = {'Temperature':'P1'}
        ranges[name] = {'Temperature':(-35,-5)}
    else:
        cols_lower = [c.lower() for c in df.columns]
        if any('ch4' in c for c in cols_lower):
            mapping = {'Humidity':'CH3','Temperature':'CH4'}
        else:
            mapping = {'Humidity':'CH1','Temperature':'CH2'}
        temp_range = (15,28) if 'olympus' in lname else (15,25)
        ranges[name] = {'Temperature':temp_range,'Humidity':(0,60)}
    mapping.update({'Date':'Date','Time':'Time'})
    col_map = {}
    for new, key in mapping.items():
        col = next((c for c in df.columns if key.lower() in c.lower()), None)
        if col:
            col_map[col] = new
    df = df[list(col_map)].rename(columns=col_map)
    df['Date'] = pd.to_datetime(df['Date'], infer_datetime_format=True, errors='coerce')
    df['DateTime'] = pd.to_datetime(df['Date'].dt.strftime('%Y-%m-%d ') + df['Time'].astype(str), errors='coerce')
    for c in df.columns:
        if c not in ['Date','Time','DateTime']:
            df[c] = pd.to_numeric(df[c].astype(str).str.replace('+',''), errors='coerce')
    dfs[name] = df.reset_index(drop=True)

# Year and Month selection
years = sorted({dt.year for df in dfs.values() for dt in df['Date'].dropna()})
months = sorted({dt.month for df in dfs.values() for dt in df['Date'].dropna()})
year = st.sidebar.selectbox("Year", years)
month = st.sidebar.selectbox("Month", months, format_func=lambda m: calendar.month_name[m])

# Plot each sensor
for name, df in dfs.items():
    st.header(name)
    with st.expander("Configuration & Chart", expanded=True):
        ts_choice = None
        if tempdfs:
            ts_choice = st.selectbox(f"Match Tempstick for {name}", [None]+list(tempdfs.keys()), key=name)
        title = st.text_input("Chart Title", value=f"{name} - {calendar.month_name[month]} {year}")
        materials = st.text_input("Materials List")
        probe_id = st.text_input("Probe ID")
        equip_id = st.text_input("Equipment ID")
        channels = st.multiselect("Channels to plot", options=list(ranges[name].keys()), default=list(ranges[name].keys()), key=name)
        if not st.button(f"Generate {name}", key=name+"btn"): continue
        sel = df[(df['Date'].dt.year==year)&(df['Date'].dt.month==month)].sort_values('DateTime').reset_index(drop=True)
        if sel.empty:
            st.warning("No data for selected period.")
            continue
        # Prepare Tempstick selection
        ts_df = None
        if ts_choice:
            ts_df = tempdfs.get(ts_choice)
            ts_df = ts_df[(ts_df['DateTime'].dt.year==year)&(ts_df['DateTime'].dt.month==month)]
        # Chart domain
        start_date = datetime(year,month,1)
        end_date = start_date + timedelta(days=calendar.monthrange(year,month)[1]-1)
        for ch in channels:
            probe_sub = sel[['DateTime', ch]].rename(columns={ch:'Value'})
            probe_sub['Source'] = 'Probe'
            df_chart = probe_sub.copy()
            if ts_df is not None and ch in ts_df.columns:
                ts_sub = ts_df[['DateTime', ch]].rename(columns={ch:'Value'})
                ts_sub['Source'] = 'Tempstick'
                df_chart = pd.concat([probe_sub, ts_sub], ignore_index=True)
            low, high = ranges[name].get(ch, (None, None))
            data_min = df_chart['Value'].min()
            data_max = df_chart['Value'].max()
            span = (high if high is not None else data_max) - (low if low is not None else data_min)
            pad = span * 0.1
            ymin = (low if low is not None else data_min) - pad
            ymax = (high if high is not None else data_max) + pad
            chart = alt.Chart(df_chart).mark_line().encode(
                x=alt.X('DateTime:T', title='Date/Time', scale=alt.Scale(domain=[start_date, end_date])),
                y=alt.Y('Value:Q', title=f"{ch} ({'°C' if 'Temp' in ch else '%RH'})", scale=alt.Scale(domain=[ymin, ymax])),
                color='Source:N'
            )
            st.altair_chart(chart, use_container_width=True)
        # OOR events
        st.subheader("Out-of-Range Events")
        sel['OOR'] = sel.apply(lambda r: any((r[ch]<lo or r[ch]>hi) for ch,(lo,hi) in ranges[name].items() if pd.notna(r[ch])), axis=1)
        sel['Group'] = (sel['OOR'] != sel['OOR'].shift(fill_value=False)).cumsum()
        events = []
        for gid, grp in sel.groupby('Group'):
            if not grp['OOR'].iloc[0]: continue
            start = grp['DateTime'].iloc[0]
            last_idx = grp.index[-1]
            if last_idx+1 < len(sel) and not sel.loc[last_idx+1,'OOR']:
                end = sel.loc[last_idx+1,'DateTime']
            else:
                end = grp['DateTime'].iloc[-1]
            duration = max((end-start).total_seconds()/60, 0)
            events.append({'Start':start,'End':end,'Duration(min)':duration})
        if events:
            ev_df = pd.DataFrame(events)
            total = ev_df['Duration(min)'].sum()
            incident = total>=60 or any(ev_df['Duration(min)']>60)
            st.table(ev_df)
            st.write(f"Total OOR minutes: {total:.1f} | Incident: {'YES' if incident else 'No'}")
