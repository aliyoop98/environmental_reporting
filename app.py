import streamlit as st
import pandas as pd
import io
import calendar
import altair as alt

st.set_page_config(page_title="Environmental Reporting", layout="wide", page_icon="📈")

st.sidebar.header("Data Upload & Date Selection")
uploaded = st.sidebar.file_uploader("Upload CSV files", type=["csv"], accept_multiple_files=True)
if not uploaded:
    st.sidebar.info("Please upload CSV files to begin.")
    st.stop()

dfs = {}
ranges = {}
for file in uploaded:
    name = file.name
    raw = file.read().decode('utf-8', errors='ignore').splitlines(True)
    try:
        idx = max(i for i, line in enumerate(raw) if 'ch1' in line.lower() or 'p1' in line.lower())
    except ValueError:
        st.sidebar.error(f"No valid header in {name}")
        continue
    content = ''.join(raw[idx:])
    try:
        df = pd.read_csv(io.StringIO(content), on_bad_lines='skip').rename(columns=str.strip)
    except Exception as e:
        st.sidebar.error(f"Failed parsing {name}: {e}")
        continue
    lname = name.lower()
    has_fridge = 'fridge' in lname
    has_freezer = 'freezer' in lname
    if has_fridge and has_freezer:
        mapping = {'Fridge Temp':'P1', 'Freezer Temp':'P2'}
    elif has_fridge or has_freezer:
        mapping = {'Temperature':'P1'}
    else:
        mapping = {'Humidity':'CH3', 'Temperature':'CH4'}
    mapping.update({'Date':'Date', 'Time':'Time'})
    col_map = {}
    missing = []
    for new, key in mapping.items():
        col = next((c for c in df.columns if key.lower() in c.lower()), None)
        if col:
            col_map[col] = new
        else:
            missing.append(key)
    if missing:
        st.sidebar.error(f"{name} missing columns: {missing}")
        continue
    df = df[list(col_map)].rename(columns=col_map)
    df['Date'] = pd.to_datetime(df['Date'], infer_datetime_format=True, errors='coerce')
    df['DateTime'] = pd.to_datetime(df['Date'].dt.strftime('%Y-%m-%d ') + df['Time'].astype(str), errors='coerce')
    for col in [c for c in df.columns if c not in ['Date', 'Time', 'DateTime']]:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace('+', ''), errors='coerce')
    df = df.reset_index(drop=True)
    dfs[name] = df
    if has_fridge and has_freezer:
        ranges[name] = {'Fridge Temp':(2,8), 'Freezer Temp':(-35,-5)}
    elif has_fridge:
        ranges[name] = {'Temperature':(2,8)}
    elif has_freezer:
        ranges[name] = {'Temperature':(-35,-5)}
    else:
        temp_range = (15,28) if 'olympus' in lname else (15,25)
        ranges[name] = {'Temperature':temp_range, 'Humidity':(0,60)}

years = sorted({d.year for df in dfs.values() for d in df['Date'].dropna()})
months = sorted({d.month for df in dfs.values() for d in df['Date'].dropna()})
if not years or not months:
    st.sidebar.error("No valid date data found.")
    st.stop()
year = st.sidebar.selectbox("Year", years)
month = st.sidebar.selectbox("Month", months, format_func=lambda m: calendar.month_name[m])

for name, df in dfs.items():
    st.markdown(f"## {name}")
    with st.expander("Chart & Metadata", expanded=True):
        form = st.form(key=name)
        title = form.text_input("Chart Title", value=f"{name} - {calendar.month_name[month]} {year}")
        materials = form.text_input("Materials List")
        probe_id = form.text_input("Probe ID")
        equip_id = form.text_input("Equipment ID")
        submit = form.form_submit_button("Generate Charts")
        if not submit:
            st.info("Enter metadata and click 'Generate Charts'.")
            continue
        sel = df[(df['Date'].dt.year==year) & (df['Date'].dt.month==month)].sort_values('DateTime').reset_index(drop=True)
        if sel.empty:
            st.warning("No data for selected period.")
            continue
        rng = ranges[name]
        def is_oor(row):
            for col, (low, high) in rng.items():
                v = row.get(col)
                if pd.notna(v) and (v < low or v > high):
                    return True
            return False
        sel['OOR'] = sel.apply(is_oor, axis=1)
        has_fridge = 'fridge' in name.lower()
        has_freezer = 'freezer' in name.lower()
        if has_fridge and has_freezer:
            channels = ['Fridge Temp','Freezer Temp']
        elif has_fridge or has_freezer:
            channels = ['Temperature']
        else:
            channels = ['Temperature','Humidity']
        for col_name in channels:
            sub = sel[['DateTime', col_name, 'OOR']].rename(columns={col_name:'Value'})
            base = alt.Chart(sub).mark_line().encode(
                x=alt.X('DateTime:T', title='Date/Time'),
                y=alt.Y('Value:Q', title=f"{col_name} ({'°C' if 'Temp' in col_name else '%RH'})"),
                color=alt.value('blue')
            )
            pts = alt.Chart(sub[sub['OOR']]).mark_point(color='red', size=50).encode(
                x='DateTime:T', y='Value:Q', tooltip=['DateTime','Value']
            )
            rules = []
            low, high = rng.get(col_name, (None, None))
            if low is not None:
                rules.append(alt.Chart(pd.DataFrame({'y':[low]})).mark_rule(color='red',strokeDash=[4,4]).encode(y='y:Q'))
            if high is not None:
                rules.append(alt.Chart(pd.DataFrame({'y':[high]})).mark_rule(color='red',strokeDash=[4,4]).encode(y='y:Q'))
            chart = base + pts
            for rule in rules:
                chart = chart + rule
            chart = chart.properties(
                title=f"{title} | Materials: {materials} | Probe: {probe_id} | Equipment: {equip_id} | {col_name}"
            ).interactive()
            st.altair_chart(chart, use_container_width=True)
        sel['Group'] = (sel['OOR'] != sel['OOR'].shift(fill_value=False)).cumsum()
        events = []
        for gid, grp in sel.groupby('Group'):
            if not grp['OOR'].iloc[0]:
                continue
            start = grp['DateTime'].iloc[0]
            last_idx = grp.index[-1]
            if last_idx + 1 < len(sel) and not sel.loc[last_idx+1, 'OOR']:
                end = sel.loc[last_idx+1, 'DateTime']
            else:
                end = grp['DateTime'].iloc[-1]
            duration = max((end - start).total_seconds() / 60, 0)
            events.append({'Start':start, 'End':end, 'Duration(min)':duration})
        if events:
            st.markdown("**Out-of-Range Events**")
            st.table(pd.DataFrame(events))
            total = sum(e['Duration(min)'] for e in events)
            incident = total >= 60 or any(e['Duration(min)'] > 60 for e in events)
            st.write(f"Total OOR minutes: {total:.1f} | Incident: {'YES' if incident else 'No'}")
