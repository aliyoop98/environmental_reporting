import streamlit as st
import pandas as pd
import io
import calendar
import altair as alt

st.set_page_config(page_title="Environmental Reporting", layout="wide")
st.title("Environmental Monitoring - Full Pipeline & Visualization")

uploaded_files = st.file_uploader(
    label="Upload CSV files",
    type=["csv"],
    accept_multiple_files=True
)
if not uploaded_files:
    st.info("Please upload one or more CSV files to begin.")
    st.stop()

dfs = {}
ranges = {}
for uploaded in uploaded_files:
    name = uploaded.name
    raw = uploaded.read()
    lines = raw.decode('utf-8', errors='ignore').splitlines(True)
    try:
        header_idx = max(i for i, line in enumerate(lines)
                         if 'ch1' in line.lower() or 'p1' in line.lower())
    except ValueError:
        st.error(f"No header row found in {name}")
        continue
    csv_text = ''.join(lines[header_idx:])
    try:
        df = pd.read_csv(
            io.StringIO(csv_text),
            sep=',',
            on_bad_lines='skip',
            skip_blank_lines=True
        )
    except Exception as e:
        st.error(f"Failed to parse {name}: {e}")
        continue
    df.columns = [c.strip() for c in df.columns]
    lname = name.lower()
    has_fridge = 'fridge' in lname
    has_freezer = 'freezer' in lname
    if has_fridge and has_freezer:
        roles = {'Fridge Temp':'P1','Freezer Temp':'P2'}
    elif has_fridge or has_freezer:
        roles = {'Temperature':'P1'}
    else:
        roles = {'Humidity':'CH3','Temperature':'CH4'}
    roles.update({'Date':'Date','Time':'Time'})
    actual_to_role = {}
    missing = []
    for role, substr in roles.items():
        match = next((c for c in df.columns if substr.lower() in c.lower()), None)
        if match:
            actual_to_role[match] = role
        else:
            missing.append(substr)
    if missing:
        st.error(f"Missing columns in {name}: {missing}")
        continue
    df_clean = df[list(actual_to_role.keys())].rename(columns=actual_to_role)
    df_clean['Date'] = pd.to_datetime(df_clean['Date'], infer_datetime_format=True, errors='coerce')
    df_clean['DateTime'] = pd.to_datetime(
        df_clean['Date'].dt.strftime('%Y-%m-%d ') + df_clean['Time'].astype(str),
        errors='coerce'
    )
    dfs[name] = df_clean
    if has_fridge and has_freezer:
        rng = {'Fridge Temp':(2,8),'Freezer Temp':(-35,-5)}
    elif has_fridge:
        rng = {'Temperature':(2,8)}
    elif has_freezer:
        rng = {'Temperature':(-35,-5)}
    else:
        rng = {
            'Temperature':(15,28) if 'olympus' in lname else (15,25),
            'Humidity':(0,60)
        }
    ranges[name] = rng
    st.success(f"Loaded {name}")
    st.write(f"Default Ranges for {name}: {rng}")

if not dfs:
    st.stop()
years = sorted({dt.year for df in dfs.values() for dt in df['Date'].dropna()})
months = sorted({dt.month for df in dfs.values() for dt in df['Date'].dropna()})
col1, col2 = st.columns(2)
with col1:
    year = st.selectbox('Select Year', years)
with col2:
    month = st.selectbox('Select Month', months, format_func=lambda x: calendar.month_name[x])

for name, df in dfs.items():
    st.header(name)
    title = st.text_input(f"Chart Title for {name}", value=f"{name} - {calendar.month_name[month]} {year}")
    materials = st.text_input(f"Materials in {name}")
    probe_id = st.text_input(f"Probe ID for {name}")
    equipment_id = st.text_input(f"Equipment ID for {name}")
    df_sel = (
        df[(df['Date'].dt.year==year)&(df['Date'].dt.month==month)]
          .sort_values('DateTime')
          .reset_index(drop=True)
    )
    if df_sel.empty:
        st.write("No data for this period.")
        continue
    rng = ranges[name]
    def check_oor(r):
        for col,(mn,mx) in rng.items():
            val = r.get(col)
            if pd.notna(val):
                try:
                    v=float(val)
                    if v<mn or v>mx:
                        return True
                except:
                    return True
        return False
    df_sel['OutOfRange'] = df_sel.apply(check_oor, axis=1)
    has_fridge = 'fridge' in name.lower()
    has_freezer = 'freezer' in name.lower()
    if has_fridge and has_freezer:
        plot_keys = ['Fridge Temp','Freezer Temp']
    elif has_fridge or has_freezer:
        plot_keys = ['Temperature']
    else:
        plot_keys = ['Temperature','Humidity']
    for col in plot_keys:
        df_plot = df_sel[['DateTime',col,'OutOfRange']].rename(columns={col:'Value'})
        base = alt.Chart(df_plot).mark_line().encode(
            x=alt.X('DateTime:T', title='Date/Time'),
            y=alt.Y('Value:Q', title=f"{col} ({'°C' if 'Temp' in col else '%RH'})")
        )
        mn,mx = rng[col]
        rules = []
        if mn is not None:
            rules.append(
                alt.Chart(pd.DataFrame({'y':[mn]}))
                   .mark_rule(color='red',strokeDash=[4,4])
                   .encode(y='y:Q')
            )
        if mx is not None:
            rules.append(
                alt.Chart(pd.DataFrame({'y':[mx]}))
                   .mark_rule(color='red',strokeDash=[4,4])
                   .encode(y='y:Q')
            )
        points = (
            alt.Chart(df_plot[df_plot['OutOfRange']])
               .mark_circle(color='red',size=50)
               .encode(
                   x='DateTime:T',
                   y='Value:Q',
                   tooltip=['DateTime','Value']
               )
        )
        chart = base + points
        for r in rules:
            chart += r
        st.altair_chart(
            chart.properties(title=f"{title} - {col}
Materials: {materials} | Probe: {probe_id} | Equipment: {equipment_id}")
                  .interactive(),
            use_container_width=True
        )
    st.markdown(
        f"**Materials:** {materials}<br>**Probe ID:** {probe_id}<br>**Equipment ID:** {equipment_id}",
        unsafe_allow_html=True
    )
    events = []
    df_sel['GroupID'] = (df_sel['OutOfRange'] != df_sel['OutOfRange'].shift(fill_value=False)).cumsum()
    for gid, grp in df_sel.groupby('GroupID'):
        if not grp['OutOfRange'].iloc[0]: continue
        start = grp['DateTime'].iloc[0]
        last_idx = grp.index[-1]
        next_rows = df_sel.iloc[last_idx+1:]
        if not next_rows.empty:
            nxt = next_rows[next_rows['OutOfRange']==False]
            end = nxt['DateTime'].iloc[0] if not nxt.empty else grp['DateTime'].iloc[-1]
        else:
            end = grp['DateTime'].iloc[-1]
        dur = max((end-start).total_seconds()/60, 0)
        events.append({'Start':start,'End':end,'Duration(min)':dur})
    total = sum(e['Duration(min)'] for e in events)
    any_long = any(e['Duration(min)']>60 for e in events)
    incident = total>=60 or any_long
    if events:
        st.subheader('Out-of-Range Events')
        st.table(pd.DataFrame(events))
        st.write(f"Total out-of-range minutes: {total:.1f}")
        st.write(f"Incident? {'YES' if incident else 'No'}")
    else:
        st.write('No out-of-range events.')
