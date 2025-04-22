import streamlit as st
import pandas as pd
import io
import calendar
import altair as alt

st.set_page_config(page_title="Environmental Reporting", layout="wide", page_icon="📈")

# Sidebar for file upload and date selection
st.sidebar.header("Data Upload & Date Selection")
uploaded = st.sidebar.file_uploader("Upload CSV files", type=["csv"], accept_multiple_files=True)
if not uploaded:
    st.sidebar.info("Please upload CSV files to begin.")
    st.stop()

dfs = {}
ranges = {}
for file in uploaded:
    name = file.name
    raw = file.read()
    lines = raw.decode('utf-8', errors='ignore').splitlines(True)
    try:
        idx = max(i for i, L in enumerate(lines) if 'ch1' in L.lower() or 'p1' in L.lower())
    except ValueError:
        st.sidebar.error(f"No header in {name}")
        continue
    text = ''.join(lines[idx:])
    try:
        df = pd.read_csv(io.StringIO(text), on_bad_lines='skip').rename(columns=str.strip)
    except Exception as e:
        st.sidebar.error(f"Failed parsing {name}: {e}")
        continue
    lname = name.lower()
    fridge = 'fridge' in lname
    freezer = 'freezer' in lname
    if fridge and freezer:
        roles = {'Fridge Temp':'P1','Freezer Temp':'P2'}
    elif fridge or freezer:
        roles = {'Temperature':'P1'}
    else:
        roles = {'Humidity':'CH3','Temperature':'CH4'}
    roles.update({'Date':'Date','Time':'Time'})
    actual = {}
    missing = []
    for R, sub in roles.items():
        col = next((c for c in df.columns if sub.lower() in c.lower()), None)
        if col: actual[col] = R
        else: missing.append(sub)
    if missing:
        st.sidebar.error(f"{name} missing: {missing}")
        continue
    df = df[list(actual)].rename(columns=actual)
    df['Date'] = pd.to_datetime(df['Date'], infer_datetime_format=True, errors='coerce')
    df['DateTime'] = pd.to_datetime(df['Date'].dt.strftime('%Y-%m-%d ') + df['Time'].astype(str), errors='coerce')
    dfs[name] = df
    if fridge and freezer:
        ranges[name] = {'Fridge Temp':(2,8),'Freezer Temp':(-35,-5)}
    elif fridge:
        ranges[name] = {'Temperature':(2,8)}
    elif freezer:
        ranges[name] = {'Temperature':(-35,-5)}
    else:
        ranges[name] = {
            'Temperature':(15,28) if 'olympus' in lname else (15,25),
            'Humidity':(0,60)
        }

# Year & Month selectors
years = sorted({d.year for df in dfs.values() for d in df['Date'].dropna()})
months = sorted({d.month for df in dfs.values() for d in df['Date'].dropna()})
year = st.sidebar.selectbox("Year", years)
month = st.sidebar.selectbox("Month", months, format_func=lambda m: calendar.month_name[m])

# Main: iterate files
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
        sel = df[(df['Date'].dt.year==year)&(df['Date'].dt.month==month)].sort_values('DateTime')
        if sel.empty:
            st.warning("No data for selected period.")
            continue
        rng = ranges[name]
        def flag(r):
            for c,(low,high) in rng.items():
                v = r.get(c)
                if pd.notna(v) and (v<low or v>high): return True
            return False
        sel['OOR'] = sel.apply(flag, axis=1)
        fridge = 'fridge' in name.lower()
        freezer = 'freezer' in name.lower()
        channels = []
        if fridge and freezer:
            channels = ['Fridge Temp','Freezer Temp']
        elif fridge or freezer:
            channels = ['Temperature']
        else:
            channels = ['Temperature','Humidity']
        cols = st.columns(len(channels))
        for col, colkey in zip(channels, cols):
            sub = sel[['DateTime',col,'OOR']].rename(columns={col:'Value'})
            chart = alt.Chart(sub).mark_line().encode(
                x=alt.X('DateTime:T', title='Date/Time'),
                y=alt.Y('Value:Q', title=col + (' (°C)' if 'Temp' in col else ' (%RH)'))
            )
            rules = []
            low, high = rng.get(col, (None,None))
            if low is not None:
                rules.append(alt.Chart(pd.DataFrame({'y':[low]})).mark_rule(color='red',strokeDash=[4,4]).encode(y='y:Q'))
            if high is not None:
                rules.append(alt.Chart(pd.DataFrame({'y':[high]})).mark_rule(color='red',strokeDash=[4,4]).encode(y='y:Q'))
            points = alt.Chart(sub[sub['OOR']]).mark_point(color='red',size=50).encode(x='DateTime:T',y='Value:Q',tooltip=['DateTime','Value'])
            final = chart + points
            for r in rules: final |= r
            final = final.properties(title=title + f" | Materials: {materials} | Probe: {probe_id} | Equipment: {equip_id}").interactive()
            colkey.altair_chart(final, use_container_width=True)
        # Show events
        sel['G'] = (sel['OOR'] != sel['OOR'].shift(fill_value=False)).cumsum()
        ev = []
        for gid, grp in sel.groupby('G'):
            if not grp['OOR'].iloc[0]: continue
            start = grp['DateTime'].iloc[0]
            idx = grp.index[-1]
            nxt = sel.iloc[idx+1:]
            end = nxt[nxt['OOR']==False]['DateTime'].iloc[0] if not nxt[nxt['OOR']==False].empty else grp['DateTime'].iloc[-1]
            dur = max((end-start).total_seconds()/60,0)
            ev.append({'Start':start,'End':end,'Duration(min)':dur})
        if ev:
            st.markdown("**Out-of-Range Events**")
            st.table(pd.DataFrame(ev))
            total = sum(e['Duration(min)'] for e in ev)
            incident = total>=60 or any(e['Duration(min)']>60 for e in ev)
            st.write(f"Total OOR minutes: {total:.1f} | Incident: {'YES' if incident else 'No'}")
