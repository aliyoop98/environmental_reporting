import streamlit as st
import pandas as pd
import io
import calendar
import altair as alt
from datetime import datetime, timedelta

st.set_page_config(page_title="Environmental Reporting", layout="wide", page_icon="📈")

# Sidebar
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

# Parse Tempsticks
# ... (unchanged tempstick parsing) ...

# Parse Probes
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
    has_fridge = 'fridge' in lname
    has_freezer = 'freezer' in lname
    is_combo = has_fridge and has_freezer
    # Assign mappings and ranges
    if is_combo:
        mapping = {'Fridge Temp':'P1', 'Freezer Temp':'P2'}
        ranges[name] = {'Fridge Temp':(2,8), 'Freezer Temp':(-35,-5)}
    elif has_fridge:
        mapping = {'Temperature':'P1'}
        ranges[name] = {'Temperature':(2,8)}
    elif has_freezer:
        mapping = {'Temperature':'P1'}
        ranges[name] = {'Temperature':(-35,-5)}
    else:
        cols = [c.lower() for c in df.columns]
        if 'ch4' in ''.join(cols):
            mapping={'Humidity':'CH3','Temperature':'CH4'}
        else:
            mapping={'Humidity':'CH1','Temperature':'CH2'}
        temp_range=(15,28) if 'olympus' in lname else (15,25)
        ranges[name]={'Temperature':temp_range,'Humidity':(0,60)}
    mapping.update({'Date':'Date','Time':'Time'})
    col_map={}
    for new,key in mapping.items():
        col = next((c for c in df.columns if key.lower() in c.lower()),None)
        if col: col_map[col]=new
    df = df[list(col_map)].rename(columns=col_map)
    df['Date']=pd.to_datetime(df['Date'], errors='coerce')
    df['DateTime']=pd.to_datetime(df['Date'].dt.strftime('%Y-%m-%d ')+df['Time'], errors='coerce')
    for c in df.columns:
        if c not in ['Date','Time','DateTime']:
            df[c]=pd.to_numeric(df[c].astype(str).str.replace('+',''), errors='coerce')
    dfs[name]=df.reset_index(drop=True)

# Year & Month
years=sorted({d.year for df in dfs.values() for d in df['Date'].dropna()})
months=sorted({d.month for df in dfs.values() for d in df['Date'].dropna()})
year=st.sidebar.selectbox('Year',years)
month=st.sidebar.selectbox('Month',months,format_func=lambda m:calendar.month_name[m])
start_date=datetime(year,month,1)
end_date=start_date+timedelta(days=calendar.monthrange(year,month)[1]-1)

# Plot
for name,df in dfs.items():
    st.header(name)
    sel=df[(df['Date'].dt.year==year)&(df['Date'].dt.month==month)].sort_values('DateTime')
    if sel.empty:
        st.warning('No data for this period')
        continue
    channels=list(ranges[name].keys())
    for ch in channels:
        df_chart=sel[['DateTime',ch]].rename(columns={ch:'Value'})
        df_chart['Source']='Probe'
        low,high=ranges[name][ch]
        # determine y domain
        data_min=df_chart['Value'].min()
        data_max=df_chart['Value'].max()
        domain_min=min(data_min,low)
        domain_max=max(data_max,high)
        pad=(domain_max-domain_min)*0.1
        ymin=domain_min-pad
        ymax=domain_max+pad
        base=alt.Chart(df_chart).encode(
            x=alt.X('DateTime:T',title='Date/Time',scale=alt.Scale(domain=[start_date,end_date])),  
            y=alt.Y('Value:Q',title=f"{ch} ({'°C' if 'Temp' in ch else '%RH'})",scale=alt.Scale(domain=[ymin,ymax]))
        )
        line=base.mark_line()
        rules=[]
        rules.append(alt.Chart(pd.DataFrame({'y':[low]})).mark_rule(color='red',strokeDash=[5,5]).encode(y='y:Q'))
        rules.append(alt.Chart(pd.DataFrame({'y':[high]})).mark_rule(color='red',strokeDash=[5,5]).encode(y='y:Q'))
        chart=alt.layer(line,*rules).properties(title=f"{name} - {calendar.month_name[month]} {year} | {ch} Range [{low},{high}]")
        st.altair_chart(chart,use_container_width=True)
