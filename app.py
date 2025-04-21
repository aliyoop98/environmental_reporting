import streamlit as st
import pandas as pd
import io

# Page config
st.set_page_config(page_title="Environmental Reporting", layout="wide")
st.title("Environmental Monitoring Out‑of‑Range Detector")

# 1. Upload CSV files
uploaded = st.file_uploader("Upload CSV files", type="csv", accept_multiple_files=True)
if not uploaded:
    st.info("Please upload one or more CSV files to begin.")
    st.stop()

# 2. Strip metadata & load DataFrames
def strip_metadata(content, keys=["CH1","P1","Temp","Humidity"]):
    lines = content.decode('utf-8', errors='ignore').splitlines(True)
    for i, line in enumerate(lines):
        if any(k.lower() in line.lower() for k in keys):
            return ''.join(lines[i:])
    raise ValueError("No header row found")

# Prepare container for file settings
def default_ranges(loc_type):
    defaults = {
        'Fridge':   {'temp_min':2,'temp_max':8,'hum_min':None,'hum_max':None},
        'Freezer':  {'temp_min':-35,'temp_max':-5,'hum_min':None,'hum_max':None},
        'Combo':    {'temp_min':-35,'temp_max':8,'hum_min':None,'hum_max':None},
        'Room':     {'temp_min':15,'temp_max':25,'hum_min':0,'hum_max':60},
        'Olympus':  {'temp_min':15,'temp_max':28,'hum_min':0,'hum_max':60}
    }
    return defaults.get(loc_type, defaults['Room'])

settings = {}
for file in uploaded:
    # sidebar settings per file
    with st.sidebar.expander(file.name, expanded=False):
        st.write(f"Configure: {file.name}")
        loc = st.selectbox("Location Type", options=['Fridge','Freezer','Combo','Room','Olympus'], key=file.name+"_loc")
        dr = default_ranges(loc)
        tmin = st.number_input("Temp Min", value=dr['temp_min'], key=file.name+"_tmin")
        tmax = st.number_input("Temp Max", value=dr['temp_max'], key=file.name+"_tmax")
        hmin = st.number_input("Hum Min", value=dr['hum_min'] if dr['hum_min'] is not None else 0, key=file.name+"_hmin")
        hmax = st.number_input("Hum Max", value=dr['hum_max'] if dr['hum_max'] is not None else 100, key=file.name+"_hmax")
        settings[file.name] = {
            'location_type': loc,
            'temp_range': (tmin, tmax) if (tmin is not None and tmax is not None) else None,
            'humidity_range': (hmin, hmax) if (hmin is not None and hmax is not None) else None
        }

# 3. Process button
if st.button("Process All Files"):
    for file in uploaded:
        content = strip_metadata(file.read())
        try:
            df = pd.read_csv(io.StringIO(content), low_memory=False)
        except Exception:
            # Fallback for irregular CSVs
            df = pd.read_csv(io.StringIO(content), engine='python', on_bad_lines='skip', low_memory=False)
        # Normalize cols
        df = df.rename(columns=lambda c: c.strip())
        col_map = {}
        for c in df.columns:
            lc = c.lower()
            if 'temp' in lc or 'ch4' in lc or 'p1' in lc or 'ch1' in lc:
                col_map[c] = 'Temp'
            elif 'hum' in lc or 'ch3' in lc:
                col_map[c] = 'Humidity'
            elif 'date' in lc:
                col_map[c] = 'Date'
            elif 'time' in lc:
                col_map[c] = 'Time'
        df = df.rename(columns=col_map)
        df = df[['Date','Time','Temp','Humidity']]

        # Parse DateTime
        df['DateTime'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'].astype(str), errors='raise')

        # Get settings
        s = settings[file.name]
        loc_type = s['location_type']
        temp_range = None if loc_type=='Olympus' else s['temp_range']
        hum_range = s['humidity_range']

        # Flag out-of-range
        def oor(r):
            if temp_range and pd.notna(r['Temp']):
                t = float(r['Temp'])
                if t<temp_range[0] or t>temp_range[1]: return True
            if hum_range and pd.notna(r['Humidity']):
                h = float(str(r['Humidity']).lstrip('+'))
                if h<hum_range[0] or h>hum_range[1]: return True
            return False
        df['OutOfRange'] = df.apply(oor, axis=1)

        # Group events
        df['GroupID'] = (df['OutOfRange'] != df['OutOfRange'].shift(fill_value=False)).cumsum()
        events = []
        for gid, grp in df[df['OutOfRange']].groupby('GroupID'):
            start = grp['DateTime'].iloc[0]; end = grp['DateTime'].iloc[-1]
            dur = (end-start).total_seconds()/60
            events.append({'Start':start,'End':end,'Duration(min)':dur})

        # Display
        st.header(file.name)
        st.metric("Location Type", loc_type)
        st.write(pd.DataFrame(events))
    st.success("Processing complete!")
