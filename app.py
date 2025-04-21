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

# 2. Metadata stripping helper
def strip_metadata(content, keys=["CH1","P1","Temp","Humidity"]):
    text = content.decode('utf-8', errors='ignore').splitlines(True)
    for i, line in enumerate(text):
        if any(k.lower() in line.lower() for k in keys):
            return ''.join(text[i:])
    return ''.join(text)  # if no metadata found, return all

# Default ranges per location type
def default_ranges(loc_type):
    return {
        'Fridge':  {'temp': (2, 8),     'hum': None},
        'Freezer': {'temp': (-35, -5),  'hum': None},
        'Combo':   {'temp': (-35, 8),   'hum': None},
        'Room':    {'temp': (15, 25),   'hum': (0, 60)},
        'Olympus': {'temp': None,       'hum': (0, 60)}
    }.get(loc_type, {'temp': (15,25), 'hum': (0,60)})

# Collect settings interactively
settings = {}
for file in uploaded:
    with st.sidebar.expander(file.name):
        st.write(f"Configure: **{file.name}**")
        loc = st.selectbox("Location Type", ['Fridge','Freezer','Combo','Room','Olympus'], key=file.name+"_loc")
        ranges = default_ranges(loc)
        if ranges['temp']:
            tmin, tmax = ranges['temp']
            tmin = st.number_input("Temp Min", value=tmin, key=file.name+"_tmin")
            tmax = st.number_input("Temp Max", value=tmax, key=file.name+"_tmax")
            temp_range = (tmin, tmax)
        else:
            temp_range = None
        if ranges['hum']:
            hmin, hmax = ranges['hum']
            hmin = st.number_input("Hum Min", value=hmin, key=file.name+"_hmin")
            hmax = st.number_input("Hum Max", value=hmax, key=file.name+"_hmax")
            hum_range = (hmin, hmax)
        else:
            hum_range = None
        settings[file.name] = {'loc': loc, 'temp': temp_range, 'hum': hum_range}

# 3. Process data
if st.button("Process All Files"):
    for file in uploaded:
        raw = file.read()
        text = strip_metadata(raw)
        # Attempt to parse CSV
        try:
            df = pd.read_csv(io.StringIO(text), on_bad_lines='skip', skip_blank_lines=True)
        except Exception as e:
            st.error(f"Error parsing {file.name}: {e}")
            continue

        # Normalize headers
        df.columns = df.columns.str.strip()
        col_map = {}
        for c in df.columns:
            lc = c.lower()
            if 'temp' in lc or 'ch4' in lc or 'p1' in lc:
                col_map[c] = 'Temp'
            elif 'hum' in lc or 'ch3' in lc:
                col_map[c] = 'Humidity'
            elif 'date' in lc:
                col_map[c] = 'Date'
            elif 'time' in lc:
                col_map[c] = 'Time'
        df = df.rename(columns=col_map)

        # Check for essential columns
        if 'Date' not in df.columns or 'Time' not in df.columns:
            st.error(f"Missing Date or Time in {file.name}")
            continue

        # Parse DateTime
        df['DateTime'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'].astype(str), errors='raise')

        # Fetch user settings
        cfg = settings[file.name]
        loc = cfg['loc']
        temp_range = cfg['temp'] if loc!='Olympus' else None
        hum_range = cfg['hum']

        # Flag out-of-range
        def is_oor(row):
            if temp_range and 'Temp' in row and pd.notna(row['Temp']):
                t = float(row['Temp'])
                if t < temp_range[0] or t > temp_range[1]: return True
            if hum_range and 'Humidity' in row and pd.notna(row['Humidity']):
                h = float(str(row['Humidity']).lstrip('+'))
                if h < hum_range[0] or h > hum_range[1]: return True
            return False
        df['OutOfRange'] = df.apply(is_oor, axis=1)

        # Group contiguous events
        df['Group'] = (df['OutOfRange'] != df['OutOfRange'].shift(fill_value=False)).cumsum()
        events = []
        for gid, grp in df[df['OutOfRange']].groupby('Group'):
            start = grp['DateTime'].iloc[0]
            end = grp['DateTime'].iloc[-1]
            dur = (end - start).total_seconds()/60
            events.append({'Start': start, 'End': end, 'Duration(min)': dur})

        # Display
        st.subheader(file.name)
        st.write(f"**Type:** {loc}")
        st.write(pd.DataFrame(events))
    st.success("Done processing all files!")
