import streamlit as st
import pandas as pd
import io

# Page config
st.set_page_config(page_title="Environmental Reporting", layout="wide")
st.title("Environmental Monitoring Out‑of‑Range Detector")

# 1. Upload CSV files
uploaded = st.file_uploader("Upload CSV files", type=["csv"], accept_multiple_files=True)
if not uploaded:
    st.info("Please upload one or more CSV files to begin.")
    st.stop()

# Read raw content into memory to allow multiple reads
raw_data = {}
for file in uploaded:
    raw_data[file.name] = file.read()

# 2. Preview stripped CSVs for sanity check
st.subheader("Preview of Uploaded Files (first 5 data rows after stripping metadata)")
def strip_metadata(content, header_keys=["CH1","P1","Temp","Humidity"]):
    lines = content.decode('utf-8', errors='ignore').splitlines(True)
    header_idx = None
    for i, line in enumerate(lines):
        low = line.lower()
        if ',' in line and any(k.lower() in low for k in header_keys):
            header_idx = i
            break
    if header_idx is None:
        header_idx = 0
    return ''.join(lines[header_idx:])

for name, raw in raw_data.items():
    text = strip_metadata(raw)
    try:
        df_preview = pd.read_csv(io.StringIO(text), on_bad_lines='skip', skip_blank_lines=True, nrows=5)
        st.markdown(f"**{name}**")
        st.dataframe(df_preview)
    except Exception as e:
        st.warning(f"Could not preview {name}: {e}")

# 3. Default ranges per location type
def default_ranges(loc_type):
    return {
        'Fridge':  {'temp': (2, 8),     'hum': None},
        'Freezer': {'temp': (-35, -5),  'hum': None},
        'Combo':   {'temp': (-35, 8),   'hum': None},
        'Room':    {'temp': (15, 25),   'hum': (0, 60)},
        'Olympus': {'temp': None,       'hum': (0, 60)}
    }.get(loc_type, {'temp': (15,25), 'hum': (0,60)})

# 4. Collect settings interactively
settings = {}
for name in raw_data:
    with st.sidebar.expander(name):
        st.write(f"Configure: **{name}**")
        loc = st.selectbox("Location Type", ['Fridge','Freezer','Combo','Room','Olympus'], key=name+"_loc")
        rng = default_ranges(loc)
        # temperature inputs
        if rng['temp'] is not None:
            tmin, tmax = rng['temp']
            tmin = st.number_input("Temp Min", value=tmin, key=name+"_tmin")
            tmax = st.number_input("Temp Max", value=tmax, key=name+"_tmax")
            temp_range = (tmin, tmax)
        else:
            temp_range = None
        # humidity inputs
        if rng['hum'] is not None:
            hmin, hmax = rng['hum']
            hmin = st.number_input("Hum Min", value=hmin, key=name+"_hmin")
            hmax = st.number_input("Hum Max", value=hmax, key=name+"_hmax")
            hum_range = (hmin, hmax)
        else:
            hum_range = None
        settings[name] = {'loc': loc, 'temp': temp_range, 'hum': hum_range}

# 5. Process data on button click
if st.button("Process All Files"):
    for name in raw_data:
        raw = raw_data[name]
        text = strip_metadata(raw)
        # parse CSV with C engine, skip bad lines
        try:
            df = pd.read_csv(io.StringIO(text), on_bad_lines='skip', skip_blank_lines=True)
        except Exception as e:
            st.error(f"Error parsing {name}: {e}")
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

        # Validate required columns
        if 'Date' not in df.columns or 'Time' not in df.columns:
            st.error(f"Missing Date or Time columns in {name}")
            continue

        # Parse unified DateTime
        df['DateTime'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'].astype(str), errors='raise')

        # Fetch configuration
        cfg = settings[name]
        loc = cfg['loc']
        temp_range = None if loc=='Olympus' else cfg['temp']
        hum_range = cfg['hum']

        # Flag out-of-range
        def is_oor(r):
            if temp_range and pd.notna(r.get('Temp', None)):
                t = float(r['Temp'])
                if t < temp_range[0] or t > temp_range[1]:
                    return True
            if hum_range and pd.notna(r.get('Humidity', None)):
                h = float(str(r['Humidity']).lstrip('+'))
                if h < hum_range[0] or h > hum_range[1]:
                    return True
            return False
        df['OutOfRange'] = df.apply(is_oor, axis=1)

        # Group contiguous events
        df['Group'] = (df['OutOfRange'] != df['OutOfRange'].shift(fill_value=False)).cumsum()
        events = []
        for gid, grp in df[df['OutOfRange']].groupby('Group'):
            start = grp['DateTime'].iloc[0]
            end = grp['DateTime'].iloc[-1]
            dur = (end - start).total_seconds() / 60
            events.append({'Start': start, 'End': end, 'Duration(min)': dur})

        # Display results
        st.subheader(name)
        st.markdown(f"**Type:** {loc}")
        if events:
            st.dataframe(pd.DataFrame(events), use_container_width=True)
        else:
            st.write("No out‑of‑range events detected.")
    st.success("Done processing all files!")
