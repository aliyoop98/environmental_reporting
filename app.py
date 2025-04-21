import streamlit as st
import pandas as pd
import io
import calendar

# Page configuration
st.set_page_config(page_title="Environmental Reporting", layout="wide")
st.title("Environmental Monitoring - Upload, Filter, Rename, Select Date & Flag OOR")

# 1. Upload CSV files
uploaded_files = st.file_uploader(
    label="Upload CSV files",
    type=["csv"],
    accept_multiple_files=True
)
if not uploaded_files:
    st.info("Please upload one or more CSV files to begin.")
    st.stop()

# 2. Process each uploaded file into cleaned DataFrame and assign default ranges
dfs = {}
ranges = {}
for uploaded in uploaded_files:
    name = uploaded.name
    raw = uploaded.read()

    # Strip metadata: header starts at last line with 'CH1' or 'P1'
    lines = raw.decode('utf-8', errors='ignore').splitlines(True)
    try:
        header_idx = max(i for i, line in enumerate(lines)
                         if 'ch1' in line.lower() or 'p1' in line.lower())
    except ValueError:
        st.error(f"No header row found in {name}")
        continue
    csv_text = ''.join(lines[header_idx:])

    # Load CSV with C-engine, skip bad lines
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

    # Clean column names
    df.columns = [c.strip() for c in df.columns]

    # Determine file type
    lname = name.lower()
    has_fridge = 'fridge' in lname
    has_freezer = 'freezer' in lname
    is_combo = has_fridge and has_freezer
    is_fridge_freezer = has_fridge ^ has_freezer

    # Map raw substrings to roles
    if is_combo:
        roles = {'Fridge Temp': 'P1', 'Freezer Temp': 'P2'}
    elif is_fridge_freezer:
        roles = {'Temperature': 'P1'}
    else:
        roles = {'Humidity': 'CH3', 'Temperature': 'CH4'}
    roles.update({'Date': 'Date', 'Time': 'Time'})

    # Match and rename
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
    # Parse Date
    df_clean['Date'] = pd.to_datetime(
        df_clean['Date'], infer_datetime_format=True, errors='coerce'
    )
    # Build DateTime
    df_clean['DateTime'] = pd.to_datetime(
        df_clean['Date'].dt.strftime('%Y-%m-%d ') + df_clean['Time'].astype(str),
        errors='coerce'
    )

    dfs[name] = df_clean

    # Assign default ranges
    if is_combo:
        rng = {'Fridge Temp': (2, 8), 'Freezer Temp': (-35, -5)}
    elif is_fridge_freezer:
        rng = {'Temperature': (2, 8) if has_fridge else (-35, -5)}
    else:
        rng = {'Temperature': (15, 28) if 'olympus' in lname else (15, 25),
               'Humidity': (0, 60)}
    ranges[name] = rng

    st.success(f"Loaded {name}")
    st.write(f"**Default Ranges for {name}:** {rng}")

if not dfs:
    st.stop()

# 3. Date selection UI
years = sorted({dt.year for df in dfs.values() for dt in df['Date'].dropna()})
months = sorted({dt.month for df in dfs.values() for dt in df['Date'].dropna()})
col1, col2 = st.columns(2)
with col1:
    year = st.selectbox('Select Year', years)
with col2:
    month = st.selectbox('Select Month', months, format_func=lambda x: calendar.month_name[x])

# 4. Display & OOR flagging
st.subheader(f"Out-of-Range Events for {calendar.month_name[month]} {year}")
for name, df in dfs.items():
    st.header(name)
    df_sel = df[(df['Date'].dt.year == year) & (df['Date'].dt.month == month)]
    if df_sel.empty:
        st.write("No data for this period.")
        continue

    # Determine OOR
    rng = ranges[name]
    def check_oor(row):
        for col, (mn, mx) in rng.items():
            if pd.notna(row.get(col)):
                try:
                    val = float(row[col])
                    if val < mn or val > mx:
                        return True
                except:
                    return True
        return False

    df_sel['OutOfRange'] = df_sel.apply(check_oor, axis=1)
    df_sel['GroupID'] = (df_sel['OutOfRange'] != df_sel['OutOfRange'].shift(fill_value=False)).cumsum()

    # Summarize events
    events = []
    for gid, grp in df_sel[df_sel['OutOfRange']].groupby('GroupID'):
        start = grp['DateTime'].iloc[0]
        end = grp['DateTime'].iloc[-1]
        dur = (end - start).total_seconds() / 60
        events.append({'Start': start, 'End': end, 'Duration(min)': dur})

    total_out = sum(e['Duration(min)'] for e in events)
    any_long = any(e['Duration(min)'] > 60 for e in events)
    incident = total_out >= 60 or any_long

    if events:
        st.subheader('Details of Out-of-Range Events')
        st.table(pd.DataFrame(events))
        st.write(f"Total out-of-range minutes: {total_out:.1f}")
        st.write(f"Incident? {'YES' if incident else 'No'}")
    else:
        st.write('No out-of-range events.')
