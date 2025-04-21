import streamlit as st
import pandas as pd
import io
import calendar

# Page configuration
st.set_page_config(page_title="Environmental Reporting", layout="wide")
st.title("Environmental Monitoring - Step 1: Upload, Filter, Rename, and Assign Ranges & Select Date")

# 1. Upload CSV files
uploaded_files = st.file_uploader(
    label="Upload CSV files",
    type=["csv"],
    accept_multiple_files=True
)
if not uploaded_files:
    st.info("Please upload one or more CSV files to begin.")
    st.stop()

# 2. Process each uploaded file into a cleaned DataFrame and assign default ranges
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

    # Load CSV
    try:
        df = pd.read_csv(
            io.StringIO(csv_text),
            sep=',',                # assume comma-delimited
            on_bad_lines='skip',    # skip malformed rows
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
    is_fridge_freezer = has_fridge ^ has_freezer  # fridge-only or freezer-only

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
    dfs[name] = df_clean

    # Assign default ranges based on asset type
    if is_combo:
        rng = {'Fridge Temp': (2, 8), 'Freezer Temp': (-35, -5)}
    elif is_fridge_freezer:
        if has_fridge:
            rng = {'Temperature': (2, 8)}
        else:
            rng = {'Temperature': (-35, -5)}
    else:
        if 'olympus' in lname:
            rng = {'Temperature': (15, 28), 'Humidity': (0, 60)}
        else:
            rng = {'Temperature': (15, 25), 'Humidity': (0, 60)}
    ranges[name] = rng

    # Display success and ranges
    st.success(f"Loaded and cleaned {name}")
    st.write(f"**Default Ranges for {name}:**")
    for col, (mn, mx) in rng.items():
        st.write(f"- {col}: {mn} to {mx}")

if not dfs:
    st.stop()

# 3. Date selection UI
# Parse Date column to datetime (allow 2- or 4-digit years)
for df in dfs.values():
    df['Date'] = pd.to_datetime(
        df['Date'],
        infer_datetime_format=True,
        errors='coerce'
    )
# Collect year/month options
years = sorted({dt.year for df in dfs.values() for dt in df['Date'].dropna()})
months = sorted({dt.month for df in dfs.values() for dt in df['Date'].dropna()})

col1, col2 = st.columns(2)
with col1:
    year = st.selectbox('Select Year', years)
with col2:
    month = st.selectbox('Select Month', months, format_func=lambda x: calendar.month_name[x])

# 4. Display filtered data by selection
st.subheader(f"Data for {calendar.month_name[month]} {year}")
for name, df in dfs.items():
    st.header(name)
    df_sel = df[(df['Date'].dt.year == year) & (df['Date'].dt.month == month)]
    if df_sel.empty:
        st.write("No data for this period.")
        continue
    if 'Fridge Temp' in df_sel.columns and 'Freezer Temp' in df_sel.columns:
        st.subheader('Fridge Temperature')
        st.dataframe(df_sel[['Fridge Temp', 'Date', 'Time']], use_container_width=True)
        st.subheader('Freezer Temperature')
        st.dataframe(df_sel[['Freezer Temp', 'Date', 'Time']], use_container_width=True)
    else:
        st.dataframe(df_sel, use_container_width=True)
