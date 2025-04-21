import streamlit as st
import pandas as pd
import io

# Page configuration
st.set_page_config(page_title="Environmental Reporting", layout="wide")
st.title("Environmental Monitoring - Step 1: Upload, Read, Filter, and Rename CSVs")

# 1. Upload CSV files
uploaded_files = st.file_uploader(
    label="Upload CSV files",
    type=["csv"],
    accept_multiple_files=True
)
if not uploaded_files:
    st.info("Please upload one or more CSV files to begin.")
    st.stop()

# Process each uploaded file
for uploaded in uploaded_files:
    name = uploaded.name
    raw = uploaded.read()

    st.header(f"Preview: {name}")

    # 2. Strip metadata: find the last line containing 'CH1' or 'P1'
    lines = raw.decode('utf-8', errors='ignore').splitlines(True)
    try:
        header_idx = max(
            i for i, line in enumerate(lines)
            if 'ch1' in line.lower() or 'p1' in line.lower()
        )
    except ValueError:
        st.error(f"No header row found in {name}")
        continue

    csv_text = ''.join(lines[header_idx:])

    # 3. Load into DataFrame using pandas C engine, skipping bad lines
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
    df.columns = [col.strip() for col in df.columns]

    # 4. Determine file type and substrings
    lname = name.lower()
    is_combo = 'combo' in lname
    is_fridge_freezer = 'fridge' in lname or 'freezer' in lname
    is_room = not (is_combo or is_fridge_freezer)

    # 5. Map substrings to column roles
    if is_combo:
        # Combo has both P1 and P2
        roles = {
            'Fridge Temp': 'P1',
            'Freezer Temp': 'P2',
        }
    elif is_fridge_freezer:
        # Simple fridge or freezer
        roles = {
            'Temperature': 'P1',
        }
    else:
        # Rooms and Olympus
        roles = {
            'Humidity': 'CH3',
            'Temperature': 'CH4',
        }
    roles['Date'] = 'Date'
    roles['Time'] = 'Time'

    # 6. Find actual columns and report missing
    actual_to_role = {}
    missing = []
    for role, substr in roles.items():
        matches = [c for c in df.columns if substr.lower() in c.lower()]
        if matches:
            actual_to_role[matches[0]] = role
        else:
            missing.append(substr)
    if missing:
        st.error(f"Missing columns in {name}: {missing}")
        continue

    # 7. Filter & rename
    df_filtered = df[list(actual_to_role.keys())].rename(columns=actual_to_role)

    # 8. Display
    if is_combo:
        # Split into two tables
        st.subheader("Combo Fridge/Freezer Data")
        # Fridge
        st.markdown("**Fridge Temperature**")
        st.dataframe(df_filtered[['Fridge Temp', 'Date', 'Time']], use_container_width=True)
        # Freezer
        st.markdown("**Freezer Temperature**")
        st.dataframe(df_filtered[['Freezer Temp', 'Date', 'Time']], use_container_width=True)
    else:
        st.subheader("Filtered & Renamed Data (first 5 rows)")
        st.dataframe(df_filtered.head(), use_container_width=True)

    st.success(f"Finished processing {name}")
