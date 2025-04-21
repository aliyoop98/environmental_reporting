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

    # 4. Determine file type and define substring-to-new-name mapping
    lname = name.lower()
    if 'combo' in lname:
        substr_map = {
            'P1': 'Fridge Temp',
            'P2': 'Freezer Temp',
            'Date': 'Date',
            'Time': 'Time'
        }
    elif 'fridge' in lname or 'freezer' in lname:
        substr_map = {
            'P1': 'Temperature',
            'Date': 'Date',
            'Time': 'Time'
        }
    else:
        substr_map = {
            'CH3': 'Humidity',
            'CH4': 'Temperature',
            'Date': 'Date',
            'Time': 'Time'
        }

    # 5. Find actual columns by substring match and rename
    actual_to_new = {}
    missing_substr = []
    for substr, new_name in substr_map.items():
        matches = [c for c in df.columns if substr.lower() in c.lower()]
        if matches:
            actual_to_new[matches[0]] = new_name
        else:
            missing_substr.append(substr)
    if missing_substr:
        st.error(f"Missing columns in {name}: {missing_substr}")
        continue

    # 6. Filter and rename DataFrame
    df_filtered = df[list(actual_to_new.keys())].rename(columns=actual_to_new)

    # 7. Display filtered & renamed data
    st.subheader("Filtered & Renamed Data (first 5 rows)")
    st.dataframe(df_filtered.head(), use_container_width=True)
