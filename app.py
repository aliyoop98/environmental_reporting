import streamlit as st
import pandas as pd
import io

# Step 1: Upload and Read CSVs with Metadata Stripping
st.set_page_config(page_title="Environmental Reporting", layout="wide")
st.title("Environmental Monitoring - Step 1: Upload and Read CSVs")

# 1. Upload CSV files
uploaded_files = st.file_uploader(
    label="Upload CSV files",
    type=["csv"],
    accept_multiple_files=True
)
if not uploaded_files:
    st.info("Please upload one or more CSV files to begin.")
    st.stop()

# 2. Read raw bytes
raw_data = {file.name: file.read() for file in uploaded_files}

# 3. Preview raw content (first 10 lines)
st.header("Raw File Preview")
for name, raw in raw_data.items():
    st.subheader(name)
    try:
        lines = raw.decode('utf-8', errors='ignore').splitlines()
        st.write("".join([f"{i}: {lines[i]}\n" for i in range(min(10, len(lines)))]))
    except Exception as e:
        st.error(f"Error decoding raw content for {name}: {e}")

# 4. Preview parsed & filtered CSV
st.subheader("Parsed CSV Preview (after stripping metadata and filtering channels)")
for name, raw in raw_data.items():
    st.markdown(f"**{name}**")
    try:
        # Strip metadata: use last CH1 or P1 line
        lines = raw.decode('utf-8', errors='ignore').splitlines(True)
        header_idx = max(i for i, line in enumerate(lines)
                         if 'ch1' in line.lower() or 'p1' in line.lower())
        clean_text = "".join(lines[header_idx:])
        df = pd.read_csv(io.StringIO(clean_text), on_bad_lines='skip', skip_blank_lines=True)

        # Determine required columns based on filename
        lower = name.lower()
        if 'combo' in lower or ('fridge' in lower and 'freezer' in lower):
            req = ['P1', 'P2', 'Date', 'Time']
        elif 'freezer' in lower:
            req = ['P1', 'Date', 'Time']
        elif 'fridge' in lower:
            req = ['P1', 'Date', 'Time']
        elif 'olympus' in lower:
            req = ['CH3', 'CH4', 'Date', 'Time']
        else:
            req = ['CH3', 'CH4', 'Date', 'Time']

        # Normalize column names
        col_map = {}
        for col in df.columns:
            lc = col.lower()
            if 'date' in lc:
                col_map[col] = 'Date'
            elif 'time' in lc:
                col_map[col] = 'Time'
            elif 'p1' in lc and 'p2' not in lc:
                col_map[col] = 'P1'
            elif 'p2' in lc:
                col_map[col] = 'P2'
            elif 'ch3' in lc:
                col_map[col] = 'CH3'
            elif 'ch4' in lc:
                col_map[col] = 'CH4'

        df = df.rename(columns=col_map)
        # Keep only required columns if they exist
        keep = [c for c in req if c in df.columns]
        df = df[keep]

        if df.empty:
            st.warning("No matching columns found.")
        else:
            st.dataframe(df.head(5))
    except Exception as e:
        st.error(f"Error parsing {name}: {e}")
