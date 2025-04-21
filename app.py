import streamlit as st
import pandas as pd
import io

# Page configuration
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

# 2. Read raw content into memory
raw_data = {file.name: file.read() for file in uploaded_files}

# 3. Preview raw CSV content (first 10 lines)
st.subheader("Raw CSV Content (first 10 lines)")
for name, raw in raw_data.items():
    st.markdown(f"**{name}**")
    text = raw.decode('utf-8', errors='ignore')
    lines = text.splitlines()
    # Display first 10 lines of raw text
    st.text("\n".join(lines[:10]))

# 4. Preview parsed CSV (after stripping metadata)
st.subheader("Parsed CSV Preview (after stripping metadata)")
for name, raw in raw_data.items():
    st.markdown(f"**{name}**")
    try:
        # Strip metadata: find first line with a comma as header start
        lines = raw.decode('utf-8', errors='ignore').splitlines(True)
        header_idx = next(
            (i for i, line in enumerate(lines) if ',' in line),
            0
        )
        csv_text = ''.join(lines[header_idx:])

        # Parse CSV into DataFrame
        df = pd.read_csv(
            io.StringIO(csv_text),
            on_bad_lines='skip',
            skip_blank_lines=True
        )
        st.dataframe(df.head(), use_container_width=True)
    except Exception as e:
        st.error(f"Error parsing {name}: {e}")

# Next: Step 2 will handle column normalization and channel filtering"
