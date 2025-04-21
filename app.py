import streamlit as st
import pandas as pd
import io

# Page configuration
st.set_page_config(page_title="Environmental Reporting", layout="wide")
st.title("Environmental Monitoring - Step 1: Upload and Read CSVs")

# Step 1: Upload CSV files
uploaded_files = st.file_uploader(
    label="Upload CSV files",
    type=["csv"],
    accept_multiple_files=True
)
if not uploaded_files:
    st.info("Please upload one or more CSV files to begin.")
    st.stop()

# Read raw content into memory
raw_data = {file.name: file.read() for file in uploaded_files}

# Display raw CSV previews
st.subheader("Raw CSV Preview (first 5 rows)")
for name, raw in raw_data.items():
    st.markdown(f"**{name}**")
    try:
        # Strip metadata: find first line with a comma as header start
        lines = raw.decode('utf-8', errors='ignore').splitlines(True)
        header_idx = next((i for i, line in enumerate(lines) if ',' in line), 0)
        csv_text = ''.join(lines[header_idx:])

        # Parse full CSV into DataFrame
        df = pd.read_csv(
            io.StringIO(csv_text),
            on_bad_lines='skip',
            skip_blank_lines=True
        )
        # Show first 5 rows
        st.dataframe(df.head(), use_container_width=True)
    except Exception as e:
        st.error(f"Error reading {name}: {e}")

# Next: In Step 2 we'll normalize columns and filter for the channels of interest
