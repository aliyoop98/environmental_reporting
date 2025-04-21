import streamlit as st
import pandas as pd
import io

# Page config
st.set_page_config(page_title="Environmental Reporting", layout="wide")
st.title("Environmental Monitoring - Select Channels")

# 1. Upload CSV files
uploaded = st.file_uploader("Upload CSV files", type=["csv"], accept_multiple_files=True)
if not uploaded:
    st.info("Please upload one or more CSV files to begin.")
    st.stop()

# Read raw content
raw_data = {file.name: file.read() for file in uploaded}

# 2. Preview stripped CSVs - only P1, P2, Date, Time
st.subheader("Preview: Only Columns P1, P2, Date, Time")
def strip_metadata(content, keys=["P1","P2","Date","Time"]):
    lines = content.decode('utf-8', errors='ignore').splitlines(True)
    header_idx = None
    for i, line in enumerate(lines):
        low = line.lower()
        if ',' in line and any(k.lower() in low for k in keys):
            header_idx = i
            break
    if header_idx is None:
        header_idx = 0
    return ''.join(lines[header_idx:])

for name, raw in raw_data.items():
    text = strip_metadata(raw)
    try:
        df_preview = pd.read_csv(
            io.StringIO(text),
            on_bad_lines='skip',
            skip_blank_lines=True,
            nrows=5
        )
        # Keep only desired columns
        cols = [c for c in df_preview.columns if c.strip() in ['P1','P2','Date','Time']]
        df_preview = df_preview[cols]
        st.markdown(f"**{name}**")
        st.dataframe(df_preview, use_container_width=True)
    except Exception as e:
        st.warning(f"Could not preview {name}: {e}")

# 3. [Next steps: filtering and processing based on P1/P2]
