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

# 4. Preview parsed CSV (strip metadata up to last header line)
st.header("Parsed CSV Preview")
for name, raw in raw_data.items():
    st.subheader(name)
    try:
        # Split into lines preserving line breaks
        lines = raw.decode('utf-8', errors='ignore').splitlines(True)
        # Find the last occurrence of CH1 or P1 in header lines
        header_idx = None
        for i, line in enumerate(lines):
            if 'CH1' in line.upper() or 'P1' in line.upper():
                header_idx = i
        if header_idx is None:
            st.error(f"No header row containing 'CH1' or 'P1' found in {name}.")
            continue
        # Combine from header to end
        parsed_text = ''.join(lines[header_idx:])
        # Load into DataFrame
        try:
            df = pd.read_csv(io.StringIO(parsed_text))
        except Exception:
            df = pd.read_csv(
                io.StringIO(parsed_text),
                engine='python',
                on_bad_lines='skip',
                skip_blank_lines=True
            )
        st.dataframe(df.head(), use_container_width=True)
    except Exception as e:
        st.error(f"Failed to parse {name}: {e}")
