import streamlit as st
import pandas as pd
import io
import json

# Page configuration
st.set_page_config(
    page_title="Environmental Reporting",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Environmental Monitoring Out-of-Range Detector")

# --- Sidebar: Location configuration ---
st.sidebar.header("Location Configuration")
st.sidebar.markdown(
    """
Paste a JSON mapping of filenames to their location settings.

Example:
```json
{
  "Clinical Fridge 6.csv": {
    "location_type": "Fridge",
    "temp_range": [2, 8],
    "humidity_range": null
  },
  "Olympus Scanner Room 9165.csv": {
    "location_type": "Olympus",
    "temp_range": null,
    "humidity_range": [0, 60]
  }
}
```"""
)
location_info_text = st.sidebar.text_area(
    "Location Info JSON (filename as key)",
    height=200,
    value="{}"
)

# Parse the sidebar JSON
try:
    location_info = json.loads(location_info_text)
except json.JSONDecodeError:
    st.sidebar.error("Invalid JSON. Please fix the mapping.")
    location_info = {}

# --- Main: File upload ---
uploaded_files = st.file_uploader(
    "Upload CSV files",
    type=["csv"],
    accept_multiple_files=True
)

if uploaded_files:
    for file in uploaded_files:
        st.header(file.name)
        try:
            content = file.read()
            # Strip metadata: find header row
            lines = content.decode('utf-8', errors='ignore').splitlines(True)
            header_idx = next(
                i for i, line in enumerate(lines)
                if any(k.lower() in line.lower() for k in ["ch1", "p1", "temp", "hum"])
            )
            clean_text = "".join(lines[header_idx:])

            # Load into DataFrame
            df = pd.read_csv(io.StringIO(clean_text), low_memory=False)
            # Normalize columns
            col_map = {}
            for c in df.columns:
                cl = c.lower()
                if "temp" in cl or "ch4" in cl or "ch1" in cl or "p1" in cl:
                    col_map[c] = "Temp"
                elif "hum" in cl or "ch3" in cl:
                    col_map[c] = "Humidity"
                elif "date" in cl:
                    col_map[c] = "Date"
                elif "time" in cl:
                    col_map[c] = "Time"
            df = df.rename(columns=col_map)
            df = df[["Date", "Time", "Temp", "Humidity"]]

            # Parse DateTime
            df["DateTime"] = pd.to_datetime(
                df["Date"].astype(str) + " " + df["Time"].astype(str),
                errors='raise'
            )

            # Get location ranges
            info = location_info.get(file.name, {})
            loc_type = info.get("location_type", "Room")
            humidity_range = info.get("humidity_range", [0, 60])
            temp_range = None if loc_type == "Olympus" else info.get("temp_range", [15, 25])

            st.write(f"**Location:** {loc_type}")
            st.write(f"Temperature Range: {temp_range}")
            st.write(f"Humidity Range: {humidity_range}")

            # Out-of-range detection
            def is_oor(row):
                if temp_range is not None and pd.notna(row["Temp"]):
                    t = float(row["Temp"])
                    if t < temp_range[0] or t > temp_range[1]:
                        return True
                if humidity_range is not None and pd.notna(row["Humidity"]):
                    h = float(str(row["Humidity"]).lstrip('+'))
                    if h < humidity_range[0] or h > humidity_range[1]:
                        return True
                return False

            df["OutOfRange"] = df.apply(is_oor, axis=1)
            # Group contiguous events
            df["GroupID"] = (df["OutOfRange"] != df["OutOfRange"].shift(fill_value=False)).cumsum()
            events = []
            for gid, grp in df[df["OutOfRange"]].groupby("GroupID"):
                start = grp["DateTime"].iloc[0]
                end = grp["DateTime"].iloc[-1]
                duration = (end - start).total_seconds() / 60
                events.append({"GroupID": gid, "Start": start, "End": end, "Duration(min)": duration})

            # Display events
            if events:
                st.table(events)
            else:
                st.write("No out-of-range events detected.")

        except Exception as e:
            st.error(f"Error processing file: {e}")
else:
    st.info("Upload one or more CSV files to begin.")
