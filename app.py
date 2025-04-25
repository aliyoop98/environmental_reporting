import streamlit as st
import pandas as pd
import altair as alt
import io
from pandas.tseries.offsets import MonthEnd

st.set_page_config(page_title="Environmental Reporting", layout="wide")
st.title("Environmental Monitoring Dashboard")

# --- SIDEBAR: UPLOADS & CONTROLS ---
st.sidebar.header("Data Upload & Settings")

# 1) Probe CSVs
probe_files = st.sidebar.file_uploader(
    "Upload Probe CSV files",
    type="csv",
    accept_multiple_files=True,
    key="probe_uploader"
)

# 2) Tempstick CSVs (optional)
ts_files = st.sidebar.file_uploader(
    "Upload Tempstick CSV files (optional)",
    type="csv",
    accept_multiple_files=True,
    key="ts_uploader"
)

# Parse Tempsticks
tempdfs = {}
if ts_files:
    for ts in ts_files:
        raw = ts.getvalue().decode(errors="ignore").splitlines()
        hdr = max(i for i, L in enumerate(raw) if "timestamp" in L.lower())
        content = "\n".join(raw[hdr:])
        df_ts = pd.read_csv(io.StringIO(content), on_bad_lines="skip", skip_blank_lines=True)
        df_ts.columns = [c.strip() for c in df_ts.columns]
        colmap = {
            **{c: "DateTime"  for c in df_ts.columns if "timestamp" in c.lower()},
            **{c: "Temperature" for c in df_ts.columns if "temperature" in c.lower()},
            **{c: "Humidity"    for c in df_ts.columns if "humidity" in c.lower()},
        }
        df_ts = df_ts.rename(columns=colmap)
        keep = [c for c in ("DateTime","Temperature","Humidity") if c in df_ts.columns]
        df_ts = df_ts[keep].copy()
        df_ts["DateTime"] = pd.to_datetime(df_ts["DateTime"], errors="coerce")
        df_ts["Temperature"] = pd.to_numeric(df_ts["Temperature"], errors="coerce")
        if "Humidity" in df_ts:
            df_ts["Humidity"] = pd.to_numeric(df_ts["Humidity"], errors="coerce")
        tempdfs[ts.name] = df_ts

if tempdfs:
    ts_choice = st.sidebar.selectbox("Select Tempstick to overlay", list(tempdfs.keys()), key="ts_choice")
    ts_df = tempdfs[ts_choice]
else:
    ts_df = None

# --- PARSE PROBE CSVs ---
parsed_probes = {}
for f in probe_files or []:
    raw = f.getvalue().decode(errors="ignore").splitlines()
    hdr = max(i for i, L in enumerate(raw) if "ch1" in L.lower() or "p1" in L.lower())
    content = "\n".join(raw[hdr:])
    df = pd.read_csv(io.StringIO(content), on_bad_lines="skip", skip_blank_lines=True)
    df.columns = [c.strip() for c in df.columns]
    colmap = {}
    for c in df.columns:
        lc = c.lower()
        if "date" in lc:       colmap[c] = "Date"
        elif "time" in lc:     colmap[c] = "Time"
        elif "p1" in lc:       colmap[c] = "P1"
        elif "p2" in lc:       colmap[c] = "P2"
        elif "ch3" in lc:      colmap[c] = "CH3"
        elif "ch4" in lc:      colmap[c] = "CH4"
    df = df.rename(columns=colmap)

    name = f.name.lower()
    is_fridge = "fridge" in name and "freezer" not in name
    is_freezer= "freezer" in name and "fridge" not in name
    is_combo  = "fridge" in name and "freezer" in name
    is_olympus= "olympus" in name

    if is_combo:
        df = df[[c for c in ("P1","P2","Date","Time") if c in df.columns]]
        df = df.rename(columns={"P1":"Fridge Temp","P2":"Freezer Temp"})
    elif is_fridge or is_freezer:
        df = df[[c for c in ("P1","Date","Time") if c in df.columns]]
        df = df.rename(columns={"P1":"Temperature"})
    else:
        hum = "CH3" if "CH3" in df.columns else None
        tmp = "CH4" if "CH4" in df.columns else None
        if not tmp and "CH2" in df.columns: tmp = "CH2"
        keep = [c for c in (hum,tmp,"Date","Time") if c]
        df = df[keep]
        df = df.rename(columns={hum:"Humidity",tmp:"Temperature"})

    df["Date"]     = pd.to_datetime(df["Date"], errors="coerce")
    df["DateTime"] = pd.to_datetime(df["Date"].dt.strftime("%Y-%m-%d")+" "+df["Time"], errors="coerce")
    parsed_probes[f.name] = df

if not parsed_probes:
    st.error("Upload at least one probe CSV.")
    st.stop()

# --- YEAR & MONTH SELECTORS ---
all_dt = pd.concat([df["DateTime"] for df in parsed_probes.values()])
years = sorted(all_dt.dt.year.dropna().unique().astype(int).tolist())
year = st.sidebar.selectbox("Year", years, index=len(years)-1)
months = list(range(1,13))
default_month = int(all_dt.dt.month.max()) if year==years[-1] else 1
month = st.sidebar.selectbox("Month", months, index=months.index(default_month))

# --- THRESHOLD RANGES ---
ranges = {}
for name, df in parsed_probes.items():
    L = name.lower()
    if "fridge" in L and "freezer" not in L:
        ranges[name] = {"Temperature":(2.0,8.0)}
    elif "freezer" in L and "fridge" not in L:
        ranges[name] = {"Temperature":(-35.0,-5.0)}
    elif "fridge" in L and "freezer" in L:
        ranges[name] = {"Fridge Temp":(2.0,8.0),"Freezer Temp":(-35.0,-5.0)}
    elif "olympus" in L:
        ranges[name] = {"Temperature":(15.0,28.0),"Humidity":(0.0,60.0)}
    else:
        ranges[name] = {"Temperature":(15.0,25.0),"Humidity":(0.0,60.0)}

# --- MAIN LOOP: EVENTS & CHARTS ---
for name, df in parsed_probes.items():
    with st.expander(name, expanded=True):
        # Metadata inputs
        title    = st.text_input("Chart Title", value=f"{name} {pd.Timestamp(year,month,1).strftime('%B %Y')}", key=f"{name}_title")
        materials= st.text_area("Materials", key=f"{name}_materials")
        pid      = st.text_input("Probe ID", key=f"{name}_probe")
        eid      = st.text_input("Equipment ID", key=f"{name}_equip")

        # filter by month
        sel = df[(df["DateTime"].dt.year==year)&(df["DateTime"].dt.month==month)].reset_index(drop=True)
        if sel.empty:
            st.warning(f"No data for {year}-{month:02d}")
            continue

        # OUT-OF-RANGE DETECTION
        low_high = ranges[name]
        def flag(r):
            v = pd.to_numeric(r["Value"], errors="coerce") if False else None
            return False

        # Prepare numeric columns for each channel separately
        out_events = {}
        for ch,(low,high) in low_high.items():
            sel_ch = sel[["DateTime",ch]].rename(columns={ch:"Value"}).copy()
            sel_ch["Value"] = pd.to_numeric(sel_ch["Value"],errors="coerce")
            sel_ch["OOR"] = sel_ch["Value"].apply(lambda v: pd.notna(v) and (v<low or v>high))
            # group contiguous
            sel_ch["GroupID"] = (sel_ch["OOR"]!=sel_ch["OOR"].shift(1,False)).cumsum()
            evs=[]
            for gid,grp in sel_ch[sel_ch["OOR"]].groupby("GroupID"):
                start = grp["DateTime"].iloc[0]
                idx_end = grp.index[-1]
                if idx_end+1<len(sel_ch):
                    end = sel_ch.at[idx_end+1,"DateTime"]
                else:
                    end = grp["DateTime"].iloc[-1]
                dur = (end-start).total_seconds()/60.0
                evs.append({"GroupID":gid,"Start":start,"End":end,"Duration (min)":dur})
            ev_df = pd.DataFrame(evs)
            total = ev_df["Duration (min)"].sum() if not ev_df.empty else 0.0
            incident = total>=60 or any(ev_df["Duration (min)"]>=60)
            out_events[ch] = (ev_df,total,incident)

            st.subheader(f"Out-of-Range Events: {ch}")
            st.table(ev_df)
            st.markdown(f"**Total OOR minutes:** {total:.1f} — **Incident:** {'YES' if incident else 'No'}")

        # PLOTTING
        for ch,(low,high) in low_high.items():
            # combine probe + tempstick
            probe = sel[["DateTime",ch]].rename(columns={ch:"Value"}).assign(Source="Probe")
            if ts_df is not None and ch in ts_df.columns:
                ts_sel = ts_df[(ts_df["DateTime"].dt.year==year)&(ts_df["DateTime"].dt.month==month)]
                ts_sel = ts_sel[["DateTime",ch]].rename(columns={ch:"Value"}).assign(Source="Tempstick")
                chart_df = pd.concat([probe,ts_sel],ignore_index=True)
            else:
                chart_df = probe

            chart_df["Value"] = pd.to_numeric(chart_df["Value"],errors="coerce")
            chart_df = chart_df.dropna(subset=["Value"])

            dmin = chart_df["Value"].min()
            dmax = chart_df["Value"].max()
            domain_min = min(dmin, low) if pd.notna(dmin) else low
            domain_max = max(dmax, high) if pd.notna(dmax) else high

            base = alt.Chart(chart_df).encode(
                x=alt.X("DateTime:T", title="Date/Time",
                        scale=alt.Scale(domain=[
                            pd.Timestamp(year,month,1),
                            pd.Timestamp(year,month,1)+MonthEnd(0)
                        ])),
                y=alt.Y("Value:Q", title=f"{ch} ({'°C' if 'Temp' in ch else '%RH'})",
                        scale=alt.Scale(domain=[domain_min,domain_max]))
            )
            line = base.mark_line().encode(color=alt.Color("Source:N",title="Series"))
            rules = alt.Chart(pd.DataFrame({"y":[low,high]})).mark_rule(strokeDash=[5,5],color="red").encode(y="y:Q")

            full_title = f"{title} – {ch} | Materials: {materials or 'None'} | Probe: {pid or '–'} | Equipment: {eid or '–'}"
            chart = (line + rules).properties(title=full_title)
            st.altair_chart(chart,use_container_width=True)
