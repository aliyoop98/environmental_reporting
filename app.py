import copy
from pathlib import Path
from typing import Optional

import streamlit as st
import pandas as pd
import io
import calendar
import altair as alt
from datetime import datetime, timedelta


def _read_csv_flexible(text: str) -> Optional[pd.DataFrame]:
    """Read CSV content supporting multiple delimiters.

    Tries automatic delimiter detection first (e.g., tab separated files) and
    falls back to the default comma behaviour if necessary.
    """

    for kwargs in ({"sep": None, "engine": "python"}, {}):
        try:
            df = pd.read_csv(io.StringIO(text), on_bad_lines="skip", **kwargs)
        except Exception:
            continue
        return df.rename(columns=lambda c: c.strip() if isinstance(c, str) else c)
    return None

st.set_page_config(page_title="Environmental Reporting", layout="wide", page_icon="⛅")
st.markdown(
    """
    <style>
        .stApp { background-color: white; }
        [data-testid="stSidebar"] { background-color: white; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.sidebar.header("🌦️ Data Upload & Configuration")
primary_probe_files = st.sidebar.file_uploader(
    "Upload Primary Probe CSV files",
    accept_multiple_files=True,
    key="primary_probe_uploader"
)

if not primary_probe_files:
    st.sidebar.info("Upload Primary Probe CSV files to begin.")
    st.stop()


def _match_tempstick_channel(ts_df: pd.DataFrame, channel: str) -> Optional[str]:
    """Return the column in a tempstick dataframe that matches a probe channel."""

    if channel in ts_df.columns:
        return channel
    ch_lower = channel.lower()
    if 'temp' in ch_lower and 'Temperature' in ts_df.columns:
        return 'Temperature'
    if 'hum' in ch_lower and 'Humidity' in ts_df.columns:
        return 'Humidity'
    return None


def _parse_probe_files(files):
    dfs = {}
    ranges = {}
    if not files:
        return dfs, ranges
    for f in files:
        f.seek(0)
        name = f.name
        raw = f.read().decode('utf-8', errors='ignore').splitlines(True)
        try:
            idx = max(i for i, line in enumerate(raw) if 'ch1' in line.lower() or 'p1' in line.lower())
        except ValueError:
            continue
        content = ''.join(raw[idx:])
        df = _read_csv_flexible(content)
        if df is None:
            continue
        lname = name.lower()
        has_fridge = any(term in lname for term in ('fridge', 'refrigerator'))
        has_freezer = 'freezer' in lname
        if has_fridge and has_freezer:
            mapping = {'Fridge Temp': 'P1', 'Freezer Temp': 'P2'}
            ranges[name] = {'Fridge Temp': (2, 8), 'Freezer Temp': (-35, -5)}
        elif has_fridge:
            mapping = {'Temperature': 'P1'}
            ranges[name] = {'Temperature': (2, 8)}
        elif has_freezer:
            mapping = {'Temperature': 'P1'}
            ranges[name] = {'Temperature': (-35, -5)}
        else:
            cols_lower = [c.lower() for c in df.columns]
            if any('ch4' in c for c in cols_lower):
                mapping = {'Humidity': 'CH3', 'Temperature': 'CH4'}
            else:
                mapping = {'Humidity': 'CH1', 'Temperature': 'CH2'}
            is_olympus = 'olympus' in lname
            temp_range = (15, 28) if is_olympus else (15, 25)
            humidity_range = (0, 80) if is_olympus else (0, 60)
            ranges[name] = {'Temperature': temp_range, 'Humidity': humidity_range}
        mapping.update({'Date': 'Date', 'Time': 'Time'})
        col_map = {}
        for new, key in mapping.items():
            col = next((c for c in df.columns if key.lower() in c.lower()), None)
            if col:
                col_map[col] = new
        df = df[list(col_map)].rename(columns=col_map)
        df['Date'] = pd.to_datetime(df['Date'], infer_datetime_format=True, errors='coerce')
        df['DateTime'] = pd.to_datetime(
            df['Date'].dt.strftime('%Y-%m-%d ') + df['Time'].astype(str),
            errors='coerce'
        )
        for c in df.columns.difference(['Date', 'Time', 'DateTime']):
            df[c] = pd.to_numeric(df[c].astype(str).str.replace('+',''), errors='coerce')
        dfs[name] = df.reset_index(drop=True)
    return dfs, ranges


def _parse_tempstick_files(files):
    tempdfs = {}
    if not files:
        return tempdfs
    for f in files:
        f.seek(0)
        raw = f.read().decode('utf-8', errors='ignore').splitlines(True)
        try:
            idx = max(i for i, line in enumerate(raw) if 'timestamp' in line.lower())
        except ValueError:
            continue
        csv_text = ''.join(raw[idx:])
        df_ts = _read_csv_flexible(csv_text)
        if df_ts is None:
            continue
        ts_col = next((c for c in df_ts.columns if 'timestamp' in c.lower()), None)
        if not ts_col:
            continue
        df_ts = df_ts.rename(columns={ts_col: 'DateTime'})
        df_ts['DateTime'] = pd.to_datetime(
            df_ts['DateTime'], infer_datetime_format=True, errors='coerce'
        )
        temp_col = next((c for c in df_ts.columns if 'temp' in c.lower()), None)
        if temp_col:
            df_ts['Temperature'] = pd.to_numeric(
                df_ts[temp_col].astype(str).str.replace('+', ''), errors='coerce'
            )
        hum_col = next((c for c in df_ts.columns if 'hum' in c.lower()), None)
        if hum_col:
            df_ts['Humidity'] = pd.to_numeric(
                df_ts[hum_col].astype(str).str.replace('+', ''), errors='coerce'
            )
        cols = ['DateTime']
        if 'Temperature' in df_ts.columns:
            cols.append('Temperature')
        if 'Humidity' in df_ts.columns:
            cols.append('Humidity')
        tempdfs[f.name] = df_ts[cols]
    return tempdfs


primary_dfs, primary_ranges = _parse_probe_files(primary_probe_files)
if not primary_dfs:
    st.sidebar.error("No valid primary probe data found.")
    st.stop()

# Year & Month selection
years = sorted({dt.year for df in primary_dfs.values() for dt in df['Date'].dropna()})
months = sorted({dt.month for df in primary_dfs.values() for dt in df['Date'].dropna()})
year = st.sidebar.selectbox("Year", years)
month = st.sidebar.selectbox("Month", months, format_func=lambda m: calendar.month_name[m])

for name in primary_dfs:
    st.session_state.setdefault(f"chart_title_{name}", Path(name).stem)
    st.session_state.setdefault(f"primary_label_{name}", "Probe")
    st.session_state.setdefault(f"materials_{name}", "")
    st.session_state.setdefault(f"probe_{name}", "")
    st.session_state.setdefault(f"equip_{name}", "")

st.session_state.setdefault("generated_results", {})
st.session_state.setdefault("saved_results", [])


def _build_outputs(
    name,
    chart_title,
    df,
    year,
    month,
    channels,
    comparison_probes,
    comparison_options,
    primary_dfs,
    secondary_dfs,
    probe_labels,
    secondary_labels,
    tempdfs,
    tempstick_labels,
    selected_tempsticks,
    primary_ranges,
    materials,
    probe_id,
    equip_id,
):
    base_title = chart_title or Path(name).stem
    period_label = f"{calendar.month_name[month]} {year}"
    title_with_period = f"{base_title} - {period_label}"
    result = {
        "name": name,
        "chart_title": title_with_period,
        "year": year,
        "month": month,
        "materials": materials,
        "probe_id": probe_id,
        "equip_id": equip_id,
        "channels": channels,
        "channel_results": {},
        "warnings": [],
    }

    sel = df[(df['Date'].dt.year == year) & (df['Date'].dt.month == month)].sort_values('DateTime').reset_index(drop=True)
    if sel.empty:
        result["warnings"].append("No data for selected period.")
        return result

    start_date = datetime(year, month, 1)
    end_date = start_date + timedelta(days=calendar.monthrange(year, month)[1] - 1)

    for ch in channels:
        channel_info = {
            "chart": None,
            "warning": None,
            "oor_table": None,
            "total_minutes": 0.0,
            "incident": False,
        }

        if ch not in sel.columns:
            channel_info["warning"] = f"Channel {ch} not found in data."
            result["channel_results"][ch] = channel_info
            continue

        series_frames = []
        probe_legends = []
        tempstick_legends = []

        probe_label = probe_labels.get(name, "Probe")
        probe_sub = sel[['DateTime', ch]].rename(columns={ch: 'Value'})
        probe_sub['Legend'] = probe_label
        series_frames.append(probe_sub)
        probe_legends.append(probe_label)

        for opt_key in comparison_probes:
            source, comp_name = comparison_options[opt_key]
            if source == "primary":
                comp_df = primary_dfs.get(comp_name)
                label_dict = probe_labels
            else:
                comp_df = secondary_dfs.get(comp_name)
                label_dict = secondary_labels
            if comp_df is None:
                continue
            comp_sel = comp_df[
                (comp_df['Date'].dt.year == year)
                & (comp_df['Date'].dt.month == month)
            ].sort_values('DateTime').reset_index(drop=True)
            if ch not in comp_sel.columns:
                continue
            comp_label = label_dict.get(comp_name, comp_name)
            comp_sub = comp_sel[['DateTime', ch]].rename(columns={ch: 'Value'})
            comp_sub['Legend'] = comp_label
            series_frames.append(comp_sub)
            probe_legends.append(comp_label)

        for ts_name in selected_tempsticks:
            ts_df = tempdfs.get(ts_name)
            if ts_df is None:
                continue
            ts_filtered = ts_df[
                (ts_df['DateTime'].dt.year == year)
                & (ts_df['DateTime'].dt.month == month)
            ].copy()
            ts_column = _match_tempstick_channel(ts_filtered, ch)
            if not ts_column:
                continue
            ts_label = tempstick_labels.get(ts_name, ts_name)
            ts_sub = ts_filtered[['DateTime', ts_column]].rename(columns={ts_column: 'Value'})
            ts_sub['Legend'] = ts_label
            series_frames.append(ts_sub)
            tempstick_legends.append(ts_label)

        if not series_frames:
            channel_info["warning"] = f"No data available to chart for channel {ch}."
            result["channel_results"][ch] = channel_info
            continue

        df_chart = pd.concat(series_frames, ignore_index=True).dropna(subset=['Value'])
        if df_chart.empty:
            channel_info["warning"] = f"No valid values available for channel {ch}."
            result["channel_results"][ch] = channel_info
            continue

        data_min = df_chart['Value'].min()
        data_max = df_chart['Value'].max()
        lo, hi = primary_ranges[name].get(ch, (data_min, data_max))
        raw_min, raw_max = min(data_min, lo), max(data_max, hi)
        span = (raw_max - raw_min) or 1
        pad = span * 0.1
        ymin, ymax = raw_min - pad, raw_max + pad
        probe_palette = ['#1f77b4', '#9467bd', '#17becf', '#7f7f7f']
        tempstick_palette = ['#ff7f0e', '#bcbd22', '#8c564b', '#e377c2']
        color_map = {}
        for idx, legend in enumerate(probe_legends):
            color_map[legend] = probe_palette[idx % len(probe_palette)]
        for idx, legend in enumerate(tempstick_legends):
            color_map[legend] = tempstick_palette[idx % len(tempstick_palette)]
        color_map.update({'Lower Limit': '#2ca02c', 'Upper Limit': '#d62728'})
        legend_entries = probe_legends + tempstick_legends
        has_limits = ch in primary_ranges[name]
        if has_limits:
            legend_entries.extend(['Lower Limit', 'Upper Limit'])
        color_domain = [entry for entry in color_map if entry in legend_entries]
        color_scale = alt.Scale(domain=color_domain, range=[color_map[e] for e in color_domain])
        base = alt.Chart(df_chart).encode(
            x=alt.X('DateTime:T', title='Date/Time', scale=alt.Scale(domain=[start_date, end_date])),
            y=alt.Y('Value:Q', title=f"{ch} ({'°C' if 'Temp' in ch else '%RH'})", scale=alt.Scale(domain=[ymin, ymax], nice=False)),
            color=alt.Color(
                'Legend:N',
                scale=color_scale,
                legend=alt.Legend(title='Probe', labelLimit=0)
            )
        )
        line = base.mark_line()
        layers = [line]
        if has_limits:
            lo, hi = primary_ranges[name][ch]
            limits_df = pd.DataFrame({'y': [lo, hi], 'Legend': ['Lower Limit', 'Upper Limit']})
            layers.append(
                alt.Chart(limits_df)
                .mark_rule(strokeDash=[4, 4])
                .encode(y='y:Q', color=alt.Color('Legend:N', scale=color_scale, legend=None))
            )
        title_lines = [
            f"{title_with_period} - {ch}",
            f"Materials: {materials} | Probe: {probe_id} | Equipment: {equip_id}"
        ]
        chart = (
            alt.layer(*layers)
            .properties(title={"text": title_lines, "anchor": "start"})
            .configure_title(fontSize=14, lineHeight=20, offset=10)
            .configure(background="white", view=alt.ViewConfig(fill="white", stroke=None))
        )
        channel_info["chart"] = chart

        ch_lo, ch_hi = primary_ranges[name].get(ch, (None, None))
        if ch_lo is not None:
            df_ch = sel[['DateTime', ch]].copy()
            df_ch['OOR'] = df_ch[ch].apply(lambda v: pd.notna(v) and (v < ch_lo or v > ch_hi))
            df_ch['Group'] = (df_ch['OOR'] != df_ch['OOR'].shift(fill_value=False)).cumsum()
            events = []
            for _, grp in df_ch.groupby('Group'):
                if not grp['OOR'].iloc[0]:
                    continue
                start = grp['DateTime'].iloc[0]
                last_idx = grp.index[-1]
                if last_idx + 1 < len(sel) and not df_ch.loc[last_idx + 1, 'OOR']:
                    end = sel.loc[last_idx + 1, 'DateTime']
                else:
                    end = grp['DateTime'].iloc[-1]
                duration = max((end - start).total_seconds() / 60, 0)
                events.append({'Start': start, 'End': end, 'Duration(min)': duration})
            if events:
                ev_df = pd.DataFrame(events)
                channel_info['oor_table'] = ev_df
                total = ev_df['Duration(min)'].sum()
                incident = total >= 60 or any(ev_df['Duration(min)'] > 60)
                channel_info['total_minutes'] = float(total)
                channel_info['incident'] = bool(incident)
            else:
                channel_info['oor_table'] = pd.DataFrame(columns=['Start', 'End', 'Duration(min)'])

        result["channel_results"][ch] = channel_info

    return result


tab_titles = [Path(name).stem for name in primary_dfs]
tabs = st.tabs(tab_titles)
for tab, name in zip(tabs, primary_dfs):
    with tab:
        df = primary_dfs[name]
        st.subheader(name)

        st.markdown("### Chart Details")
        chart_title_key = f"chart_title_{name}"
        chart_title = st.text_input("Chart Title", key=chart_title_key)
        materials_key = f"materials_{name}"
        materials = st.text_input("Materials List", key=materials_key)
        probe_id_key = f"probe_{name}"
        probe_id = st.text_input("Probe ID", key=probe_id_key)
        equip_id_key = f"equip_{name}"
        equip_id = st.text_input("Equipment ID", key=equip_id_key)

        st.markdown("### Data Uploads")
        secondary_files = st.file_uploader(
            "Upload Secondary Probe CSV files (optional)",
            accept_multiple_files=True,
            key=f"secondary_{name}"
        )
        secondary_dfs, _ = _parse_probe_files(secondary_files)
        tempstick_files = st.file_uploader(
            "Upload Tempstick CSV files (optional)",
            accept_multiple_files=True,
            key=f"tempstick_{name}"
        )
        tempdfs = _parse_tempstick_files(tempstick_files)

        st.markdown("### Legend Labels")
        primary_label_key = f"primary_label_{name}"
        st.text_input("Legend label for primary data", key=primary_label_key)
        all_primary_labels = {
            fname: (st.session_state.get(f"primary_label_{fname}") or "Probe")
            for fname in primary_dfs
        }

        secondary_labels = {}
        if secondary_dfs:
            st.caption("Secondary probe files")
            for sec_name in secondary_dfs:
                key = f"label_secondary_{name}_{sec_name}"
                st.session_state.setdefault(key, sec_name)
                secondary_labels[sec_name] = st.text_input(
                    f"Label for {sec_name}",
                    key=key
                )

        tempstick_labels = {}
        if tempdfs:
            st.caption("Tempstick files")
            for ts_name in tempdfs:
                key = f"label_tempstick_{name}_{ts_name}"
                st.session_state.setdefault(key, ts_name)
                tempstick_labels[ts_name] = st.text_input(
                    f"Label for {ts_name}",
                    key=key
                )

        selected_tempsticks = (
            st.multiselect(
                "Match Tempsticks",
                options=list(tempdfs.keys()),
                format_func=lambda opt: tempstick_labels.get(opt, opt),
                key=f"ts_{name}"
            )
            if tempdfs
            else []
        )

        comparison_options = {}
        for other_name in primary_dfs:
            if other_name == name:
                continue
            comparison_options[f"primary::{other_name}"] = ("primary", other_name)
        for sec_name in secondary_dfs:
            comparison_options[f"secondary::{sec_name}"] = ("secondary", sec_name)

        def _format_option(opt_key):
            source, fname = comparison_options[opt_key]
            if source == "primary":
                return all_primary_labels.get(fname, fname)
            label = secondary_labels.get(fname, fname)
            return f"{label} (Secondary)"

        format_func = _format_option if comparison_options else None
        comparison_probes = st.multiselect(
            "Additional probe files to overlay",
            options=list(comparison_options.keys()),
            format_func=format_func,
            key=f"compare_{name}"
        )

        channel_keys = list(primary_ranges[name].keys())
        channels = st.multiselect(
            "Channels to plot",
            options=channel_keys,
            default=channel_keys,
            key=f"channels_{name}"
        )

        generate_clicked = st.button(f"Generate {name}", key=f"btn_{name}")
        if generate_clicked:
            chart_title_value = (chart_title or "").strip() or Path(name).stem
            materials_value = materials or ""
            probe_id_value = probe_id or ""
            equip_id_value = equip_id or ""
            result = _build_outputs(
                name,
                chart_title_value,
                df,
                year,
                month,
                channels,
                comparison_probes,
                comparison_options,
                primary_dfs,
                secondary_dfs,
                all_primary_labels,
                secondary_labels,
                tempdfs,
                tempstick_labels,
                selected_tempsticks,
                primary_ranges,
                materials_value,
                probe_id_value,
                equip_id_value,
            )
            st.session_state["generated_results"][name] = result

        current_result = st.session_state["generated_results"].get(name)
        if current_result:
            if current_result["year"] != year or current_result["month"] != month:
                st.info(
                    "Displaying previously generated results. Adjust filters and press Generate to refresh."
                )
            if current_result["warnings"]:
                for msg in current_result["warnings"]:
                    st.warning(msg)

            has_content = False
            for ch in current_result["channels"]:
                ch_result = current_result["channel_results"].get(ch)
                if not ch_result:
                    continue
                if ch_result.get("warning"):
                    st.warning(ch_result["warning"])
                    continue
                if ch_result.get("chart"):
                    st.altair_chart(ch_result["chart"], use_container_width=True)
                    has_content = True
                if ch_result.get("oor_table") is not None:
                    st.markdown(f"### {ch} OOR Events")
                    ev_df = ch_result["oor_table"]
                    if not ev_df.empty:
                        st.table(ev_df)
                        st.write(f"**Total OOR minutes:** {ch_result['total_minutes']:.1f}")
                        st.write(f"**Incident:** {'YES' if ch_result['incident'] else 'No'}")
                    else:
                        st.info("No out-of-range events detected.")
                    has_content = True

            if has_content and st.button("Save results for this session", key=f"save_{name}"):
                st.session_state["saved_results"].append(copy.deepcopy(current_result))
                st.success("Results saved. Scroll down to review saved charts and tables.")

if st.session_state["saved_results"]:
    st.header("Saved Charts & OOR Summaries")
    for idx, saved in enumerate(st.session_state["saved_results"], start=1):
        st.subheader(f"{idx}. {saved['chart_title']}")
        for ch in saved["channels"]:
            saved_ch = saved["channel_results"].get(ch)
            if not saved_ch or saved_ch.get("warning"):
                continue
            st.markdown(f"#### {ch}")
            if saved_ch.get("chart"):
                st.altair_chart(saved_ch["chart"], use_container_width=True)
            if saved_ch.get("oor_table") is not None:
                ev_df = saved_ch["oor_table"]
                if not ev_df.empty:
                    st.table(ev_df)
                    st.write(f"**Total OOR minutes:** {saved_ch['total_minutes']:.1f}")
                    st.write(f"**Incident:** {'YES' if saved_ch['incident'] else 'No'}")
                else:
                    st.info("No out-of-range events detected.")
