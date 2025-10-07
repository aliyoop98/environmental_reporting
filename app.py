import copy
import hashlib
from pathlib import Path
import sys
from typing import Optional

import streamlit as st
import pandas as pd
import calendar
import altair as alt
from datetime import datetime, timedelta

# Ensure local imports resolve even when the repository root is not on PYTHONPATH.
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from data_processing import _parse_probe_files, _parse_tempstick_files, parse_serial_csv


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


def _state_key(prefix: str, identifier: str) -> str:
    digest = hashlib.sha1(identifier.encode('utf-8')).hexdigest()[:10]
    return f"{prefix}_{digest}"

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
    serial_overlays,
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
        serial_legends = []

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

        for overlay in serial_overlays:
            overlay_df = overlay.get('df')
            if overlay_df is None or ch not in overlay_df.columns:
                continue
            overlay_filtered = overlay_df[
                (overlay_df['DateTime'].dt.year == year)
                & (overlay_df['DateTime'].dt.month == month)
            ].copy()
            if overlay_filtered.empty or ch not in overlay_filtered.columns:
                continue
            overlay_label = overlay.get('label') or 'Serial'
            overlay_sub = overlay_filtered[['DateTime', ch]].rename(columns={ch: 'Value'})
            overlay_sub['Legend'] = overlay_label
            series_frames.append(overlay_sub)
            serial_legends.append(overlay_label)

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
        serial_palette = ['#ff9896', '#98df8a', '#c5b0d5', '#c49c94']
        color_map = {}
        for idx, legend in enumerate(probe_legends):
            color_map[legend] = probe_palette[idx % len(probe_palette)]
        for idx, legend in enumerate(tempstick_legends):
            color_map[legend] = tempstick_palette[idx % len(tempstick_palette)]
        for idx, legend in enumerate(serial_legends):
            color_map[legend] = serial_palette[idx % len(serial_palette)]
        color_map.update({'Lower Limit': '#2ca02c', 'Upper Limit': '#d62728'})
        legend_entries = probe_legends + tempstick_legends + serial_legends
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

        serial_files = st.file_uploader(
            "Upload Consolidated Serial CSV files (optional)",
            accept_multiple_files=True,
            key=f"serial_{name}"
        )
        serial_data = parse_serial_csv(serial_files)
        serial_label_state_keys = {}
        if serial_data:
            st.caption("Serial number datasets")
            for option_key, info in serial_data.items():
                state_key = _state_key(f"serial_label_{name}", str(option_key))
                serial_label_state_keys[option_key] = state_key
                st.session_state.setdefault(state_key, info['default_label'])
                st.text_input(
                    f"Legend label for {info['option_label']}",
                    key=state_key
                )

        def _format_serial_option(opt):
            state_key = serial_label_state_keys.get(opt)
            if state_key:
                value = st.session_state.get(state_key)
                if value:
                    return value
            info = serial_data.get(opt) if serial_data else None
            if info:
                return info['option_label']
            return opt

        serial_selection = (
            st.multiselect(
                "Assign serial overlays",
                options=list(serial_data.keys()),
                format_func=_format_serial_option,
                key=f"serial_select_{name}"
            )
            if serial_data
            else []
        )

        selected_serial_overlays = []
        for opt in serial_selection:
            info = serial_data.get(opt)
            if not info:
                continue
            state_key = serial_label_state_keys.get(opt)
            legend_label = (
                st.session_state.get(state_key)
                if state_key and st.session_state.get(state_key)
                else info['default_label']
            )
            selected_serial_overlays.append(
                {
                    'df': info['df'],
                    'label': legend_label,
                }
            )

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
                selected_serial_overlays,
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
