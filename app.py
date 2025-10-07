import copy
import hashlib
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple

import streamlit as st
import pandas as pd
import calendar
import altair as alt
from datetime import datetime, timedelta

# Ensure imports work whether the app is executed as a module (e.g. via
# ``python -m environmental_reporting.app``) or as a script where the repository
# directory might not already be on ``sys.path``.
CURRENT_DIR = Path(__file__).resolve().parent

if __package__:
    from .data_processing import (
        _parse_probe_files,
        _parse_tempstick_files,
        parse_serial_csv,
    )
else:
    if str(CURRENT_DIR) not in sys.path:
        sys.path.insert(0, str(CURRENT_DIR))
    from data_processing import (
        _parse_probe_files,
        _parse_tempstick_files,
        parse_serial_csv,
    )


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
serial_files = st.sidebar.file_uploader(
    "Upload Consolidated Serial CSV files",
    accept_multiple_files=True,
    key="serial_uploader"
)
probe_files = st.sidebar.file_uploader(
    "Upload Probe CSV files (optional)",
    accept_multiple_files=True,
    key="probe_uploader"
)


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

def _prepare_serial_primary(
    serial_data: Dict[str, Dict[str, object]]
) -> Tuple[
    Dict[str, pd.DataFrame],
    Dict[str, Dict[str, Tuple[float, float]]],
    Dict[str, Dict[str, object]],
]:
    """Aggregate serial datasets by serial number for primary charting."""

    serial_groups: Dict[str, Dict[str, object]] = {}
    for key, info in serial_data.items():
        df = info.get('df') if isinstance(info, dict) else None
        if df is None:
            continue
        serial_value = str(info.get('serial') or '').strip()
        group_key = serial_value or str(key)
        group = serial_groups.setdefault(
            group_key,
            {
                'dfs': [],
                'range_maps': [],
                'option_labels': [],
                'default_labels': [],
                'source_names': [],
                'serial': serial_value,
            },
        )
        group['dfs'].append(df)
        range_map = info.get('range_map') if isinstance(info, dict) else None
        if isinstance(range_map, dict):
            group['range_maps'].append(range_map)
        option_label = info.get('option_label') if isinstance(info, dict) else None
        if option_label:
            group['option_labels'].append(option_label)
        default_label = info.get('default_label') if isinstance(info, dict) else None
        if default_label:
            group['default_labels'].append(default_label)
        source_name = info.get('source_name') if isinstance(info, dict) else None
        if source_name:
            group['source_names'].append(source_name)

    primary_dfs: Dict[str, pd.DataFrame] = {}
    primary_ranges: Dict[str, Dict[str, Tuple[float, float]]] = {}
    metadata: Dict[str, Dict[str, object]] = {}

    for group_key, group in serial_groups.items():
        dfs: List[pd.DataFrame] = group.get('dfs', [])  # type: ignore[assignment]
        if not dfs:
            continue
        combined = pd.concat(dfs, ignore_index=True)
        if 'DateTime' in combined.columns:
            combined = combined.sort_values('DateTime')
        if 'Date' in combined.columns:
            combined['Date'] = pd.to_datetime(combined['Date'], errors='coerce')
        combined = combined.reset_index(drop=True)

        range_map: Dict[str, Tuple[float, float]] = {}
        for rm in group.get('range_maps', []):  # type: ignore[assignment]
            range_map.update(rm)

        channel_candidates = [
            col
            for col in combined.columns
            if col not in {'Date', 'Time', 'DateTime'}
        ]

        descriptor = next(
            (label for label in group.get('option_labels', []) if label),
            '',
        )
        if not descriptor:
            descriptor = next(
                (label for label in group.get('default_labels', []) if label),
                '',
            )

        serial_value = group.get('serial') or ''
        tab_label = ''
        if serial_value and descriptor and serial_value not in descriptor:
            tab_label = f"{serial_value} – {descriptor}"
        elif serial_value:
            tab_label = str(serial_value)
        elif descriptor:
            tab_label = descriptor
        else:
            tab_label = group_key

        legend_default = descriptor or serial_value or tab_label
        source_names = sorted({name for name in group.get('source_names', []) if name})

        metadata[group_key] = {
            'serial': serial_value,
            'tab_label': tab_label,
            'legend_default': legend_default,
            'descriptor': descriptor,
            'source_label': ', '.join(source_names),
            'channels': channel_candidates,
        }

        primary_dfs[group_key] = combined
        primary_ranges[group_key] = range_map

    return primary_dfs, primary_ranges, metadata

serial_data = parse_serial_csv(serial_files)
if not serial_data:
    st.sidebar.info("Upload consolidated serial CSV files to begin.")
    st.stop()

primary_dfs, primary_ranges, serial_metadata = _prepare_serial_primary(serial_data)
if not primary_dfs:
    st.sidebar.info("No valid serial data found in the uploaded files.")
    st.stop()


def _collect_years_months(dataframes: Dict[str, pd.DataFrame]) -> Tuple[List[int], List[int]]:
    """Return sorted lists of distinct years and months from uploaded data."""

    years: set[int] = set()
    months: set[int] = set()

    for df in dataframes.values():
        date_series = pd.Series(dtype="datetime64[ns]")
        if "Date" in df.columns:
            date_series = pd.to_datetime(df["Date"], errors="coerce")
        elif "DateTime" in df.columns:
            date_series = pd.to_datetime(df["DateTime"], errors="coerce")

        valid = date_series.dropna()
        if valid.empty:
            continue

        years.update(valid.dt.year.unique().tolist())
        months.update(valid.dt.month.unique().tolist())

    return sorted(years), sorted(months)

probe_dfs, _ = _parse_probe_files(probe_files)
probe_labels: Dict[str, str] = {}
if probe_dfs:
    st.sidebar.caption("Probe datasets")
    for probe_name in probe_dfs:
        label_key = _state_key("probe_label", probe_name)
        default_label = Path(probe_name).stem
        st.session_state.setdefault(label_key, default_label)
        value = st.sidebar.text_input(
            f"Legend label for {probe_name}",
            key=label_key
        )
        probe_labels[probe_name] = value.strip() if isinstance(value, str) and value.strip() else default_label

def _serial_sort_key(item: Tuple[str, Dict[str, object]]) -> Tuple[str, str]:
    key, meta = item
    serial_value = str(meta.get('serial') or '').strip()
    tab_label = str(meta.get('tab_label') or key)
    primary_value = serial_value or tab_label
    return (primary_value.lower(), tab_label.lower())


serial_keys = [key for key, _ in sorted(serial_metadata.items(), key=_serial_sort_key)]

unique_serials = sorted(
    {
        str(meta.get('serial') or key).strip()
        for key, meta in serial_metadata.items()
        if str(meta.get('serial') or key).strip()
    }
)
if unique_serials:
    st.sidebar.markdown("### Serial Numbers Detected")
    st.sidebar.markdown("\n".join(f"- {serial}" for serial in unique_serials))

probe_assignments: Dict[str, List[str]] = {}
serial_options = serial_keys


def _format_serial_option(option: str) -> str:
    meta = serial_metadata.get(option, {})
    serial_value = meta.get('serial')
    descriptor = meta.get('descriptor')
    tab_label = meta.get('tab_label')
    if serial_value and descriptor and serial_value not in descriptor:
        return f"{serial_value} – {descriptor}"
    return tab_label or option


if probe_dfs and serial_options:
    st.sidebar.markdown("### Probe Assignments")
    for probe_name in probe_dfs:
        assign_key = _state_key("probe_assignment", probe_name)
        if assign_key in st.session_state:
            existing = [
                opt for opt in st.session_state[assign_key] if opt in serial_options
            ]
            if existing != st.session_state[assign_key]:
                st.session_state[assign_key] = existing
        else:
            st.session_state[assign_key] = []
        probe_assignments[probe_name] = st.sidebar.multiselect(
            f"Assign {probe_name}",
            options=serial_options,
            format_func=_format_serial_option,
            key=assign_key,
        )
# Year & Month selection
years, months = _collect_years_months(primary_dfs)
if not years or not months:
    st.sidebar.warning("No valid timestamp data found in the uploaded serial files.")
    st.stop()
year = st.sidebar.selectbox("Year", years)
month = st.sidebar.selectbox("Month", months, format_func=lambda m: calendar.month_name[m])

for name in serial_keys:
    meta = serial_metadata.get(name, {})
    default_title = meta.get('tab_label') or str(name)
    chart_key = f"chart_title_{name}"
    if chart_key not in st.session_state:
        st.session_state[chart_key] = default_title
    primary_label_key = f"primary_label_{name}"
    default_primary_label = meta.get('legend_default') or default_title or "Serial Data"
    if primary_label_key not in st.session_state or st.session_state[primary_label_key] in {"", "Probe"}:
        st.session_state[primary_label_key] = default_primary_label
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


tab_titles = [serial_metadata[name].get('tab_label') or str(name) for name in serial_keys]
tabs = st.tabs(tab_titles)
for tab, name in zip(tabs, serial_keys):
    with tab:
        df = primary_dfs[name]
        meta = serial_metadata.get(name, {})
        st.subheader(meta.get('tab_label') or str(name))
        source_label = meta.get('source_label')
        if source_label:
            st.caption(f"Source: {source_label}")

        st.markdown("### Chart Details")
        chart_title_key = f"chart_title_{name}"
        chart_title = st.text_input("Chart Title", key=chart_title_key)
        materials_key = f"materials_{name}"
        materials = st.text_input("Materials List", key=materials_key)
        probe_id_key = f"probe_{name}"
        probe_id = st.text_input("Probe ID", key=probe_id_key)
        equip_id_key = f"equip_{name}"
        equip_id = st.text_input("Equipment ID", key=equip_id_key)

        st.markdown("### Optional Overlays")
        assigned_probes = [
            probe_name
            for probe_name, assigned in probe_assignments.items()
            if name in assigned
        ]
        selected_probe_overlays: List[str] = []
        probe_selection_key = f"probe_select_{name}"
        if assigned_probes:
            if probe_selection_key in st.session_state:
                current = [
                    opt for opt in st.session_state[probe_selection_key]
                    if opt in assigned_probes
                ]
                st.session_state[probe_selection_key] = current
            selected_probe_overlays = st.multiselect(
                "Assigned probe overlays",
                options=assigned_probes,
                default=assigned_probes,
                format_func=lambda opt: probe_labels.get(opt, Path(opt).stem),
                key=probe_selection_key,
            )
        elif probe_dfs:
            st.info("Assign probe datasets in the sidebar to overlay them with this serial.")
        else:
            st.caption("No probe datasets uploaded.")

        tempstick_files = st.file_uploader(
            "Upload Tempstick CSV files (optional)",
            accept_multiple_files=True,
            key=f"tempstick_{name}"
        )
        tempdfs = _parse_tempstick_files(tempstick_files)

        tempstick_labels: Dict[str, str] = {}
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

        st.markdown("### Legend Labels")
        primary_label_key = f"primary_label_{name}"
        st.text_input("Legend label for serial data", key=primary_label_key)
        all_primary_labels = {
            fname: (
                st.session_state.get(f"primary_label_{fname}")
                or serial_metadata.get(fname, {}).get('legend_default')
                or str(fname)
            )
            for fname in serial_keys
        }

        secondary_labels = {
            sec_name: probe_labels.get(sec_name, Path(sec_name).stem)
            for sec_name in probe_dfs
        }

        comparison_options = {
            f"secondary::{sec_name}": ("secondary", sec_name)
            for sec_name in selected_probe_overlays
        }
        comparison_probes = list(comparison_options.keys())

        channel_defaults = list(primary_ranges.get(name, {}).keys())
        if not channel_defaults:
            channel_defaults = [
                col
                for col in df.columns
                if col not in {'Date', 'Time', 'DateTime'}
                and pd.api.types.is_numeric_dtype(df[col])
            ]
        channels = st.multiselect(
            "Channels to plot",
            options=channel_defaults,
            default=channel_defaults,
            key=f"channels_{name}"
        )

        selected_serial_overlays: List[Dict[str, object]] = []

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
                probe_dfs,
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
