import io
from typing import Optional, Tuple, Dict

import pandas as pd

NEW_SCHEMA_COLUMNS = {
    'timestamp',
    'serial number',
    'channel',
    'data',
    'unit of measure',
}

DEFAULT_TEMP_RANGE: Tuple[float, float] = (15, 25)
DEFAULT_HUMIDITY_RANGE: Tuple[float, float] = (0, 60)


def _read_csv_flexible(text: str) -> Optional[pd.DataFrame]:
    """Read CSV content supporting multiple delimiters."""

    for kwargs in ({"sep": None, "engine": "python"}, {}):
        try:
            df = pd.read_csv(io.StringIO(text), on_bad_lines="skip", **kwargs)
        except Exception:
            continue
        return df.rename(columns=lambda c: c.strip() if isinstance(c, str) else c)
    return None


def _classify_measurement(channel: str, unit: str) -> Optional[str]:
    """Return canonical measurement name based on channel/unit metadata."""

    channel = (channel or '').strip().lower()
    unit = (unit or '').strip().lower()
    if any(token in channel for token in ('temp', '°c')) or 'c' in unit:
        return 'Temperature'
    if any(token in channel for token in ('hum', '%')) or '%' in unit or 'rh' in unit:
        return 'Humidity'
    return None


def _parse_probe_files(files):
    dfs = {}
    ranges: Dict[str, Dict[str, Tuple[float, float]]] = {}
    if not files:
        return dfs, ranges
    default_temp_range = DEFAULT_TEMP_RANGE
    default_humidity_range = DEFAULT_HUMIDITY_RANGE
    for f in files:
        f.seek(0)
        name = f.name
        raw = f.read().decode('utf-8', errors='ignore').splitlines(True)
        full_content = ''.join(raw)
        df_full = _read_csv_flexible(full_content)
        if df_full is not None:
            normalized = {
                c.strip().lower()
                for c in df_full.columns
                if isinstance(c, str)
            }
            if NEW_SCHEMA_COLUMNS.issubset(normalized):
                df_new = df_full.rename(
                    columns={
                        col: col.strip()
                        for col in df_full.columns
                        if isinstance(col, str)
                    }
                )
                canonical = {}
                for target in [
                    'Timestamp',
                    'Serial Number',
                    'Channel',
                    'Data',
                    'Unit of Measure',
                ]:
                    source = next(
                        (
                            c
                            for c in df_new.columns
                            if c.strip().lower() == target.lower()
                        ),
                        None,
                    )
                    if source:
                        canonical[source] = target
                df_new = df_new[list(canonical)].rename(columns=canonical)
                df_new['Timestamp'] = pd.to_datetime(
                    df_new['Timestamp'], errors='coerce'
                )
                df_new['Serial Number'] = (
                    df_new['Serial Number'].astype(str).str.strip()
                )
                df_new['Channel'] = df_new['Channel'].astype(str)
                df_new['Unit of Measure'] = df_new['Unit of Measure'].astype(str)
                df_new = df_new.dropna(subset=['Timestamp', 'Serial Number'])
                df_new['Data'] = pd.to_numeric(df_new['Data'], errors='coerce')
                if df_new.empty:
                    continue
                for serial, sdf in df_new.groupby('Serial Number'):
                    sdf = sdf.copy()
                    sdf['Measurement'] = sdf.apply(
                        lambda row: _classify_measurement(
                            row.get('Channel'), row.get('Unit of Measure')
                        ),
                        axis=1,
                    )
                    sdf = sdf.dropna(subset=['Measurement'])
                    if sdf.empty:
                        continue
                    sdf = sdf.sort_values('Timestamp')
                    pivot = sdf.pivot_table(
                        index='Timestamp',
                        columns='Measurement',
                        values='Data',
                        aggfunc='last',
                    )
                    if pivot.empty:
                        continue
                    pivot = pivot.rename_axis(None, axis=1).sort_index()
                    pivot['DateTime'] = pivot.index
                    pivot['Date'] = pivot['DateTime'].dt.date
                    pivot['Time'] = pivot['DateTime'].dt.strftime('%H:%M:%S')
                    ordered_cols = ['Date', 'Time', 'DateTime']
                    for col in ['Temperature', 'Humidity']:
                        if col in pivot.columns:
                            ordered_cols.append(col)
                    pivot = pivot[ordered_cols].reset_index(drop=True)
                    label = f"{name} [{serial}]"
                    dfs[label] = pivot
                    range_map = {}
                    if 'Temperature' in ordered_cols:
                        range_map['Temperature'] = default_temp_range
                    if 'Humidity' in ordered_cols:
                        range_map['Humidity'] = default_humidity_range
                    if range_map:
                        ranges[label] = range_map
                continue
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
            temp_range = (15, 28) if is_olympus else default_temp_range
            humidity_range = (0, 80) if is_olympus else default_humidity_range
            ranges[name] = {'Temperature': temp_range, 'Humidity': humidity_range}
        mapping.update({'Date': 'Date', 'Time': 'Time'})
        col_map = {}
        for new, key in mapping.items():
            col = next((c for c in df.columns if key.lower() in c.lower()), None)
            if col:
                col_map[col] = new
        df = df[list(col_map)].rename(columns=col_map)
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
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
            df_ts['DateTime'], errors='coerce'
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
