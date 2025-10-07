import io
from typing import Optional, Tuple, Dict, Iterable, Mapping

import pandas as pd

NEW_SCHEMA_ALIASES: Mapping[str, Tuple[str, ...]] = {
    'Timestamp': (
        'timestamp',
        'time stamp',
        'date time',
        'date/time',
        'datetime',
    ),
    'Serial Number': (
        'serial number',
        'serial num',
        'serial no',
        'serial #',
        'serial',
        'serialnumber',
        'serialno',
    ),
    'Channel': (
        'channel',
        'sensor channel',
        'probe channel',
        'sensor',
    ),
    'Data': (
        'data',
        'value',
        'reading',
    ),
    'Unit of Measure': (
        'unit of measure',
        'unit',
        'units',
        'uom',
        'measurement unit',
    ),
}

_SPACE_NAME_HINTS: Tuple[Tuple[str, ...], ...] = (
    ("assignment", "name"),
    ("space", "name"),
    ("location", "name"),
    ("asset", "name"),
    ("subject", "name"),
    ("target", "name"),
    ("environment", "name"),
    ("area", "name"),
    ("zone", "name"),
    ("equipment", "name"),
    ("ambient", "name"),
    ("probe", "name"),
)

_SPACE_TYPE_HINTS: Tuple[Tuple[str, ...], ...] = (
    ("assignment", "type"),
    ("space", "type"),
    ("location", "type"),
    ("asset", "type"),
    ("subject", "type"),
    ("target", "type"),
    ("environment", "type"),
    ("area", "type"),
    ("zone", "type"),
    ("probe", "type"),
    ("space", "category"),
    ("environment", "category"),
)

DEFAULT_TEMP_RANGE: Tuple[float, float] = (15, 25)
DEFAULT_HUMIDITY_RANGE: Tuple[float, float] = (0, 60)


def _read_csv_flexible(text: str) -> Optional[pd.DataFrame]:
    """Read CSV content supporting multiple delimiters."""

    for kwargs in ({"sep": None, "engine": "python"}, {}):
        try:
            df = pd.read_csv(
                io.StringIO(text), on_bad_lines="skip", index_col=False, **kwargs
            )
        except Exception:
            continue
        return df.rename(columns=lambda c: c.strip() if isinstance(c, str) else c)
    return None


def _normalize_header(value: str) -> str:
    """Return a normalized representation of a CSV header for matching."""

    cleaned = value.strip().lower()
    cleaned = cleaned.replace('#', ' number ')
    cleaned = cleaned.replace('-', ' ')
    cleaned = cleaned.replace('_', ' ')
    cleaned = ' '.join(cleaned.split())
    return cleaned


def _match_new_schema_columns(columns: Iterable[str]) -> Optional[Dict[str, str]]:
    """Return mapping of original headers to canonical names for the new schema."""

    normalized = {
        col: _normalize_header(col)
        for col in columns
        if isinstance(col, str)
    }
    matches: Dict[str, str] = {}
    used: set[str] = set()
    for canonical, aliases in NEW_SCHEMA_ALIASES.items():
        match = next(
            (
                original
                for original, norm in normalized.items()
                if original not in used and norm in aliases
            ),
            None,
        )
        if match is None:
            return None
        matches[match] = canonical
        used.add(match)
    return matches


def _classify_measurement(channel: str, unit: str) -> Optional[str]:
    """Return canonical measurement name based on channel/unit metadata."""

    channel = (channel or '').strip().lower()
    unit = (unit or '').strip().lower()

    # Normalise the unit text so we can reliably inspect the tokens regardless of
    # punctuation, case, or unicode symbols (e.g. "°F").
    unit_normalised = (
        unit.replace('°', ' ')
        .replace('degrees', 'deg')
        .replace('/', ' ')
        .replace('-', ' ')
        .replace('_', ' ')
    )
    unit_tokens = {token for token in unit_normalised.split() if token}

    humidity_tokens = {'%', 'percent', 'humidity', 'humid', 'rh'}
    if (
        any(token in channel for token in ('hum', '%'))
        or '%' in unit
        or unit_tokens & humidity_tokens
    ):
        return 'Humidity'

    temperature_tokens = {
        'temp',
        'temperature',
        'degc',
        'c',
        'celsius',
        'degf',
        'f',
        'fahrenheit',
        'degk',
        'k',
        'kelvin',
    }
    if (
        any(token in channel for token in ('temp', '°c', '°f'))
        or unit_tokens & temperature_tokens
    ):
        return 'Temperature'

    return None


def _find_column_by_terms(columns: Iterable[str], hints: Iterable[Tuple[str, ...]]) -> Optional[str]:
    """Return the first column that matches any group of hint terms."""

    for terms in hints:
        for col in columns:
            if not isinstance(col, str):
                continue
            lower = col.lower()
            if all(term in lower for term in terms):
                return col
    return None


def _clean_metadata_series(series: pd.Series) -> pd.Series:
    """Normalize free-text metadata columns used for grouping."""

    normalized = []
    for value in series.astype(str).tolist():
        text = value.strip()
        normalized.append("" if text.lower() in {"", "nan", "none", "null"} else text)
    return pd.Series(normalized, index=series.index)


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
            canonical_map = _match_new_schema_columns(df_full.columns)
            if canonical_map:
                strip_map = {
                    col: col.strip()
                    for col in df_full.columns
                    if isinstance(col, str)
                }
                df_new = df_full.rename(columns=strip_map)
                canonical = {
                    strip_map.get(original, original): canonical_name
                    for original, canonical_name in canonical_map.items()
                }
                df_new = df_new.rename(columns=canonical)
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
                space_name_col = _find_column_by_terms(df_new.columns, _SPACE_NAME_HINTS)
                space_type_col = _find_column_by_terms(df_new.columns, _SPACE_TYPE_HINTS)
                if space_name_col:
                    df_new['_SpaceName'] = _clean_metadata_series(df_new[space_name_col])
                else:
                    df_new['_SpaceName'] = ''
                if space_type_col:
                    df_new['_SpaceType'] = _clean_metadata_series(df_new[space_type_col])
                else:
                    df_new['_SpaceType'] = ''

                group_columns = ['Serial Number']
                if space_name_col:
                    group_columns.append('_SpaceName')
                if space_type_col:
                    group_columns.append('_SpaceType')

                for keys, sdf in df_new.groupby(group_columns):
                    if not isinstance(keys, tuple):
                        keys = (keys,)
                    key_map = dict(zip(group_columns, keys))
                    serial = str(key_map.get('Serial Number', '')).strip()
                    space_name = str(key_map.get('_SpaceName', '')).strip()
                    space_type = str(key_map.get('_SpaceType', '')).strip()
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
                    display_name = ''
                    if space_name and space_type:
                        display_name = f"{space_name} ({space_type})"
                    elif space_name:
                        display_name = space_name
                    elif space_type:
                        display_name = space_type
                    label_prefix = name if not display_name else f"{name} - {display_name}"
                    label = f"{label_prefix} [{serial}]" if serial else label_prefix
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
