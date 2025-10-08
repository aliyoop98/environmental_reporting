import io
from pathlib import Path
from typing import Optional, Tuple, Dict, Iterable, Mapping, List

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
    """Read CSV content supporting multiple delimiters and BOM-bearing headers."""

    if text:
        # Remove UTF-8 BOM from the start and any stray BOM occurrences within the text.
        text = text.lstrip("\ufeff")
        text = text.replace("\ufeff", "")

    for kwargs in ({"sep": None, "engine": "python"}, {}):
        try:
            df = pd.read_csv(
                io.StringIO(text), on_bad_lines="skip", index_col=False, **kwargs
            )
        except Exception:
            continue

        def _clean_column(column: str) -> str:
            if not isinstance(column, str):
                return column
            return (
                column.replace("\ufeff", "")
                .replace("\u200b", "")
                .replace("\u200c", "")
                .replace("\u200d", "")
                .strip()
            )

        return df.rename(columns=_clean_column)
    return None


def _normalize_header(value: str) -> str:
    """Return a normalized representation of a CSV header for matching."""

    value = (
        value.replace("\ufeff", "")
        .replace("\u200b", "")
        .replace("\u200c", "")
        .replace("\u200d", "")
    )

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

    def _find_match(
        aliases: Tuple[str, ...], used: set[str]
    ) -> Optional[str]:
        return next(
            (
                original
                for original, norm in normalized.items()
                if original not in used and norm in aliases
            ),
            None,
        )

    matches: Dict[str, str] = {}
    used: set[str] = set()

    required_fields = (
        'Timestamp',
        'Serial Number',
        'Channel',
        'Data',
    )
    optional_fields = (
        'Unit of Measure',
    )

    for canonical in required_fields:
        aliases = NEW_SCHEMA_ALIASES.get(canonical, ())
        match = _find_match(aliases, used)
        if match is None:
            return None
        matches[match] = canonical
        used.add(match)

    for canonical in optional_fields:
        aliases = NEW_SCHEMA_ALIASES.get(canonical, ())
        match = _find_match(aliases, used)
        if match is not None:
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


def _process_new_schema_df(
    df_full: pd.DataFrame,
    name: str,
    default_temp_range: Tuple[float, float],
    default_humidity_range: Tuple[float, float],
) -> List[Dict[str, object]]:
    canonical_map = _match_new_schema_columns(df_full.columns)
    if not canonical_map:
        return []

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
    if 'Unit of Measure' in df_new.columns:
        df_new['Unit of Measure'] = df_new['Unit of Measure'].astype(str)
    else:
        df_new['Unit of Measure'] = ''
    df_new = df_new.dropna(subset=['Timestamp', 'Serial Number'])
    df_new['Data'] = pd.to_numeric(df_new['Data'], errors='coerce')
    if df_new.empty:
        return []

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

    results: List[Dict[str, object]] = []
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
        unit_series = sdf['Unit of Measure'].fillna('').astype(str)
        unit_clean = unit_series.str.strip().str.lower()
        missing_measurement = sdf['Measurement'].isna()
        unit_missing = unit_clean.isin({'', 'nan', 'none', 'null'})
        fallback_mask = missing_measurement & unit_missing
        if fallback_mask.any():
            channel_norm = (
                sdf.loc[fallback_mask, 'Channel']
                .astype(str)
                .str.strip()
                .str.lower()
            )
            fallback_map = {
                'sensor1': 'Humidity',
                'sensor2': 'Temperature',
            }
            sdf.loc[fallback_mask, 'Measurement'] = channel_norm.map(fallback_map)
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

        range_map: Dict[str, Tuple[float, float]] = {}
        channels: List[str] = []
        if 'Temperature' in ordered_cols:
            range_map['Temperature'] = default_temp_range
            channels.append('Temperature')
        if 'Humidity' in ordered_cols:
            range_map['Humidity'] = default_humidity_range
            channels.append('Humidity')

        default_label = display_name or serial or Path(name).stem
        option_label = ''
        if serial and display_name:
            option_label = f"{serial} – {display_name}"
        elif serial:
            option_label = serial
        elif display_name:
            option_label = display_name
        else:
            option_label = label

        results.append(
            {
                'key': label,
                'df': pivot,
                'range_map': range_map,
                'serial': serial,
                'display_name': display_name,
                'default_label': default_label,
                'option_label': option_label,
                'source_name': name,
                'channels': channels,
            }
        )

    return results


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
            new_schema_groups = _process_new_schema_df(
                df_full, name, default_temp_range, default_humidity_range
            )
            if new_schema_groups:
                for group in new_schema_groups:
                    key = group['key']  # type: ignore[index]
                    dfs[key] = group['df']  # type: ignore[index]
                    range_map = group['range_map']  # type: ignore[index]
                    if range_map:
                        ranges[key] = range_map
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


def parse_serial_csv(files):
    serials: Dict[str, Dict[str, object]] = {}
    if not files:
        return serials
    default_temp_range = DEFAULT_TEMP_RANGE
    default_humidity_range = DEFAULT_HUMIDITY_RANGE
    for f in files:
        f.seek(0)
        raw = f.read().decode('utf-8', errors='ignore').splitlines(True)
        full_content = ''.join(raw)
        df_full = _read_csv_flexible(full_content)
        if df_full is None:
            continue
        groups = _process_new_schema_df(
            df_full, f.name, default_temp_range, default_humidity_range
        )
        for group in groups:
            key = group['key']  # type: ignore[index]
            serials[key] = {
                'df': group['df'],  # type: ignore[index]
                'range_map': group['range_map'],  # type: ignore[index]
                'serial': group['serial'],  # type: ignore[index]
                'default_label': group['default_label'],  # type: ignore[index]
                'option_label': group['option_label'],  # type: ignore[index]
                'source_name': group['source_name'],  # type: ignore[index]
                'channels': group['channels'],  # type: ignore[index]
            }
    return serials


def serial_data_to_primary(serials: Mapping[str, Dict[str, object]]) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Dict[str, Tuple[float, float]]]]:
    """Convert parsed serial datasets into primary probe-style structures.

    Parameters
    ----------
    serials:
        Mapping of serial dataset keys to the metadata produced by
        :func:`parse_serial_csv`.

    Returns
    -------
    Tuple[Dict[str, pd.DataFrame], Dict[str, Dict[str, Tuple[float, float]]]]
        Two dictionaries mirroring the output of :func:`_parse_probe_files`.
        The first maps dataset keys to processed dataframes suitable for the
        primary probe workflow, ensuring the ``Date`` column uses a
        ``datetime64`` dtype so downstream ``.dt`` accessors are valid.  The
        second maps dataset keys to their channel range configuration.
    """

    primary_dfs: Dict[str, pd.DataFrame] = {}
    primary_ranges: Dict[str, Dict[str, Tuple[float, float]]] = {}

    for key, info in serials.items():
        df = info.get('df')
        if df is None:
            continue
        df_converted = df.copy()
        if 'Date' in df_converted.columns:
            df_converted['Date'] = pd.to_datetime(df_converted['Date'], errors='coerce')
        primary_dfs[key] = df_converted
        range_map = info.get('range_map')
        if isinstance(range_map, dict):
            primary_ranges[key] = range_map
        else:
            primary_ranges[key] = {}

    return primary_dfs, primary_ranges


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
