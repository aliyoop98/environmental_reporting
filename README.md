# Environmental Reporting v2.1

This version is built around one canonical ingestion path for the Streamlit app. Every serial becomes one dataframe with:

`DateTime | Temperature | Humidity | Date | Time`

## What v2.1 fixes

- Preserves Temperature and Humidity independently when the two sensors do **not** report at exactly the same timestamps.
- Coalesces duplicate timestamps without allowing one channel to erase the other.
- Uses unit labels first, then serial-specific overrides, explicit channel names, and finally sensor-number fallback.
- Adds an upload-validation table showing source rows, plotted points, duplicates, unpaired timestamps, detected physical-channel mapping, and parser status for every serial.
- Resets channel-selection widget state whenever uploaded file contents change, preventing Streamlit from keeping an old one-channel selection for the same serial/month.
- Adds an explicit parser version to the cache signature so parser fixes cannot reuse stale cached results.
- Downsamples charts correctly with `ceil()` while OOR calculations continue to use the full-resolution series.
- Replaces slow row-wise channel inference and timestamp coalescing with vectorized/pandas-native operations.

On the 160,903-row August 2026 production export used for validation, the parser completed in about 3 seconds in the development container versus about 23 seconds before the optimization. Runtime on Streamlit hosting will vary.

## GitHub files

Place these files together in the Streamlit app directory:

- `app.py`
- `data_processing.py`
- `oor.py`
- `serial_channel_overrides.csv`
- `environment_profiles.csv`
- `requirements.txt`

Optional QA files:

- `regression_three_serials.csv` — compact fixture derived from the August 2026 production export and containing paired, duplicate, and unpaired rows for serials `250269653`, `250269655`, and `250269656`.
- `production_validation_aug2026.csv` — validation summary generated from the full uploaded August 2026 export.
- `test_traceable_channels.csv` — synthetic edge cases for blank units, reversed sensor numbering, and units embedded in the Data field.

`traceable_ingest.py` is no longer needed. Its useful serial-override concept has been incorporated into the active parser/configuration path.

## Serial override CSV

`serial_channel_overrides.csv` is intentionally small. Add a mapping only when a device/model requires a deterministic fallback, for example:

```csv
Serial Number,Channel,Measurement,Notes
123456789,sensor1,Temperature,Model-specific fallback
123456789,sensor2,Humidity,Model-specific fallback
```

Valid `Measurement` values are `Temperature` and `Humidity`.

Clear unit labels still take precedence over an override, because units are the strongest evidence in a Traceable export.

## Environment profiles

Edit `environment_profiles.csv` to change default OOR limits without editing Python. `Auto` uses parser-inferred defaults; the Streamlit UI can still override limits for the selected report.

## Validation behavior

The app never requires Temperature and Humidity to share the same timestamp. If one channel is present without the other, that available reading remains in the canonical dataset. Duplicate rows at the same timestamp are coalesced per channel using the latest non-null value.

A serial receives `ERROR` in Upload validation if source rows indicate a measurement but no canonical points survive, or if multiple physical channels resolve incorrectly. Do not rely on charts/OOR results for a serial showing `ERROR` until its mapping is corrected.
