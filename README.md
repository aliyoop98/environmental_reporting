# Environmental Reporting v2.2

Environmental Reporting v2.2 keeps the two-channel fixes from v2.1 and restores support for the per-device Traceable `Report*.csv` files used by older workflows.

Every supported input is normalized to one dataframe per serial:

`DateTime | Temperature | Humidity | Date | Time`

## Supported CSV formats

### 1. Traceable per-device Report exports

These files begin with metadata such as:

- `Device name:`
- `Device id:`
- `Location:`
- `Channel Data for ...`

They contain multiple independent CSV blocks rather than one normal table. v2.2 reads each block separately, retains numeric environmental measurements, and ignores text-only event channels such as `Door Open` / `Door Close`.

The device ID becomes the serial number and the device name becomes the display descriptor. Refrigerator/freezer profiles are inferred from device metadata rather than from random UUID text in the filename.

### 2. Consolidated / multi-serial Traceable exports

Expected raw fields include:

`Timestamp | Serial Number | Channel | Data | Unit of Measure`

Temperature and Humidity are preserved independently even when their timestamps do not match perfectly. Duplicate timestamps are coalesced per channel without allowing one channel to erase the other.

### 3. Normalized wide CSVs

The app can also re-read its own cleaned export format:

`DateTime | Serial Number | Temperature | Humidity`

At least one of Temperature or Humidity must be present.

## What v2.2 fixes

- Restores compatibility with Traceable `Report*.csv` block-based exports.
- Filters report blocks by numeric measurement evidence so door/event channels are not misclassified as Humidity or Temperature.
- Reads plain engineering-unit values such as `3.69 C` in addition to `°C`, `%`, and `%RH` forms.
- Uses `Device id` as the serial for report exports and carries `Device name` into the Streamlit selector label.
- Prevents filename UUID fragments such as `-80cc` from accidentally selecting the `Freezer -80` profile.
- Adds the detected input format to the Upload validation table.
- Keeps all v2.1 two-channel fixes for consolidated exports, including independent timestamp preservation, serial overrides, upload-content-aware widget state, parser cache versioning, and corrected chart downsampling.

## Validation performed for v2.2

The parser was regression-tested against:

- seven real `Report7600_52xx` per-device exports,
- the 160,903-row consolidated August 2026 production export,
- the three known two-channel problem serials (`250269653`, `250269655`, `250269656`),
- a synthetic block-report fixture containing Temperature, Humidity, and a text-only door sensor,
- a mixed upload containing all seven Report files plus the consolidated production file.

The mixed test produced 16 serial datasets: 7 report-format devices and 9 consolidated serials, all with `PASS` ingestion validation.

## GitHub files

Place these required files together in the Streamlit app directory:

- `app.py`
- `data_processing.py`
- `oor.py`
- `serial_channel_overrides.csv`
- `environment_profiles.csv`
- `requirements.txt`

Recommended documentation / QA files:

- `README.md`
- `test_report_format.csv` — synthetic Traceable Report-block fixture with Temperature, Humidity, and door-event data.
- `test_traceable_channels.csv` — consolidated edge cases for blank units, reversed sensor numbering, and units embedded in Data.
- `regression_three_serials.csv` — compact fixture derived from the August 2026 consolidated export for the three previously problematic serials.

`traceable_ingest.py` is not required. All active ingestion logic is in `data_processing.py`.

## Serial override CSV

`serial_channel_overrides.csv` provides deterministic fallbacks for known devices when units/channel labels are incomplete.

Example:

```csv
Serial Number,Channel,Measurement,Notes
123456789,sensor1,Temperature,Model-specific fallback
123456789,sensor2,Humidity,Model-specific fallback
```

Valid `Measurement` values are `Temperature` and `Humidity`. Clear unit evidence still has priority over overrides.

## Environment profiles

Edit `environment_profiles.csv` to change default OOR limits without changing Python. In the Streamlit app, `Auto` uses the inferred profile and users can explicitly select another profile when needed.

## Upload validation

After upload, expand **Upload validation**. The table shows:

- detected format,
- serial,
- raw measurement rows,
- Temperature/Humidity source rows and plotted points,
- duplicate and unpaired timestamps,
- physical channel mapping,
- date range,
- source filename,
- parser status/message.

A serial with `ERROR` should be investigated before relying on charts or OOR results.
