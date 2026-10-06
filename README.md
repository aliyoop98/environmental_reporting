# Environmental Reporting v2.3

Environmental Reporting v2.3 keeps the multi-format ingestion and two-channel fixes from v2.2 and adds reusable report metadata keyed by serial number.

Every supported environmental input is normalized to one dataframe per serial:

`DateTime | Temperature | Humidity | Date | Time`

## Supported environmental CSV formats

1. Traceable per-device `Report*.csv` exports with `Device name`, `Device id`, and multiple `Channel Data for ...` blocks.
2. Consolidated / multi-serial Traceable exports with fields such as `Timestamp`, `Serial Number`, `Channel`, `Data`, and `Unit of Measure`.
3. Normalized wide CSVs with `DateTime`, `Serial Number`, and Temperature and/or Humidity.

Report-block ingestion keeps numeric environmental channels and ignores text-only channels such as `Door Open` / `Door Close`. Consolidated ingestion preserves Temperature and Humidity independently even when timestamps do not match perfectly.

## New in v2.3: reusable report metadata

The Streamlit sidebar now accepts an optional report metadata mapping in CSV or Excel format (`.csv`, `.xlsx`, `.xlsm`).

The mapping is keyed by serial number and can provide defaults for:

- Chart Title
- Probe ID
- Equipment ID
- Materials
- Environmental Profile
- Temperature lower / upper limits
- Humidity lower / upper limits
- Notes

Only `Serial Number` is required. Blank optional cells tell the app to keep its automatic/default value.

The app accepts common header aliases, so existing sheets often work without being rebuilt. Examples include `Serial`, `Device ID`, `Title`, `Probe`, `Equipment`, `Material`, `Temp Min`, `Temp Max`, `RH Min`, and `RH Max`.

If a serial appears more than once in the metadata file, the last row is used.

### Metadata does not lock the report

Uploaded metadata provides defaults only. Every populated value remains editable in Streamlit for one-off reports. Changing the metadata file resets those defaults cleanly because the metadata file content is included in the report-detail widget state key.

## Metadata template

Two generic templates are included:

- `report_metadata_template.xlsx`
- `report_metadata_template.csv`

After environmental files are uploaded, the sidebar also provides **Download metadata template for detected serials**. That workbook is pre-filled with every serial found in the current upload, making it faster to build a master mapping.

The Excel template contains:

- a `Metadata` sheet for the serial mappings,
- an `Instructions` sheet describing required and optional columns.

## Chart header format

Every Temperature/Humidity graph now uses the same two-line header structure as the reference reports:

`<Chart Title> - <Month Year> - <Channel>`

`Materials: <Materials> | Probe: <Probe ID> | Equipment: <Equipment ID>`

When Probe ID is available, the main chart series is labeled `Probe <ID>` in the legend along with `Lower Limit` and `Upper Limit`.

## Environmental profiles and limits

`environment_profiles.csv` remains the central default profile configuration. `Auto` uses parser-inferred limits. A metadata row may optionally specify a named profile or explicit channel limits; explicit metadata limits take precedence over profile defaults. Users can still edit the limits in Streamlit before relying on OOR analysis.

## Serial channel overrides

`serial_channel_overrides.csv` provides deterministic fallbacks for known devices when units/channel labels are incomplete. Clear unit evidence still has priority over overrides.

## Upload validation

The Upload validation table shows parser status, input format, serial, Temperature/Humidity source rows and plotted points, duplicates, unpaired timestamps, physical channels, date range, and source filenames.

If a metadata mapping is uploaded, a separate **Report metadata mapping** expander shows which mapping rows match the current environmental upload and which detected serials have no mapping row.

## GitHub files

Required application files:

- `app.py`
- `data_processing.py`
- `oor.py`
- `report_metadata.py`
- `serial_channel_overrides.csv`
- `environment_profiles.csv`
- `requirements.txt`

Recommended supporting files:

- `README.md`
- `report_metadata_template.xlsx`
- `report_metadata_template.csv`
- `test_report_format.csv`
- `test_traceable_channels.csv`
- `regression_three_serials.csv`

`traceable_ingest.py` is not required. All active environmental ingestion logic is in `data_processing.py`.
