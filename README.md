# Environmental Reporting v2.4

Environmental Reporting v2.4 adds TempStick backup logger support to the v2.3 Traceable workflow.

## Supported primary Traceable inputs

1. Traceable per-device `Report*.csv` exports containing device metadata and `Channel Data for ...` blocks.
2. Consolidated / multi-serial Traceable exports with `Timestamp`, `Serial Number`, `Channel`, `Data`, and units.
3. Normalized wide CSVs with `DateTime`, `Serial Number`, and Temperature and/or Humidity.

## TempStick backup overlay

The sidebar includes **TempStick backup CSV files (optional)**. TempStick custom exports may contain a sensor metadata row before the actual table; v2.4 detects that preamble and reads the subsequent `Timestamp,Temperature,Humidity,...` table.

After upload, each TempStick sensor is mapped to the Traceable serial it backs up. The app then overlays the TempStick data on the same monthly Temperature/Humidity graph. Traceable is shown as the primary series and TempStick as a separate dashed backup series. The backup values do not overwrite or merge into the stored Traceable dataframe.

If the report metadata mapping contains a `TempStick ID` for a serial, an uploaded TempStick with that ID is mapped automatically. Manual mapping remains available in the sidebar.

## OOR source policy

Traceable remains the default source for OOR analysis. When a mapped TempStick contains the selected channel, the user may choose:

- **Traceable probe only**
- **TempStick backup only**
- **Combined - prefer Traceable, fill gaps from TempStick**
- **Combined - prefer TempStick, fill gaps from Traceable**

Combined modes do not average or interpolate values. The preferred source is retained whenever the other source has a reading within the selected matching tolerance (default 5 minutes). The non-preferred source is used only where the preferred source has no nearby reading. This lets the user explicitly resolve an OOR-versus-in-range disagreement while still using the backup logger to fill a true primary-data gap.

The OOR event output records the contributing source. If an event contains values from both devices, the Source field lists both sources.

## Report metadata mapping

The optional metadata mapping supports:

- Serial Number
- TempStick ID
- Chart Title
- Probe ID
- Equipment ID
- Materials
- Environmental Profile
- Temperature lower / upper limits
- Humidity lower / upper limits
- Notes

Only `Serial Number` is required. `TempStick ID` is optional and is used only to automate backup mapping.

## Exports

The normalized CSV download remains the normalized **Traceable** dataset. If a TempStick is mapped for the selected serial/month, the report ZIP also contains a separate normalized TempStick backup CSV. OOR CSVs reflect the OOR source policy selected in the app.

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

No additional Python dependency is required for TempStick support beyond the existing v2.3 requirements.
