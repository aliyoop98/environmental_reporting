# Environmental Reporting

This Streamlit application generates out-of-range (OOR) compliance reports from Traceable data exports.

## Exports
- CSV out-of-range tables and a text audit summary are available under each serial tab.
- A **Bundle export (ZIP)** includes:
  - Combined OOR CSV
  - Per-channel OOR CSVs
  - Audit summary TXT
  - **Fixed-size PNG charts** (default 1280×480 @ 2× scale) for clean pasting into Word/Docs.

### Reproducible PNG charts
We use **vl-convert-python** to render Altair/Vega-Lite specs at a fixed canvas size.
This makes chart images crisp and **independent of browser/sidebar size**.

If PNGs are missing from the ZIP, ensure the dependency is installed:
```
pip install vl-convert-python
```
