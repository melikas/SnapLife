Sleep data export and processing

This folder contains processed sleep data and the script used to generate it.

Files included:
- `sleepdata/csv-sleep ratio-hourly.csv` — hourly sleep aggregates per `pid`.
- `sleep_hourly_from_bson.py` — script used to create the CSV from `fitbit.bson`.

Notes:
- The CSV columns (exact order) are: `pid, datetime, hour, sleep_state, wake_ratio, light_ratio, deep_ratio, rem_ratio, sleep_intensity, minutes`.
- Missing values are written as `nan`.
- If you want to re-run the processing, make sure `fitbit.bson` is present and run `python sleep_hourly_from_bson.py`.

Recommended next steps:
- Use Git LFS for large CSVs if you plan to push them to a remote repository.
