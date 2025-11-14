#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Script: sleep_hourly_from_bson.py
# Reads local fitbit.bson, extracts sleep 'levels' data, aggregates per-hour ratios and saves CSV

import os
from bson import decode_file_iter
import pandas as pd
import numpy as np
from collections import defaultdict


def parse_level_point(pt):
    ts = pt.get("dateTime") or pt.get("date_time") or pt.get("datetime") or pt.get("time")
    if ts is None:
        return None
    try:
        ts = pd.to_datetime(ts, infer_datetime_format=True)
    except Exception:
        try:
            ts = pd.to_datetime(str(ts))
        except Exception:
            return None

    seconds = pt.get("seconds") or pt.get("duration") or pt.get("sec") or pt.get("secs")
    try:
        seconds = int(seconds) if seconds is not None else 0
    except Exception:
        try:
            seconds = int(float(seconds))
        except Exception:
            seconds = 0

    level = pt.get("level") or pt.get("value") or pt.get("name") or ""
    level = level.lower() if isinstance(level, str) else str(level).lower()

    intensity = pt.get("intensity") if pt.get("intensity") is not None else np.nan
    try:
        intensity = float(intensity)
    except Exception:
        intensity = np.nan

    return ts, seconds, level, intensity


def level_weight(level):
    w = {"wake": 0.0, "awake": 0.0, "light": 1.0, "rem": 1.5, "deep": 2.0}
    return w.get(level.lower(), 1.0)


def process_bson_sleep(bson_path="fitbit.bson"):
    if not os.path.exists(bson_path):
        raise FileNotFoundError(f"{bson_path} not found in {os.getcwd()}")

    agg = defaultdict(lambda: {"wake": 0, "light": 0, "deep": 0, "rem": 0, "intensity_seconds_sum": 0.0, "total_seconds": 0})

    with open(bson_path, "rb") as fh:
        doc_count = 0
        for doc in decode_file_iter(fh):
            doc_count += 1
            if doc_count % 10000 == 0:
                print(f"Processed {doc_count} BSON documents...", flush=True)
            payload = doc.get("data") if isinstance(doc.get("data"), dict) else doc
            typ = doc.get("type") or payload.get("type")
            if typ != "sleep":
                continue

            pid = str(doc.get("id") or payload.get("id") or payload.get("userId") or payload.get("user_id") or "")

            sleep_obj = payload if doc.get("data") is None else payload
            levels = None
            if isinstance(sleep_obj, dict):
                levels = sleep_obj.get("levels") or (sleep_obj.get("data") and sleep_obj.get("data").get("levels"))

            # Ensure we create placeholder hourly records for the whole sleep session
            # when start/end timestamps exist, so hours without minute-level points
            # will appear in the output with NaNs.
            start_ts = None
            end_ts = None
            if isinstance(sleep_obj, dict):
                # common field names for start/end
                start_ts = sleep_obj.get("startTime") or sleep_obj.get("start") or sleep_obj.get("start_time") or sleep_obj.get("startDate")
                end_ts = sleep_obj.get("endTime") or sleep_obj.get("end") or sleep_obj.get("end_time") or sleep_obj.get("endDate")
            try:
                if start_ts is not None:
                    start_ts = pd.to_datetime(start_ts)
                if end_ts is not None:
                    end_ts = pd.to_datetime(end_ts)
            except Exception:
                start_ts = end_ts = None

            # If start and end are present, create hourly keys (placeholders) between them
            if start_ts is not None and end_ts is not None and start_ts <= end_ts:
                # floor to hour
                start_hour = pd.Timestamp(start_ts).floor("H")
                end_hour = pd.Timestamp(end_ts).floor("H")
                hrs = pd.date_range(start_hour, end_hour, freq="H")
                for h in hrs:
                    agg[(pid, h)]  # touch to ensure placeholder entry in defaultdict

            data_points = (levels.get("data") if levels else []) or []
            for pt in data_points:
                parsed = parse_level_point(pt)
                if parsed is None:
                    continue
                ts, seconds, level, intensity = parsed
                if seconds <= 0:
                    continue

                hour_start = pd.Timestamp(ts.floor("H"))
                key = (pid, hour_start)

                if "deep" in level:
                    lvl = "deep"
                elif "rem" in level:
                    lvl = "rem"
                elif "light" in level:
                    lvl = "light"
                elif "wake" in level or "awake" in level:
                    lvl = "wake"
                else:
                    lvl = "light"

                agg[key][lvl] += seconds
                agg[key]["total_seconds"] += seconds
                w = intensity if not np.isnan(intensity) else level_weight(lvl)
                agg[key]["intensity_seconds_sum"] += w * seconds

    # After aggregation, ensure no hour has more than 3600 seconds recorded.
    # If it does, scale the per-level seconds and intensity sum proportionally so
    # ratios and intensity remain consistent but total_seconds caps at 3600.
    for k, v in list(agg.items()):
        tot = v.get("total_seconds", 0)
        if tot > 3600:
            sf = 3600.0 / float(tot)
            v["wake"] = v["wake"] * sf
            v["light"] = v["light"] * sf
            v["deep"] = v["deep"] * sf
            v["rem"] = v["rem"] * sf
            v["intensity_seconds_sum"] = v["intensity_seconds_sum"] * sf
            v["total_seconds"] = 3600.0

    rows = []
    for (pid, hour_start), vals in agg.items():
        total_raw = vals["total_seconds"]
        if total_raw == 0:
            wake_r = light_r = deep_r = rem_r = np.nan
            intensity = np.nan
            dominant = np.nan
            minutes = np.nan
        else:
            # Use local copies of seconds per-level and intensity sum
            wake_s = float(vals["wake"])
            light_s = float(vals["light"])
            deep_s = float(vals["deep"])
            rem_s = float(vals["rem"])
            intensity_sum = float(vals["intensity_seconds_sum"])

            # If total seconds exceed one hour, scale seconds proportionally to cap at 3600s
            total = float(total_raw)
            if total > 3600.0:
                scale = 3600.0 / total
                wake_s *= scale
                light_s *= scale
                deep_s *= scale
                rem_s *= scale
                intensity_sum *= scale
                total = 3600.0

            # Compute ratios from (possibly scaled) seconds
            wake_r = wake_s / total if total > 0 else np.nan
            light_r = light_s / total if total > 0 else np.nan
            deep_r = deep_s / total if total > 0 else np.nan
            rem_r = rem_s / total if total > 0 else np.nan

            # Normalize in case of tiny FP rounding issues
            s = (0.0 if np.isnan(wake_r) else wake_r) + (0.0 if np.isnan(light_r) else light_r) + (0.0 if np.isnan(deep_r) else deep_r) + (0.0 if np.isnan(rem_r) else rem_r)
            if s != 0 and not np.isnan(s):
                wake_r /= s
                light_r /= s
                deep_r /= s
                rem_r /= s

            intensity = intensity_sum / total if total > 0 else np.nan
            level_order = {"wake": wake_s, "light": light_s, "deep": deep_s, "rem": rem_s}
            dominant = max(level_order, key=lambda k: level_order[k]) if any(v > 0 for v in level_order.values()) else np.nan

            # total seconds converted to minutes (includes all sleep types)
            try:
                minutes = int(round(total / 60.0))
            except Exception:
                minutes = np.nan
        rows.append({
            "pid": pid,
            "datetime": hour_start.to_pydatetime(),
            "hour": int(hour_start.hour),
            "sleep_state": dominant,
            "wake_ratio": wake_r,
            "light_ratio": light_r,
            "deep_ratio": deep_r,
            "rem_ratio": rem_r,
            "sleep_intensity": intensity,
            "minutes": minutes,
        })

    df_out = pd.DataFrame(rows, columns=["pid", "datetime", "hour", "sleep_state", "wake_ratio", "light_ratio", "deep_ratio", "rem_ratio", "sleep_intensity", "minutes"])

    # Ensure datetime is a pandas datetime
    df_out["datetime"] = pd.to_datetime(df_out["datetime"])

    # Build a full hourly grid per pid from each pid's min->max hour and reindex so
    # missing hours become rows with NaN values (as requested).
    full_index_tuples = []
    for pid, sub in df_out.groupby("pid"):
        try:
            min_dt = pd.to_datetime(sub["datetime"].min()).floor("H")
            max_dt = pd.to_datetime(sub["datetime"].max()).floor("H")
        except Exception:
            continue
        if pd.isna(min_dt) or pd.isna(max_dt):
            continue
        hrs = pd.date_range(min_dt, max_dt, freq="H")
        full_index_tuples.extend([(pid, h) for h in hrs])

    if full_index_tuples:
        full_index = pd.MultiIndex.from_tuples(full_index_tuples, names=["pid", "datetime"])
        df_full = df_out.set_index(["pid", "datetime"]).reindex(full_index).reset_index()
        # fill hour column from datetime for reindexed rows
        df_full["hour"] = pd.to_datetime(df_full["datetime"]).dt.hour
    # keep requested column order (include minutes)
    df_out = df_full[["pid", "datetime", "hour", "sleep_state", "wake_ratio", "light_ratio", "deep_ratio", "rem_ratio", "sleep_intensity", "minutes"]]

    return df_out


if __name__ == "__main__":
    out_csv = "csv-sleep ratio-hourly.csv"
    print("Processing fitbit.bson -> building hourly sleep ratios...")
    df = process_bson_sleep("fitbit.bson")
    print(f"Rows produced: {len(df)}. Saving to {out_csv}...")
    # Write explicit 'nan' for missing values so empty/missing levels are visible in the CSV
    df.to_csv(out_csv, index=False, na_rep='nan')
    print("Done.")
