# -*- coding: utf-8 -*-
"""
        "deep": 0,
        "rem": 0,
        "intensity_seconds_sum": 0.0,
        "total_seconds": 0,
    })

    # read BSON file and keep only sleep-type documents
    with open(bson_path, "rb") as fh:
        for doc in decode_file_iter(fh):
            # Many exports wrap the payload in a 'data' field; support both shapes
            payload = doc.get("data") if isinstance(doc.get("data"), dict) else doc
            # type marker may be top-level or under payload
            typ = doc.get("type") or payload.get("type")
            if typ != "sleep":
                continue

            pid = str(doc.get("id") or payload.get("id") or payload.get("userId") or payload.get("user_id") )

            # data-level sleep object
            sleep_obj = payload if doc.get("data") is None else payload

            # levels structure may be in sleep_obj['levels'] or under sleep_obj directly
            levels = None
            if isinstance(sleep_obj, dict):
                levels = sleep_obj.get("levels") or (sleep_obj.get("data") and sleep_obj.get("data").get("levels"))

            if not levels:
                # no minute-level levels available for this doc -> nothing to aggregate
                continue

            data_points = levels.get("data") or []

            for pt in data_points:
                parsed = parse_level_point(pt)
                if parsed is None:
                    continue
                ts, seconds, level, intensity = parsed
                if seconds <= 0:
                    continue

                # hour window start (datetime aligned to hour)
                hour_start = pd.Timestamp(ts.floor("H"))
                key = (pid, hour_start)

                # map common level names to canonical ones
                if "deep" in level:
                    lvl = "deep"
                elif "rem" in level:
                    lvl = "rem"
                elif "light" in level:
                    lvl = "light"
                elif "wake" in level or "awake" in level:
                    lvl = "wake"
                else:
                    # unknown -> treat as light (conservative)
                    lvl = "light"

                agg[key][lvl] += seconds
                agg[key]["total_seconds"] += seconds
                # if intensity present use it, otherwise fallback to our level weight
                w = intensity if not np.isnan(intensity) else level_weight(lvl)
                agg[key]["intensity_seconds_sum"] += w * seconds

    # Build dataframe rows
    rows = []
    for (pid, hour_start), vals in agg.items():
        total = vals["total_seconds"]
        if total == 0:
            wake_r = light_r = deep_r = rem_r = np.nan
            intensity = np.nan
            dominant = np.nan
        else:
            wake_r = vals["wake"] / total
            light_r = vals["light"] / total
            deep_r = vals["deep"] / total
            rem_r = vals["rem"] / total
            # normalize numerically to avoid tiny float rounding errors
            s = wake_r + light_r + deep_r + rem_r
            if s != 0 and not np.isnan(s):
                wake_r /= s
                light_r /= s
                deep_r /= s
                rem_r /= s
# -*- coding: utf-8 -*-
# Minimal script: read fitbit.bson, extract sleep-level data and write an hourly CSV
# Output: csv-sleep ratio-hourly.csv with columns
# pid, datetime, hour, sleep_state, wake_ratio, light_ratio, deep_ratio, rem_ratio, sleep_intensity

import os
from bson import decode_file_iter
import pandas as pd
import numpy as np
from datetime import datetime
from collections import defaultdict


def parse_level_point(pt):
    """Return (ts:datetime, seconds:int, level:str, intensity:float_or_nan) from a level point dict.
    Handles common field name variations gracefully.
    """
    # timestamp field
    ts = pt.get("dateTime") or pt.get("date_time") or pt.get("datetime") or pt.get("time")
    if ts is None:
        return None
    #!/usr/bin/env python3
    # -*- coding: utf-8 -*-
    # Sleep-only extraction: read fitbit.bson, compute hourly sleep ratios and save CSV

    import os
    from bson import decode_file_iter
    import pandas as pd
    import numpy as np
    from collections import defaultdict


    def parse_level_point(pt):
        """Return (ts:Timestamp, seconds:int, level:str, intensity:float_or_nan) from a level point dict.
        Handles common field name variations.
        """
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
            for doc in decode_file_iter(fh):
                payload = doc.get("data") if isinstance(doc.get("data"), dict) else doc
                typ = doc.get("type") or payload.get("type")
                if typ != "sleep":
                    continue

                pid = str(doc.get("id") or payload.get("id") or payload.get("userId") or payload.get("user_id") or "")

                sleep_obj = payload if doc.get("data") is None else payload
                levels = None
                if isinstance(sleep_obj, dict):
                    levels = sleep_obj.get("levels") or (sleep_obj.get("data") and sleep_obj.get("data").get("levels"))
                if not levels:
                    continue

                data_points = levels.get("data") or []
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

        rows = []
        for (pid, hour_start), vals in agg.items():
            total = vals["total_seconds"]
            if total == 0:
                wake_r = light_r = deep_r = rem_r = np.nan
                intensity = np.nan
                dominant = np.nan
            else:
                wake_r = vals["wake"] / total
                light_r = vals["light"] / total
                deep_r = vals["deep"] / total
                rem_r = vals["rem"] / total
                s = wake_r + light_r + deep_r + rem_r
                if s != 0 and not np.isnan(s):
                    wake_r /= s
                    light_r /= s
                    deep_r /= s
                    rem_r /= s
                intensity = vals["intensity_seconds_sum"] / total
                level_order = {"wake": vals["wake"], "light": vals["light"], "deep": vals["deep"], "rem": vals["rem"]}
                dominant = max(level_order, key=lambda k: level_order[k]) if any(v > 0 for v in level_order.values()) else np.nan

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
            })

        df_out = pd.DataFrame(rows, columns=["pid", "datetime", "hour", "sleep_state", "wake_ratio", "light_ratio", "deep_ratio", "rem_ratio", "sleep_intensity"])
        return df_out


    def main():
        out_csv = "csv-sleep ratio-hourly.csv"
        print("Processing fitbit.bson -> building hourly sleep ratios...")
        df = process_bson_sleep("fitbit.bson")
        print(f"Rows produced: {len(df)}. Saving to {out_csv}...")
        df.to_csv(out_csv, index=False)
        print("Done.")


    if __name__ == "__main__":
        main()
    # convert timestamp date object and then to datetime64[ns]
    users_calories["date"] = pd.to_datetime(users_calories["date"], format="%m/%d/%y %H:%M:%S").dt.date
    users_calories["date"] = pd.to_datetime(users_calories["date"], format="%Y/%m/%d")
    print("Date Conversion Completed")
    users_calories.to_pickle('data/users_calories.pkl')

users_calories.head()

users_calories.dtypes

# group by date and then take the average
users_calories.calories = users_calories.calories.astype(float)
users_calories = users_calories.groupby(['id', 'date']).sum()
users_calories.reset_index(drop=False, inplace=True)

users_calories.head()

# merge
df = df.merge(users_calories, how='outer', on=['id', 'date'])

df.head()

# checking for duplicates; if the two values are identical it means that all duplicates (in terms of date and ID) have also equal values.
print("Duplicates based on ID and Date: {}".format(df[df.duplicated(subset=['id','date'], keep=False)].shape[0]))
print("Duplicates based on ID and Date and Value: {}".format(df[df.duplicated(subset=['id','date', 'calories'], keep=False)].shape[0]))

"""### VO2 Max
VO2 Max is a measurement of how well your body uses oxygen when you’re working out at your hardest. It is widely accepted as the gold standard for grading cardiovascular fitness: the higher your VO2 Max, the more fit you are (source). This metric can also indicate performance potential for endurance-based activities including running, biking, and swimming (source).

VO2 Max is traditionally measured in a lab where you run on a treadmill or ride a stationary bike until exhaustion with a mask strapped to your nose and mouth to gauge the amount of air you inhale and exhale. While this method provides the most accurate measure of VO2 Max, your Fitbit device can estimate this value for you with less effort and discomfort.

Fitbit estimates your VO2 Max and refers to it as your cardio fitness score at a 2-day granularity (N=6364).
"""

users_vo2max = pd.DataFrame(columns=["id", "data"])
for user in users:
    user_data = pd.DataFrame(list(
        db.fitbit.find({"$and": [
            {"type": "demographic_vo2_max"},
            {"id": user}
        ]},
            {"id": 1, "data.dateTime": 1, "data.value.filteredDemographicVO2Max": 1, "_id": 0}
        )
    ))

    users_vo2max = pd.concat([users_vo2max, user_data], axis=0)

# split data column (json format) into two columns
users_vo2max["date"] = users_vo2max["data"].apply(lambda d: d["dateTime"])
users_vo2max["filteredDemographicVO2Max"] = users_vo2max["data"].apply(lambda d: d["value"].get("filteredDemographicVO2Max"))
users_vo2max.drop(["data"], inplace=True, axis=1)

# convert timestamp date object and then to datetime64[ns]
users_vo2max["date"] = pd.to_datetime(pd.to_datetime(users_vo2max["date"]).dt.date)

users_vo2max.head()

# merge
df = df.merge(users_vo2max, how='outer', on=['id', 'date'])

df.head()

# checking for duplicates; if the two values are identical it means that all duplicates (in terms of date and ID) have also equal values.
print("Duplicates based on ID and Date: {}".format(df[df.duplicated(subset=['id','date'], keep=False)].shape[0]))
print("Duplicates based on ID and Date and Value: {}".format(df[df.duplicated(subset=['id','date', 'filteredDemographicVO2Max'], keep=False)].shape[0]))

print("Size before duplicate elimination: {}".format(df.shape[0]))
df = df.groupby(["id", "date"]).first()
print("Size after duplicate elimination: {}".format(df.shape[0]))

"""### Distance
Distance walked by the user at 2-min granularity.
"""

if os.path.exists('data/users_distance.pkl'):
    print("Reading from pickle...")
    f = open("data/users_distance.pkl", "rb")
    # disable garbage collector
    gc.disable()
    # read pickle
    users_distance = pickle.load(f)
    # enable garbage collector again
    gc.enable()
    f.close()
else:
    users_distance = pd.DataFrame(columns=["id", "data"])
    for user in tqdm(users):
        user_data = pd.DataFrame(list(
            db.fitbit.find({"$and": [
                {"type": "distance"},
                {"id": user}
            ]},
                {"id": 1, "data.dateTime": 1, "data.value": 1, "_id": 0}
            )
        ))

        users_distance = pd.concat([users_distance, user_data], axis=0)

    print("Column Split...")
    # split data column (json format) into two columns
    users_distance.reset_index(drop=True, inplace=True)
    users_distance["date"] = users_distance["data"].swifter.apply(lambda d: d["dateTime"])
    users_distance["distance"] = users_distance["data"].swifter.apply(lambda d: d["value"])
    users_distance.drop(["data"], inplace=True, axis=1)
    print("Column Split Completed")
    print("Date Conversion...")
    # convert timestamp date object and then to datetime64[ns]
    users_distance["date"] = pd.to_datetime(users_distance["date"], infer_datetime_format=True).dt.date
    users_distance["date"] = pd.to_datetime(users_distance["date"], infer_datetime_format=True)
    print("Date Conversion Completed")
    users_distance.to_pickle('data/users_distance.pkl')

users_distance.head()

# group by date and then take the average
users_distance.distance = users_distance.distance.astype(float)
users_distance = users_distance.groupby(['id', 'date']).sum()
users_distance.reset_index(drop=False, inplace=True)
users_distance.distance = users_distance.distance / 100  # converts cm to m

users_distance.distance.hist()

# merge
df = df.merge(users_distance, how='outer', on=['id', 'date'])

df.head()

# checking for duplicates; if the two values are identical it means that all duplicates (in terms of date and ID) have also equal values.
print("Duplicates based on ID and Date: {}".format(df[df.duplicated(subset=['id','date'], keep=False)].shape[0]))
print("Duplicates based on ID and Date and Value: {}".format(df[df.duplicated(subset=['id','date', 'distance'], keep=False)].shape[0]))

"""### Exercise
The types of exercise the user has performed (N=5416). The names can be in multiple languages, hence we will use the exercise type code.
"""

users_exercise = pd.DataFrame(columns=["id", "data"])
for user in users:
    user_data = pd.DataFrame(list(
        db.fitbit.find({"$and": [
            {"type": "exercise"},
            {"id": user}
        ]},
            {"id": 1, "data.originalStartTime": 1, "data.activityTypeId": 1, "_id": 0}
        )
    ))

    users_exercise = pd.concat([users_exercise, user_data], axis=0)

# split data column (json format) into two columns
users_exercise["date"] = users_exercise["data"].apply(lambda d: d["originalStartTime"])
users_exercise["activityType"] = users_exercise["data"].apply(lambda d: d["activityTypeId"])
users_exercise.drop(["data"], inplace=True, axis=1)

# convert timestamp date object and then to datetime64[ns]
users_exercise["date"] = pd.to_datetime(pd.to_datetime(users_exercise["date"], infer_datetime_format=True).dt.date)

users_exercise.head()

users_exercise.dtypes

# Get distinct activity types
activity_types = pd.DataFrame(list(
        db.fitbit.find(
            {"type": "exercise"},
            {"id": 1, "data.activityTypeId": 1, "data.activityName": 1, "_id": 0}
        )
    ))
activity_types["activityTypeId"] = activity_types.data.apply(lambda d: d["activityTypeId"])
activity_types["activityName"] = activity_types.data.apply(lambda d: d["activityName"])
activity_types.drop(["data", "id"], inplace=True, axis=1)
activity_types = activity_types.drop_duplicates().reset_index(drop=True)
print(activity_types.head(50))
ACTIVITIES = {
    90013: "Walk",
    15000: "Sport",
    3001: "Aerobic Workout",
    52000: "Yoga/Pilates",
    90024: "Swim",
    90001: "Bike",
    20047: "Elliptical",
    2131: "Weights",
    55001: "Spinning",
    1071: "Bike",
    90009: "Run",
    20049: "Treadmill",
    53000: "Yoga/Pilates",
    55002: "Martial Arts",
    2040: "Circuit Training",
    2065: "Stairclimber",
    3000: "Workout",
    90012: "Hike",
    12339646: "Run",
    12350445: "Walk",
    23418750: "Swim",
    55003: "Bootcamp",
    15430: "Martial Arts",
    20057: "Interval Workout",
    15675: "Tennis",
    61980497: "Workout"
}

users_exercise["activityType"] = users_exercise["activityType"].apply(lambda a: ACTIVITIES.get(a))
users_exercise.head()

users_exercise = users_exercise.groupby(['id', 'date']).activityType.apply(list).reset_index(drop=False)

users_exercise.activityType.value_counts()

users_exercise.activityType = users_exercise.activityType.swifter.apply(lambda l: list(set(l)) if isinstance(l, list) else l)

users_exercise.activityType.value_counts()

# merge
df = df.merge(users_exercise, how='outer', on=['id', 'date'])

df.activityType.value_counts()

# checking for duplicates; if the two values are identical it means that all duplicates (in terms of date and ID) have also equal values.
print("Duplicates based on ID and Date: {}".format(df[df.duplicated(subset=['id','date'], keep=False)].shape[0]))
print("Duplicates based on ID and Date and Value: {}".format(df[df.astype(str).duplicated(subset=['id','date', 'activityType'], keep=False)].shape[0]))

df.to_pickle("data/temp_df_1.pkl")

"""### Heart Rate
User’s BPM at 10-sec granularity.
"""

if os.path.exists('data/temp_df_1.pkl'):
    print("Reading DataFrame from pickle...")
    f = open("data/temp_df_1.pkl", "rb")
    # disable garbage collector
    gc.disable()
    # read pickle
    df = pickle.load(f)
    # enable garbage collector again
    gc.enable()
    f.close()
    print("Reading completed.")

if os.path.exists('data/users_hr_daily.pkl'):
    print("Reading daily data from pickle...")
    f = open("data/users_hr_daily.pkl", "rb")
    # disable garbage collector
    gc.disable()
    # read pickle
    users_hr = pickle.load(f)
    # enable garbage collector again
    gc.enable()
    f.close()
    print("Reading completed.")
elif os.path.exists('data/users_hr.pkl'):
    print("Reading raw data from pickle...")
    f = open("data/users_hr.pkl", "rb")
    # disable garbage collector
    gc.disable()
    # read pickle
    users_hr = pickle.load(f)
    # enable garbage collector again
    gc.enable()
    f.close()
    print("Reading completed.")

    users_hr.bpm = users_hr.bpm.astype(float)
    users_hr = users_hr.groupby(['id', 'date']).mean()
    users_hr.reset_index(drop=False, inplace=True)
else:
    warnings.warn("\nTo read and aggregate heart rate data from MongoDB you need to ensure index existence for both query (type, id) and projection (data.dateTime, data.value.bpm) fields (compound index of four fields)...\n")
    rows = 0
    users_hr = pd.DataFrame(columns=["id", "date", "bpm"])
    for user in tqdm(users):
        user_data = pd.DataFrame(list(
            db.fitbit.find({"$and": [
                {"id": user},
                {"type": "heart_rate"}
            ]},
                {"id": 1, "data.dateTime": 1, "data.value.bpm": 1, "_id": 0}
            )
        ))

        # split data column (json format) into two columns
        # user_data.reset_index(drop=True, inplace=True)
        user_data["date"] = user_data["data"].swifter.apply(lambda d: d["dateTime"])
        user_data["bpm"] = user_data["data"].swifter.apply(lambda d: d["value"].get("bpm"))
        user_data.drop(["data"], inplace=True, axis=1)
        # convert timestamp date object and then to datetime64[ns]
        user_data["date"] = pd.to_datetime(pd.to_datetime(user_data["date"], infer_datetime_format="%Y-%m-%dT%H:%M:%S").dt.date, infer_datetime_format=True)
        # Additional code for counting heart rate instances
        # temp1 = user_data[(user_data['date'] >= "2021-05-24") & (user_data['date'] <= "2021-07-26")]  #Round1
        # temp2 = user_data[(user_data['date'] >= "2021-11-15") & (user_data['date'] <= "2022-01-17")]  #Round2
        # user_data = pd.concat([temp1, temp2])
        # rows += user_data.shape[0]
        # print("Updated rows: {}".format(rows))

        # group by date and then take the average
        user_data.bpm = user_data.bpm.astype(float)
        user_data = user_data.groupby(['id', 'date']).mean()
        user_data.reset_index(drop=False, inplace=True)

        users_hr = pd.concat([users_hr, user_data], axis=0)

    users_hr.to_pickle('data/users_hr_daily.pkl')
    # print("Final rows: {}".format(rows))

users_hr.head(20)

# convert timestamp date object and then to datetime64[ns]
users_hr["date"] = pd.to_datetime(pd.to_datetime(users_hr["date"], format="%Y/%m/%d  %H:%M:%S").dt.date, format="%Y/%m/%d")

users_hr.head()

# merge
df = df.merge(users_hr, how='outer', on=['id', 'date'])

df.head()

# checking for duplicates; if the two values are identical it means that all duplicates (in terms of date and ID) have also equal values.
print("Duplicates based on ID and Date: {}".format(df[df.duplicated(subset=['id','date'], keep=False)].shape[0]))
print("Duplicates based on ID and Date and Value: {}".format(df[df.duplicated(subset=['id','date', 'bpm'], keep=False)].shape[0]))

"""### Lightly Active Minutes
Light activity minutes (N=12086).
"""

users_active_minutes = pd.DataFrame(columns=["id", "data"])
for user in users:
    user_data = pd.DataFrame(list(
        db.fitbit.find({"$and": [
            {"type": "lightly_active_minutes"},
            {"id": user}
        ]},
            {"id": 1, "data.dateTime": 1, "data.value": 1, "_id": 0}
        )
    ))

    users_active_minutes = pd.concat([users_active_minutes, user_data], axis=0)

# split data column (json format) into two columns
users_active_minutes["date"] = users_active_minutes["data"].apply(lambda d: d["dateTime"])
users_active_minutes["lightly_active_minutes"] = users_active_minutes["data"].apply(lambda d: d["value"])
users_active_minutes.drop(["data"], inplace=True, axis=1)

# convert timestamp date object and then to datetime64[ns]
users_active_minutes["date"] = pd.to_datetime(pd.to_datetime(users_active_minutes["date"], format="%m/%d/%y %H:%M:%S").dt.date, format="%Y/%m/%d")

users_active_minutes.head()

# merge
df = df.merge(users_active_minutes, how='outer', on=['id', 'date'])

df.head(20)

# checking for duplicates; if the two values are identical it means that all duplicates (in terms of date and ID) have also equal values.
print("Duplicates based on ID and Date: {}".format(df[df.duplicated(subset=['id','date'], keep=False)].shape[0]))
print("Duplicates based on ID and Date and Value: {}".format(df[df.duplicated(subset=['id','date', 'lightly_active_minutes'], keep=False)].shape[0]))

# checking for duplicates; if the two values are identical it means that all duplicates (in terms of date and ID) have also equal values.
no_dups = df.astype(str).drop_duplicates()
no_dups[no_dups.astype(str).duplicated(keep = False, subset=['id','date'])]

# Keep max to avoid 0 values when non-zero values exist for the same date and user ID
print("Size before duplicate elimination: {}".format(df.shape[0]))
df = df.groupby(["id", "date"]).max()
df.reset_index(drop=False, inplace=True)
print("Size after duplicate elimination: {}".format(df.shape[0]))

df.head(300)

"""### Moderately Active Minutes
Moderate activity minutes (N=12086).
"""

users_active_minutes = pd.DataFrame(columns=["id", "data"])
for user in users:
    user_data = pd.DataFrame(list(
        db.fitbit.find({"$and": [
            {"type": "moderately_active_minutes"},
            {"id": user}
        ]},
            {"id": 1, "data.dateTime": 1, "data.value": 1, "_id": 0}
        )
    ))

    users_active_minutes = pd.concat([users_active_minutes, user_data], axis=0)

# split data column (json format) into two columns
users_active_minutes["date"] = users_active_minutes["data"].apply(lambda d: d["dateTime"])
users_active_minutes["moderately_active_minutes"] = users_active_minutes["data"].apply(lambda d: d["value"])
users_active_minutes.drop(["data"], inplace=True, axis=1)

# convert timestamp date object and then to datetime64[ns]
users_active_minutes["date"] = pd.to_datetime(pd.to_datetime(users_active_minutes["date"], format="%m/%d/%y %H:%M:%S").dt.date, format="%Y/%m/%d")

users_active_minutes.head()

# merge
df = df.merge(users_active_minutes, how='outer', on=['id', 'date'])

df.head()

# checking for duplicates; if the two values are identical it means that all duplicates (in terms of date and ID) have also equal values.
print("Duplicates based on ID and Date: {}".format(df[df.duplicated(subset=['id','date'], keep=False)].shape[0]))
print("Duplicates based on ID and Date and Value: {}".format(df[df.duplicated(subset=['id','date', 'moderately_active_minutes'], keep=False)].shape[0]))

# checking for duplicates; if the two values are identical it means that all duplicates (in terms of date and ID) have also equal values.
no_dups = df.astype(str).drop_duplicates()
no_dups[no_dups.astype(str).duplicated(keep = False, subset=['id','date'])]

print("Size before duplicate elimination: {}".format(df.shape[0]))
df = df.groupby(["id", "date"]).max()
df.reset_index(drop=False, inplace=True)
print("Size after duplicate elimination: {}".format(df.shape[0]))

"""### Very Active Minutes
Very activity minutes (N=12086).
"""

users_active_minutes = pd.DataFrame(columns=["id", "data"])
for user in users:
    user_data = pd.DataFrame(list(
        db.fitbit.find({"$and": [
            {"type": "very_active_minutes"},
            {"id": user}
        ]},
            {"id": 1, "data.dateTime": 1, "data.value": 1, "_id": 0}
        )
    ))

    users_active_minutes = pd.concat([users_active_minutes, user_data], axis=0)

# split data column (json format) into two columns
users_active_minutes["date"] = users_active_minutes["data"].apply(lambda d: d["dateTime"])
users_active_minutes["very_active_minutes"] = users_active_minutes["data"].apply(lambda d: d["value"])
users_active_minutes.drop(["data"], inplace=True, axis=1)

# convert timestamp date object and then to datetime64[ns]
users_active_minutes["date"] = pd.to_datetime(pd.to_datetime(users_active_minutes["date"], format="%m/%d/%y %H:%M:%S").dt.date, format="%Y/%m/%d")

users_active_minutes.head()

# merge
df = df.merge(users_active_minutes, how='outer', on=['id', 'date'])

df.head()

# checking for duplicates; if the two values are identical it means that all duplicates (in terms of date and ID) have also equal values.
print("Duplicates based on ID and Date: {}".format(df[df.duplicated(subset=['id','date'], keep=False)].shape[0]))
print("Duplicates based on ID and Date and Value: {}".format(df[df.duplicated(subset=['id','date', 'very_active_minutes'], keep=False)].shape[0]))

print("Size before duplicate elimination: {}".format(df.shape[0]))
df = df.groupby(["id", "date"]).max()
df.reset_index(drop=False, inplace=True)
print("Size after duplicate elimination: {}".format(df.shape[0]))

"""### Sedentary Minutes
Sedentary minutes (N=12086).
"""

users_active_minutes = pd.DataFrame(columns=["id", "data"])
for user in users:
    user_data = pd.DataFrame(list(
        db.fitbit.find({"$and": [
            {"type": "sedentary_minutes"},
            {"id": user}
        ]},
            {"id": 1, "data.dateTime": 1, "data.value": 1, "_id": 0}
        )
    ))

    users_active_minutes = pd.concat([users_active_minutes, user_data], axis=0)

# split data column (json format) into two columns
users_active_minutes["date"] = users_active_minutes["data"].apply(lambda d: d["dateTime"])
users_active_minutes["sedentary_minutes"] = users_active_minutes["data"].apply(lambda d: d["value"])
users_active_minutes.drop(["data"], inplace=True, axis=1)

# convert timestamp date object and then to datetime64[ns]
users_active_minutes["date"] = pd.to_datetime(pd.to_datetime(users_active_minutes["date"], format="%m/%d/%y %H:%M:%S").dt.date, format="%Y/%m/%d")

users_active_minutes.head()

# merge
df = df.merge(users_active_minutes, how='outer', on=['id', 'date'])

df.head()

df.to_pickle('data/temp_df_2.pkl')

df = pd.read_pickle('data/temp_df_2.pkl')

df.head(20)

# checking for duplicates; if the two values are identical it means that all duplicates (in terms of date and ID) have also equal values.
print("Duplicates based on ID and Date: {}".format(df[df.duplicated(subset=['id','date'], keep=False)].shape[0]))
print("Duplicates based on ID and Date and Value: {}".format(df[df.duplicated(subset=['id','date', 'sedentary_minutes'], keep=False)].shape[0]))

# checking for duplicates; if the two values are identical it means that all duplicates (in terms of date and ID) have also equal values.
no_dups = df.astype(str).drop_duplicates()
no_dups[no_dups.astype(str).duplicated(keep = False, subset=['id','date'])]

print("Size before duplicate elimination: {}".format(df.shape[0]))
df = df.groupby(["id", "date"]).max()
df.reset_index(drop=False, inplace=True)
print("Size after duplicate elimination: {}".format(df.shape[0]))

"""### Mindfulness Sessions
These data can help us understand if the user conducted an EDA session voluntarily (N=27785).
"""

def try_parsing_date(text):
    for fmt in ('%Y-%m-%dT%H:%M:%S%z', '%Y-%m-%dT%H:%M%z'):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            pass
    raise ValueError('no valid date format found for {}'.format(text))

users_mindfulness = pd.DataFrame(columns=["id", "data"])
for user in users:
    user_data = pd.DataFrame(list(
        db.fitbit.find({"$and": [
            {"type": "mindfulness_sessions"},
            {"id": user}
        ]},
            {"id": 1, "data.start_date_time": 1, "data.session_type": 1, "_id": 0}
        )
    ))

    users_mindfulness = pd.concat([users_mindfulness, user_data], axis=0)

# split data column (json format) into two columns
users_mindfulness["date"] = users_mindfulness["data"].apply(lambda d: d["start_date_time"])
users_mindfulness["activityType"] = users_mindfulness["data"].apply(lambda d: d["session_type"])
users_mindfulness.drop(["data"], inplace=True, axis=1)

# convert timestamp date object and then to datetime64[ns]
# users_mindfulness["date"] = pd.to_datetime(users_mindfulness["date"], format='%Y-%m-%dT%H:%M:%S%z')
users_mindfulness["date"] = pd.to_datetime(pd.to_datetime(users_mindfulness["date"], infer_datetime_format=True).dt.date)

users_mindfulness["mindfulness_session"] = True  # instead of storing the session type, only store if user engaged in session
users_mindfulness.drop(['activityType'], axis=1, inplace=True)
users_mindfulness = users_mindfulness.drop_duplicates()

users_mindfulness.head()

# merge
df = df.merge(users_mindfulness, how='outer', on=['id', 'date'])
df.mindfulness_session.fillna('False', inplace=True)

df.head(20)

# checking for duplicates; if the two values are identical it means that all duplicates (in terms of date and ID) have also equal values.
print("Duplicates based on ID and Date: {}".format(df[df.duplicated(subset=['id','date'], keep=False)].shape[0]))
print("Duplicates based on ID and Date and Value: {}".format(df[df.duplicated(subset=['id','date', 'mindfulness_session'], keep=False)].shape[0]))

"""### Mindfulness EDA Data Sessions
These entries contain the skin conductance level, which we can get on average for a day (i.e., as an indication of stress).
"""

users_eda = pd.DataFrame(columns=["id", "data"])
for user in users:
    user_data = pd.DataFrame(list(
        db.fitbit.find({"$and": [
            {"type": "mindfulness_eda_data_sessions"},
            {"id": user}
        ]},
            {"id": 1, "data.timestamp": 1, "data.scl_avg": 1, "_id": 0}
        )
    ))

    users_eda = pd.concat([users_eda, user_data], axis=0)

# split data column (json format) into two columns
users_eda["date"] = users_eda["data"].apply(lambda d: d["timestamp"])
users_eda["scl_avg"] = users_eda["data"].apply(lambda d: d["scl_avg"])
users_eda.drop(["data"], inplace=True, axis=1)

users_eda.reset_index(drop=True, inplace=True)
# convert timestamp date object and then to datetime64[ns]
users_eda["date"] = pd.to_datetime(pd.to_datetime(users_exercise["date"], infer_datetime_format=True).dt.date)
# users_eda["date"] = pd.to_datetime(users_eda["date"].swifter.apply(lambda ts: datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d')), format='%Y-%m-%d')

users_eda.head()

users_eda.dtypes

# group by date and then take the average
users_eda = users_eda.groupby(['id', 'date']).mean()
users_eda.reset_index(drop=False, inplace=True)

users_eda.head()

# merge
df = df.merge(users_eda, how='outer', on=['id', 'date'])

df.head(20)

# checking for duplicates; if the two values are identical it means that all duplicates (in terms of date and ID) have also equal values.
print("Duplicates based on ID and Date: {}".format(df[df.duplicated(subset=['id','date'], keep=False)].shape[0]))
print("Duplicates based on ID and Date and Value: {}".format(df[df.duplicated(subset=['id','date', 'scl_avg'], keep=False)].shape[0]))

"""### Resting Heart Rate
A normal resting heart rate for adults ranges from 60 to 100 beats per minute. Generally, a lower heart rate at rest implies more efficient heart function and better cardiovascular fitness.
"""

users_rhr = pd.DataFrame(columns=["id", "data"])
for user in users:
    user_data = pd.DataFrame(list(
        db.fitbit.find({"$and": [
            {"type": "resting_heart_rate"},
            {"id": user}
        ]},
            {"id": 1, "data.value.date": 1, "data.value.value": 1, "_id": 0}
        )
    ))

    users_rhr = pd.concat([users_rhr, user_data], axis=0)

# split data column (json format) into two columns
users_rhr["date"] = users_rhr["data"].apply(lambda d: d["value"].get("date"))
users_rhr["resting_hr"] = users_rhr["data"].apply(lambda d: d["value"].get("value"))
users_rhr.drop(["data"], inplace=True, axis=1)

# convert timestamp date object and then to datetime64[ns]
users_rhr["date"] = pd.to_datetime(users_rhr["date"], format="%m/%d/%y")

users_rhr.head(50)

users_rhr = users_rhr[users_rhr.resting_hr != 0.0]

users_rhr.head(20)

# merge
df = df.merge(users_rhr, how='outer', on=['id', 'date'])

df.head(20)

# checking for duplicates; if the two values are identical it means that all duplicates (in terms of date and ID) have also equal values.
print("Duplicates based on ID and Date: {}".format(df[df.duplicated(subset=['id','date'], keep=False)].shape[0]))
print("Duplicates based on ID and Date and Value: {}".format(df[df.duplicated(subset=['id','date', 'resting_hr'], keep=False)].shape[0]))

print("Size before duplicate elimination: {}".format(df.shape[0]))
df = df.groupby(["id", "date"]).first()
df.reset_index(drop=False, inplace=True)
print("Size after duplicate elimination: {}".format(df.shape[0]))

"""### Sleep
Details about a user’s sleep IF they wore their Fitbit to bed.
"""

def try_sleep_minutes(d, type):
    try:
        ratio = d["levels"].get("summary").get(type).get("minutes")/d["levels"].get("summary").get(type).get("thirtyDayAvgMinutes")
    except AttributeError:
        ratio = np.nan
    except ZeroDivisionError:
        ratio = np.nan

    return ratio

# todo Test if each user has one main sleep session per day
# Problem with {type:"sleep", id: ObjectId('621e301e67b776a240608a72'), "data.dateOfSleep": "2021-06-23"}

users_sleep = pd.DataFrame(columns=["id", "data"])
for user in users:
    user_data = pd.DataFrame(list(
        db.fitbit.find({"$and": [
            {"type": "sleep"},
            {"id": user},
            {"data.mainSleep": True}
        ]},
            {"id": 1, "data.dateOfSleep": 1, "data.duration": 1, "data.minutesToFallAsleep": 1, "data.minutesAsleep": 1, "data.minutesAwake": 1, "data.minutesAfterWakeup": 1, "data.efficiency": 1, "data.levels.summary.deep.minutes": 1, "data.levels.summary.deep.thirtyDayAvgMinutes": 1, "data.levels.summary.wake.minutes": 1, "data.levels.summary.wake.thirtyDayAvgMinutes": 1, "data.levels.summary.light.minutes": 1, "data.levels.summary.light.thirtyDayAvgMinutes": 1, "data.levels.summary.rem.minutes": 1, "data.levels.summary.rem.thirtyDayAvgMinutes": 1, "_id": 0}
        )
    ))

    users_sleep = pd.concat([users_sleep, user_data], axis=0)

# split data column (json format) into two columns
users_sleep["date"] = users_sleep["data"].apply(lambda d: d["dateOfSleep"])
users_sleep["sleep_duration"] = users_sleep["data"].apply(lambda d: d["duration"])
users_sleep["minutesToFallAsleep"] = users_sleep["data"].apply(lambda d: d["minutesToFallAsleep"])
users_sleep["minutesAsleep"] = users_sleep["data"].apply(lambda d: d["minutesAsleep"])
users_sleep["minutesAwake"] = users_sleep["data"].apply(lambda d: d["minutesAwake"])
users_sleep["minutesAfterWakeup"] = users_sleep["data"].apply(lambda d: d["minutesAfterWakeup"])
users_sleep["sleep_efficiency"] = users_sleep["data"].apply(lambda d: d["efficiency"])
users_sleep["sleep_deep_ratio"] = users_sleep["data"].apply(lambda d: try_sleep_minutes(d, "deep"))
users_sleep["sleep_wake_ratio"] = users_sleep["data"].apply(lambda d: try_sleep_minutes(d, "wake"))
users_sleep["sleep_light_ratio"] = users_sleep["data"].apply(lambda d: try_sleep_minutes(d, "light"))
users_sleep["sleep_rem_ratio"] = users_sleep["data"].apply(lambda d: try_sleep_minutes(d, "rem"))
users_sleep.drop(["data"], inplace=True, axis=1)

# convert timestamp date object and then to datetime64[ns]
users_sleep["date"] = pd.to_datetime(users_sleep["date"], infer_datetime_format=True)

users_sleep.head(50)

# merge
df = df.merge(users_sleep, how='outer', on=['id', 'date'])

df.head()

# checking for duplicates; if the two values are identical it means that all duplicates (in terms of date and ID) have also equal values.
print("Duplicates based on ID and Date: {}".format(df[df.duplicated(subset=['id','date', 'sleep_duration'], keep=False)].shape[0]))
print("Duplicates based on ID and Date and Value: {}".format(df[df.duplicated(subset=['id','date', 'sleep_rem_ratio'], keep=False)].shape[0]))

# checking for duplicates; if the two values are identical it means that all duplicates (in terms of date and ID) have also equal values.
no_dups = df.astype(str).drop_duplicates()
no_dups[no_dups.astype(str).duplicated(keep = False, subset=['id','date', 'sleep_duration'])]

print("Size before duplicate elimination: {}".format(df.shape[0]))
df = df.groupby(["id", "date"]).first()
df.reset_index(drop=False, inplace=True)
print("Size after duplicate elimination: {}".format(df.shape[0]))

# checking for duplicates; if the two values are identical it means that all duplicates (in terms of date and ID) have also equal values.
no_dups = df.astype(str).drop_duplicates()
no_dups[no_dups.astype(str).duplicated(keep = False, subset=['id','date', 'sleep_duration'])]

df.to_pickle('./data/temp_df_3.pkl')

df = pd.read_pickle('./data/temp_df_3.pkl')

"""### Steps
Getting steps from pre-computed file.
"""

# Loading the data of daily user steps
if not os.path.exists("data/users_steps_daily.pkl"):
    steps_dataframe = pd.DataFrame(columns=["date", "steps", "id"])
    for user in tqdm(users):
        user_dataframe = pd.DataFrame(list(
            db.fitbit.find(
                {"type": "steps",
                 "id": user},
                {"data.dateTime": 1, "data.value": 1, "id": 1, "_id": 0}
            )
        ))
        user_dataframe['date'] = user_dataframe['data'].swifter.progress_bar(False).apply(lambda d: d['dateTime'])
        user_dataframe['steps'] = user_dataframe['data'].swifter.progress_bar(False).apply(lambda d: d['value'])
        user_dataframe.drop(["data"], inplace=True, axis=1)

        # basic preprocessing for steps - transformations
        user_dataframe['steps'] = pd.to_numeric(user_dataframe['steps'])  # was string
        user_dataframe["date"] = pd.to_datetime(pd.to_datetime(user_dataframe["date"], infer_datetime_format=True).dt.date, infer_datetime_format=True)

        steps_dataframe = pd.concat([steps_dataframe, user_dataframe], axis=0, ignore_index=True)

    # group by date and then take the sum
    steps_dataframe = steps_dataframe.groupby(['id', 'date']).sum()
    steps_dataframe.reset_index(drop=False, inplace=True)
    steps_dataframe.to_pickle("data/users_steps_daily.pkl")

infile = open('data/users_steps_daily.pkl','rb')
steps_daily = pickle.load(infile)
infile.close()
# steps_daily = steps_daily[["date", "steps", "id"]]
steps_daily # year-month-day e.g. 2021-05-24

# merge
df = df.merge(steps_daily, how='outer', on=['id', 'date'])

df.head(20)

# checking for duplicates; if the two values are identical it means that all duplicates (in terms of date and ID) have also equal values.
print("Duplicates based on ID and Date: {}".format(df[df.duplicated(subset=['id','date'], keep=False)].shape[0]))
print("Duplicates based on ID and Date and Value: {}".format(df[df.duplicated(subset=['id','date', 'steps'], keep=False)].shape[0]))

print("Size before duplicate elimination: {}".format(df.shape[0]))
df = df.groupby(["id", "date"]).first()
df.reset_index(drop=False, inplace=True)
print("Size after duplicate elimination: {}".format(df.shape[0]))

df.steps.hist()

"""### Time in Heart Rate Zones
Minutes below zone, and in zones 1, 2, and 3.
"""

users_time_in_heart_rate_zones = pd.DataFrame(columns=["id", "data"])
for user in users:
    user_data = pd.DataFrame(list(
        db.fitbit.find({"$and": [
            {"type": "time_in_heart_rate_zones"},
            {"id": user}
        ]},
            {"id": 1, "data.dateTime": 1, "data.value": 1, "_id": 0}
        )
    ))

    users_time_in_heart_rate_zones = pd.concat([users_time_in_heart_rate_zones, user_data], axis=0)

# split data column (json format) into two columns
users_time_in_heart_rate_zones["date"] = users_time_in_heart_rate_zones["data"].apply(lambda d: d["dateTime"])
users_time_in_heart_rate_zones["minutes_in_default_zone_1"] = users_time_in_heart_rate_zones["data"].apply(lambda d: d["value"].get("valuesInZones").get("IN_DEFAULT_ZONE_1"))
users_time_in_heart_rate_zones["minutes_below_default_zone_1"] = users_time_in_heart_rate_zones["data"].apply(lambda d: d["value"].get("valuesInZones").get("BELOW_DEFAULT_ZONE_1"))
users_time_in_heart_rate_zones["minutes_in_default_zone_2"] = users_time_in_heart_rate_zones["data"].apply(lambda d: d["value"].get("valuesInZones").get("IN_DEFAULT_ZONE_2"))
users_time_in_heart_rate_zones["minutes_in_default_zone_3"] = users_time_in_heart_rate_zones["data"].apply(lambda d: d["value"].get("valuesInZones").get("IN_DEFAULT_ZONE_3"))
users_time_in_heart_rate_zones.drop(["data"], inplace=True, axis=1)

# convert timestamp date object and then to datetime64[ns]
users_time_in_heart_rate_zones["date"] = pd.to_datetime(pd.to_datetime(users_time_in_heart_rate_zones["date"], format="%m/%d/%y %H:%M:%S").dt.date, format="%Y/%m/%d")

users_time_in_heart_rate_zones.head(20)

# merge
df = df.merge(users_time_in_heart_rate_zones, how='outer', on=['id', 'date'])

df.head(20)

# checking for duplicates; if the two values are identical it means that all duplicates (in terms of date and ID) have also equal values.
print("Duplicates based on ID and Date: {}".format(df[df.duplicated(subset=['id','date'], keep=False)].shape[0]))
print("Duplicates based on ID and Date and Value: {}".format(df[df.duplicated(subset=['id','date', 'minutes_below_default_zone_1'], keep=False)].shape[0]))

df['minutes_below_default_zone_1'].hist()

df['minutes_in_default_zone_1'].hist()

df['minutes_in_default_zone_2'].hist()

df['minutes_in_default_zone_3'].hist()

"""### Profile
Demographic Information about the user (N=70).
"""

def get_age(date_of_birth):
    today = date.today()
    return today.year - date_of_birth.year - ((today.month, today.day) < (date_of_birth.month, date_of_birth.day))

users_profiles = pd.DataFrame(columns=["id", "data"])
for user in users:
    user_data = pd.DataFrame(list(
        db.fitbit.find({"$and": [
            {"type": "Profile"},
            {"id": user}
        ]},
            # {"id": 1, "data.date_of_birth": 1, "data.gender": 1, "data.height": 1, "data.weight": 1,  "_id": 0}
            {"id": 1, "data.age": 1, "data.gender": 1, "data.bmi": 1,  "_id": 0}
        )
    ))

    users_profiles = pd.concat([users_profiles, user_data], axis=0)

# users_profiles["age"] = users_profiles["data"].apply(lambda d: d["date_of_birth"])
# users_profiles["age"] = pd.to_datetime(users_profiles["age"], format="%Y/%m/%d")
# users_profiles["age"] = users_profiles["age"].apply(lambda d: get_age(d))
# users_profiles["height"] = users_profiles["data"].apply(lambda d: d["height"])
# users_profiles["weight"] = users_profiles["data"].apply(lambda d: d["weight"])
# users_profiles["gender"] = users_profiles["data"].apply(lambda d: d["gender"])


users_profiles["age"] = users_profiles["data"].apply(lambda d: d["age"] if "age" in d else np.NaN)
users_profiles["gender"] = users_profiles["data"].apply(lambda d: d["gender"] if "gender" in d else np.NaN)
users_profiles["bmi"] = users_profiles["data"].apply(lambda d: d["bmi"] if "bmi" in d else np.NaN)

users_profiles.drop(['data'], axis=1, inplace=True)
users_profiles.head(70)

# merge
df = df.merge(users_profiles, how='left', on=['id'])

df.head(100)

"""### Saving to pickle file"""

df.to_pickle('./data/daily_fitbit_df_unprocessed.pkl')

df = pd.read_pickle('./data/daily_fitbit_df_unprocessed.pkl')

"""## Integrating SEMA Data
The SEMA ecological ecological momentary assessment surveys include data about the users' daily step goals (maximum once per day) as well as emotions and locations (maximum three times a day)

### Integrating step goals
This field includes the user's daily step goal range (N=1921).
"""

# user goals in SEMA
SEMA_GOALS_TO_MIN = {
    "2000": 0,
    "4999": 2000,
    "7999": 5000,
    "9999": 8000,
    "14999": 10000,
    "19999": 15000,
    "24999": 20000,
    "25000": 25000,
    "NO_GOAL": 0
}

SEMA_GOALS_TO_MAX = {
    "2000": 2000,
    "4999": 5000,
    "7999": 8000,
    "9999": 10000,
    "14999": 15000,
    "19999": 20000,
    "24999": 25000,
    "25000": 30000,
    "NO_GOAL": 0
}

SEMA_LABELS = {
    "2000": "Less than 2000",
    "4999": "2000-4999",
    "7999": "5000-7999",
    "9999": "8000-9999",
    "14999": "10000-14999",
    "19999": "15000-19999",
    "24999": "20000-24999",
    "25000": "More than 25000",
    "NO_GOAL": "No Goal",
    np.nan: "No Goal",
    None: "No Goal"
}

if not os.path.exists("data/users_step_goals_daily.pkl"):
    # Get user self-reported goals from SEMA
    sema_goals = pd.DataFrame(columns=["_id", "user_id", "data"])
    for user in users:
        user_data = pd.DataFrame(list(
                        db.sema.find({ "$and": [
                            { "data.STEPS": { "$ne": "<no-response>" } },
                            {"user_id": user}
                        ] },
                            {"data.STEPS": 1, "id": 1, "user_id": 1, "data.STARTED_TS": 1}
                        )
                    ))

        sema_goals = pd.concat([sema_goals, user_data], axis=0)

    # split data column (json format) into two columns
    sema_goals["timestamp"] = sema_goals["data"].apply(lambda d: d["STARTED_TS"])
    sema_goals["step_goal"] = sema_goals["data"].apply(lambda d: d["STEPS"])
    sema_goals.drop(["data", "_id"], inplace=True, axis=1)

    # convert timestamp to day format
    sema_goals["date"] = pd.to_datetime(sema_goals["timestamp"], infer_datetime_format=True).dt.date
    sema_goals["date"] = pd.to_datetime(sema_goals["date"], infer_datetime_format=True) # convert from object to datetime
    sema_goals.drop(["timestamp"], inplace=True, axis=1)

    # add min goal and max goal columns
    sema_goals['min_goal'] = sema_goals.step_goal.apply(lambda s: SEMA_GOALS_TO_MIN.get(s))
    sema_goals['max_goal'] = sema_goals.step_goal.apply(lambda s: SEMA_GOALS_TO_MAX.get(s))

    # add goal labels
    sema_goals['step_goal_label'] = sema_goals['step_goal'].apply(lambda v: SEMA_LABELS[v])
    sema_goals[['date', 'user_id', 'step_goal', 'min_goal', 'max_goal', 'step_goal_label']].to_pickle('./data/users_step_goals_daily.pkl')

users_step_goals = pd.read_pickle('./data/users_step_goals_daily.pkl')
users_step_goals['id'] = users_step_goals.user_id.copy()
users_step_goals.drop(['user_id'], axis=1, inplace=True)

users_step_goals.head(10)

# merge
df = df.merge(users_step_goals, how='outer', on=['id', 'date'])

df.head(50)

# checking for duplicates; if the two values are identical it means that all duplicates (in terms of date and ID) have also equal values.
print("Duplicates based on ID and Date: {}".format(df[df.duplicated(subset=['id','date'], keep=False)].shape[0]))
print("Duplicates based on ID and Date and Value: {}".format(df[df.duplicated(subset=['id','date', 'step_goal_label'], keep=False)].shape[0]))

# checking for duplicates; if the two values are identical it means that all duplicates (in terms of date and ID) have also equal values.
no_dups = df.astype(str).drop_duplicates()
no_dups[no_dups.astype(str).duplicated(keep = False, subset=['id','date'])]

print("Size before duplicate elimination: {}".format(df.shape[0]))
df = df.groupby(["id", "date"]).first()
df.reset_index(drop=False, inplace=True)
print("Size after duplicate elimination: {}".format(df.shape[0]))

"""### Integrating Εmotions & Location"""

# read data
if not os.path.exists('./data/sema_mood_place.pkl'):
    # Get user self-reported goals from SEMA
    users = db.sema.distinct('user_id')

    sema_mood = pd.DataFrame(columns=["_id", "user_id", "data"])
    for user in users:
        user_data = pd.DataFrame(list(
                        db.sema.find({
                            "$or": [
                                {
                                    "$and": [
                                        { "data.MOOD": { "$ne": "<no-response>" } },
                                        {"data.MOOD": { "$ne": None }},
                                        {"user_id": user}
                                    ]
                                },
                                {
                                    "$and": [
                                        { "data.PLACE": { "$ne": "<no-response>" } },
                                        {"data.PLACE": { "$ne": None }},
                                        {"user_id": user}
                                    ]
                                }
                            ]
                        },
                            {"data.MOOD": 1, "data.PLACE": 1, "id": 1, "_id": 0, "user_id": 1, "data.STARTED_TS": 1}
                        )
                    ))

        sema_mood = pd.concat([sema_mood, user_data], axis=0)

sema_mood["date"] = pd.to_datetime(pd.to_datetime(sema_mood["data"].apply(lambda d: d["STARTED_TS"]), infer_datetime_format=True).dt.date, infer_datetime_format=True)
sema_mood["time"] = pd.to_datetime(sema_mood["data"].apply(lambda d: d["STARTED_TS"]), infer_datetime_format=True).dt.time
sema_mood["data.MOOD"] = sema_mood["data"].apply(lambda d: d["MOOD"])
sema_mood["data.PLACE"] = sema_mood["data"].apply(lambda d: d["PLACE"])
sema_mood.drop(["_id", "data"], axis=1, inplace=True)
sema_mood.to_pickle('./data/sema_mood_place.pkl')

infile = open('./data/sema_mood_place.pkl','rb')
sema = pickle.load(infile)
infile.close()

sema.head() # year-month-day e.g. 2021-05-24

# process dateTime to date
# sema['date'] = pd.to_datetime(sema['Dates'])
# sema.drop(['data.CREATED_TS', 'Dates'], axis=1, inplace=True)

# replace not common moods
sema['data.MOOD'] = sema['data.MOOD'].apply(lambda mood: 'SAD' if mood == 'SADNESS' else ('HAPPY' if mood == 'JOY' else mood))
sema = sema[(sema['data.MOOD'] != 'FEAR') & (sema['data.MOOD'] != 'SURPRISE') & (sema['data.MOOD'] != 'ANGER') & (sema['data.MOOD'] != '<no-response>')]
print(sema['data.MOOD'].value_counts())
print(sema['data.PLACE'].value_counts())
# drop unessecary columns
sema.drop(['time'], axis=1, inplace=True)

sema.head()

sema_moods = pd.get_dummies(sema['data.MOOD'])
sema_places = pd.get_dummies(sema['data.PLACE'])
print(sema_moods.head())
print(sema_places.head())

# combine one-hot encoding with actual df
sema = pd.concat([sema, sema_moods, sema_places], axis=1)
sema.drop(['data.MOOD', 'data.PLACE'], axis=1, inplace=True)
sema.head(50)

sema_grouped = sema.groupby(['date', 'user_id']).max()
sema_grouped.reset_index(drop=False, inplace=True)
sema_grouped['id'] = sema_grouped['user_id'].copy()
sema_grouped.drop(['user_id'], axis=1, inplace=True)
sema_grouped.reset_index(drop=True, inplace=True)
sema_grouped.id = sema_grouped.id.swifter.apply(lambda id: ObjectId(id))
sema_grouped.head()

sema_grouped.date.hist()
plt.xticks(rotation=90)

# merge
df = df.merge(sema_grouped, how='outer', on=['id', 'date'])

df.head(50)

# checking for duplicates; if the two values are identical it means that all duplicates (in terms of date and ID) have also equal values.
print("Duplicates based on ID and Date: {}".format(df[df.duplicated(subset=['id','date'], keep=False)].shape[0]))
print("Duplicates based on ID and Date and Value: {}".format(df[df.duplicated(subset=['id','date', 'full_sleep_breathing_rate'], keep=False)].shape[0]))

df.to_pickle('./data/daily_fitbit_sema_df_unprocessed.pkl')

df = pd.read_pickle('./data/Daily Anonymized Files/daily_fitbit_sema_df_unprocessed.pkl')

df.shape[0]

df.to_csv('./data/daily_fitbit_sema_df_unprocessed.csv')

"""## Integrating Surveys Data
During the experiment the participants completed the following surveys:
* State-Trait Anxiety Inventory (STAI): 323 weekly responses
* Positive Affect Negative Affect Scale (PANAS): 311 weekly responses
* Physical Activity Readiness Questionnaire (PAR-Q): 58 pre-experiment responses
* Behavioral Regulation in Exercise Questionnaire (BREQ-2): 101 pre- and post-experiment responses
* Big Five Personality Test: 53 pre-experiment responses
* TTM Stages and Processes of Behavior Change: 104 pre- and post-experiment responses
* Demographics Questionnaire: 63 pre-experiment responses

"""

df = pd.read_pickle('./data/daily_fitbit_sema_df_unprocessed.pkl')

df.head(50)

"""### Integrating Personality Type Responses"""

users_personality = pd.DataFrame(columns=["user_id", "data"])
for user in users:
    user_data = pd.DataFrame(list(
        db.surveys.find({"$and": [
            {"type": "bfpt"},
            {"user_id": user}
        ]},
            {"_id": 0}
        )
    ))

    users_personality = pd.concat([users_personality, user_data], axis=0)

users_personality["submitdate"] = users_personality["data"].apply(lambda d: d["submitdate"])
users_personality["submitdate"] = pd.to_datetime(users_personality["submitdate"], infer_datetime_format=True).dt.date

for ipip in range(1,51):
    if ipip < 10:
        col_name = "ipip[SQ00{}]".format(ipip)
    else:
        col_name = "ipip[SQ0{}]".format(ipip)
    users_personality[col_name] = users_personality["data"].apply(lambda d: d[col_name])

users_personality.drop(['data'], axis=1, inplace=True)

print("We have {} survey responses for the IPIP (Big-5 Personality) scale.".format(users_personality.shape[0]))

users_personality.drop_duplicates(subset=['user_id'], keep="first")
print("We have {} survey responses for the IPIP (Big-5 Personality) scale after duplicates elimination.".format(users_personality.shape[0]))

bfpt_scoring = pd.read_csv("data/utils/BFPT-Coding.csv", sep=";")
bfpt_scoring.head()

ipip_plus = bfpt_scoring[bfpt_scoring.plus == True].code
ipip_minus = bfpt_scoring[bfpt_scoring.plus == False].code
ipip_minus

# Converting IPIP Item Responses to Scale Scores
# For + keyed items, the response "Very Inaccurate" is assigned a value of 1, "Moderately Inaccurate" a value of 2, "Neither Inaccurate nor Accurate" a 3, "Moderately Accurate" a 4, and "Very Accurate" a value of 5.
# For - keyed items, the response "Very Inaccurate" is assigned a value of 5, "Moderately Inaccurate" a value of 4, "Neither Inaccurate nor Accurate" a 3, "Moderately Accurate" a 2, and "Very Accurate" a value of 1.
def inverse_score(score, min, max):
    return max - score + min

# def inverse_score(score):
#     if score == 1:
#         return 5
#     if score == 2:
#         return 4
#     if score == 3:
#         return 3
#     if score == 4:
#         return 2
#     return 1

print("Users' personality responses before inverting minus keyed items:")
print(users_personality.iloc[:2,3:5].head())

for col in users_personality.columns:
    # inversing scores for the minus keyed items; the plus keyed items stay as is
    if col in ipip_minus.values:
        users_personality[col] = users_personality[col].apply(lambda score: inverse_score(score, 1, 5))

print("Users' personality responses after inverting minus keyed items:")
print(users_personality.iloc[:2,3:5].head())

ipip_extraversion = bfpt_scoring[bfpt_scoring.factor == 1].code
ipip_agreeableness = bfpt_scoring[bfpt_scoring.factor == 2].code
ipip_conscientiousness = bfpt_scoring[bfpt_scoring.factor == 3].code
ipip_stability = bfpt_scoring[bfpt_scoring.factor == 4].code
ipip_intellect = bfpt_scoring[bfpt_scoring.factor == 5].code

# find a summary per factor per user
# Factor I (Surgency or Extraversion)
users_personality["extraversion"] = users_personality[ipip_extraversion].sum(axis=1)
# Factor II (Agreeableness)
users_personality["agreeableness"] = users_personality[ipip_agreeableness].sum(axis=1)
# Factor III (Conscientiousness)
users_personality["conscientiousness"] = users_personality[ipip_conscientiousness].sum(axis=1)
# Factor IV (Emotional Stability)
users_personality["stability"] = users_personality[ipip_stability].sum(axis=1)
# Factor V (Intellect or Imagination)
users_personality["intellect"] = users_personality[ipip_intellect].sum(axis=1)

"""#### Factor I (Surgency or Extraversion)"""

users_personality.extraversion.hist()

print(users_personality.describe().extraversion)

"""#### Factor II (Agreeableness)"""

print(users_personality.describe().agreeableness)

users_personality.agreeableness.hist()

"""#### Factor III (Conscientiousness)"""

print(users_personality.describe().conscientiousness)

users_personality.conscientiousness.hist()

"""#### Factor IV (Emotional Stability)"""

print(users_personality.describe().stability)

users_personality.stability.hist()

"""#### Factor V (Intellect or Imagination)"""

print(users_personality.describe().intellect)

users_personality.intellect.hist()

"""To interpret individuals' scores, one might calculate the mean and standard deviation (SD) for a sample of persons, usually of the same sex and a particular age range, and interpret scores within one-half SD of the mean as "average." Scores outside that range can be interpreted as "low" or "high." If the scores are normally distributed, this would result in approximately 38% of persons being classified as average, about 31% as low, and 31% as high."""

# incorporate gender information
users_personality = users_personality.merge(df[['id','gender']].drop_duplicates(), how='left', left_on='user_id', right_on='id')
users_personality

users_personality.drop_duplicates(subset='id', inplace=True)
users_personality.gender.fillna('FEMALE', inplace=True)
users_personality.drop(['id'], inplace=True, axis=1)

MEAN_1_FEMALE, STD_1_FEMALE = users_personality.groupby('gender').extraversion.mean()["FEMALE"], users_personality.groupby('gender').extraversion.std()["FEMALE"]
MEAN_1_MALE, STD_1_MALE = users_personality.groupby('gender').extraversion.mean()["MALE"], users_personality.groupby('gender').extraversion.std()["MALE"]

MEAN_2_FEMALE, STD_2_FEMALE = users_personality.groupby('gender').agreeableness.mean()["FEMALE"], users_personality.groupby('gender').agreeableness.std()["FEMALE"]
MEAN_2_MALE, STD_2_MALE = users_personality.groupby('gender').agreeableness.mean()["MALE"], users_personality.groupby('gender').agreeableness.std()["MALE"]

MEAN_3_FEMALE, STD_3_FEMALE = users_personality.groupby('gender').conscientiousness.mean()["FEMALE"], users_personality.groupby('gender').conscientiousness.std()["FEMALE"]
MEAN_3_MALE, STD_3_MALE = users_personality.groupby('gender').conscientiousness.mean()["MALE"], users_personality.groupby('gender').conscientiousness.std()["MALE"]

MEAN_4_FEMALE, STD_4_FEMALE = users_personality.groupby('gender').stability.mean()["FEMALE"], users_personality.groupby('gender').stability.std()["FEMALE"]
MEAN_4_MALE, STD_4_MALE = users_personality.groupby('gender').stability.mean()["MALE"], users_personality.groupby('gender').stability.std()["MALE"]

MEAN_5_FEMALE, STD_5_FEMALE = users_personality.groupby('gender').intellect.mean()["FEMALE"], users_personality.groupby('gender').intellect.std()["FEMALE"]
MEAN_5_MALE, STD_5_MALE = users_personality.groupby('gender').intellect.mean()["MALE"], users_personality.groupby('gender').intellect.std()["MALE"]

def get_personality_category(score, mean, std):
    if score > mean + 0.5*std:
        return 'HIGH'
    if score < mean - 0.5*std:
        return 'LOW'
    return 'AVERAGE'

users_personality['ipip_extraversion_category'] = users_personality.apply(lambda row: get_personality_category(row.extraversion, MEAN_1_MALE, STD_1_MALE) if row.gender == "MALE" else get_personality_category(row.extraversion, MEAN_1_FEMALE, STD_1_FEMALE), axis=1)

users_personality['ipip_agreeableness_category'] = users_personality.apply(lambda row: get_personality_category(row.agreeableness, MEAN_2_MALE, STD_2_MALE) if row.gender == "MALE" else get_personality_category(row.agreeableness, MEAN_2_FEMALE, STD_2_FEMALE), axis=1)

users_personality['ipip_conscientiousness_category'] = users_personality.apply(lambda row: get_personality_category(row.conscientiousness, MEAN_3_MALE, STD_3_MALE) if row.gender == "MALE" else get_personality_category(row.conscientiousness, MEAN_3_FEMALE, STD_3_FEMALE), axis=1)

users_personality['ipip_stability_category'] = users_personality.apply(lambda row: get_personality_category(row.stability, MEAN_4_MALE, STD_4_MALE) if row.gender == "MALE" else get_personality_category(row.stability, MEAN_4_FEMALE, STD_4_FEMALE), axis=1)

users_personality['ipip_intellect_category'] = users_personality.apply(lambda row: get_personality_category(row.intellect, MEAN_5_MALE, STD_5_MALE) if row.gender == "MALE" else get_personality_category(row.intellect, MEAN_5_FEMALE, STD_5_FEMALE), axis=1)

users_personality.head(10)

print(users_personality[['gender', 'ipip_extraversion_category']].value_counts())
print(users_personality[['gender', 'ipip_agreeableness_category']].value_counts())
print(users_personality[['gender', 'ipip_conscientiousness_category']].value_counts())
print(users_personality[['gender', 'ipip_stability_category']].value_counts())
print(users_personality[['gender', 'ipip_intellect_category']].value_counts())

users_personality = users_personality[users_personality.columns.drop(list(users_personality.filter(regex='ipip\[SQ')))]
users_personality

if not os.path.exists("data/surveys"):
    os.makedirs("data/surveys")

users_personality.to_csv("data/surveys/personality.csv")
users_personality.to_pickle("data/surveys/personality.pkl")

"""### Integrating BREQ-2 Type Responses"""

users_breq = pd.DataFrame(columns=["user_id", "data"])
for user in users:
    user_data = pd.DataFrame(list(
        db.surveys.find({"$and": [
            {"type": "breq"},
            {"user_id": user}
        ]},
            {"_id": 0}
        )
    ))

    users_breq = pd.concat([users_breq, user_data], axis=0)

users_breq["submitdate"] = users_breq["data"].apply(lambda d: d["submitdate"])
users_breq["submitdate"] = pd.to_datetime(users_breq["submitdate"], infer_datetime_format=True).dt.date

for engage in range(1,20):
    if engage < 10:
        col_name = "engage[SQ00{}]".format(engage)
    else:
        col_name = "engage[SQ0{}]".format(engage)
    users_breq[col_name] = users_breq["data"].apply(lambda d: d[col_name])

users_breq.drop(['data'], axis=1, inplace=True)

print("We have {} survey responses for the BREQ-2 scale.".format(users_breq.shape[0]))

users_breq

# find a mean per factor per user
users_breq["breq_amotivation"] = users_breq[["engage[SQ005]", "engage[SQ009]", "engage[SQ012]", "engage[SQ019]"]].mean(axis=1)
users_breq["breq_external_regulation"] = users_breq[["engage[SQ001]", "engage[SQ006]", "engage[SQ011]", "engage[SQ016]"]].mean(axis=1)
users_breq["breq_introjected_regulation"] = users_breq[["engage[SQ002]", "engage[SQ007]", "engage[SQ013]"]].mean(axis=1)
users_breq["breq_identified_regulation"] = users_breq[["engage[SQ003]", "engage[SQ008]", "engage[SQ014]", "engage[SQ017]"]].mean(axis=1)
users_breq["breq_intrinsic_regulation"] = users_breq[["engage[SQ004]", "engage[SQ010]", "engage[SQ015]", "engage[SQ018]"]].mean(axis=1)
users_breq

temp = users_breq[["breq_amotivation", "breq_external_regulation", "breq_introjected_regulation", "breq_identified_regulation", "breq_intrinsic_regulation"]].agg(['idxmax','max'], axis=1).mask(lambda x: x['max'].eq(0))
users_breq.loc[:, "breq_self_determination"] = temp.loc[:, "idxmax"]
users_breq["breq_self_determination"].replace("breq_", "", inplace=True, regex=True)
users_breq

users_breq.drop_duplicates(subset=["user_id", "submitdate"], keep="last", inplace=True)
print("We have {} survey responses for the BREQ-2 scale after duplicate elimination.".format(users_breq.shape[0]))

users_breq["breq_self_determination"].hist()
plt.xticks(rotation=90)

users_breq = users_breq[users_breq.columns.drop(list(users_breq.filter(regex='engage')))]
users_breq

users_breq.to_csv("data/surveys/breq.csv")
users_breq.to_pickle("data/surveys/breq.pkl")

"""### Integrating TTM Responses"""

users_ttm = pd.DataFrame(columns=["user_id", "data"])
for user in users:
    user_data = pd.DataFrame(list(
        db.surveys.find({"$and": [
            {"type": "ttmspbf"},
            {"user_id": user}
        ]},
            {"_id": 0}
        )
    ))

    users_ttm = pd.concat([users_ttm, user_data], axis=0)

users_ttm["submitdate"] = users_ttm["data"].apply(lambda d: d["submitdate"])
users_ttm["stage"] = users_ttm["data"].apply(lambda d: d["stage"])
users_ttm["submitdate"] = pd.to_datetime(users_ttm["submitdate"], infer_datetime_format=True).dt.date

for engage in range(2,32):
    if engage < 10:
        col_name = "processes[SQ00{}]".format(engage)
        new_col_name = "processes[SQ00{}]".format(engage-1)
    else:
        col_name = "processes[SQ0{}]".format(engage)
        if engage == 10:
            new_col_name = "processes[SQ00{}]".format(engage-1)
        else:
            new_col_name = "processes[SQ0{}]".format(engage-1)
    users_ttm[new_col_name] = users_ttm["data"].apply(lambda d: d[col_name])

users_ttm.drop(['data'], axis=1, inplace=True)

print("We have {} survey responses for the TTM and Processes of Change scale.".format(users_ttm.shape[0]))

users_ttm

def define_stage_of_change(response):
    if response == "No, and I do not intend to do regular physical activity in the next 6 months.":
        return "Precontemplation"
    if response == "No, but I intend to do regular physical activity in the next 6 months.":
        return "Contemplation"
    if response == "No, but I intend to do regular physical activity in the next 30 days.":
        return "Preparation"
    if response == "Yes, I have been doing physical activity regularly, but for less than 6 months.":
        return "Action"
    return "Maintenance"

users_ttm["stage"] = users_ttm.stage.apply(lambda response: define_stage_of_change(response))
users_ttm

users_ttm.drop_duplicates(subset=["user_id", "submitdate"], keep="last", inplace=True)
print("We have {} survey responses for the TTM and Processes of Change scale after duplicate elimination.".format(users_ttm.shape[0]))

users_ttm.stage.hist()

users_ttm["ttm_consciousness_raising"] = users_ttm[["processes[SQ001]","processes[SQ011]", "processes[SQ021]"]].mean(axis=1)
users_ttm["ttm_dramatic_relief"] = users_ttm[["processes[SQ002]","processes[SQ012]", "processes[SQ022]"]].mean(axis=1)
users_ttm["ttm_environmental_reevaluation"] = users_ttm[["processes[SQ003]","processes[SQ013]", "processes[SQ023]"]].mean(axis=1)
users_ttm["ttm_self_reevaluation"] = users_ttm[["processes[SQ004]","processes[SQ014]", "processes[SQ024]"]].mean(axis=1)
users_ttm["ttm_social_liberation"] = users_ttm[["processes[SQ005]","processes[SQ015]", "processes[SQ025]"]].mean(axis=1)
users_ttm["ttm_counterconditioning"] = users_ttm[["processes[SQ006]","processes[SQ016]", "processes[SQ026]"]].mean(axis=1)
users_ttm["ttm_helping_relationships"] = users_ttm[["processes[SQ007]","processes[SQ017]", "processes[SQ027]"]].mean(axis=1)
users_ttm["ttm_reinforcement_management"] = users_ttm[["processes[SQ008]","processes[SQ018]", "processes[SQ028]"]].mean(axis=1)
users_ttm["ttm_self_liberation"] = users_ttm[["processes[SQ009]","processes[SQ019]", "processes[SQ029]"]].mean(axis=1)
users_ttm["ttm_stimulus_control"] = users_ttm[["processes[SQ010]","processes[SQ020]", "processes[SQ030]"]].mean(axis=1)

users_ttm = users_ttm[users_ttm.columns.drop(list(users_ttm.filter(regex='processes')))]
users_ttm

users_ttm.to_csv("data/surveys/ttm.csv")
users_ttm.to_pickle("data/surveys/ttm.pkl")

"""### Integrating STAI Data"""

users_stai = pd.DataFrame(columns=["user_id", "data"])
for user in users:
    user_data = pd.DataFrame(list(
        db.surveys.find({"$and": [
            {"type": "stai"},
            {"user_id": user}
        ]},
            {"_id": 0}
        )
    ))

    users_stai = pd.concat([users_stai, user_data], axis=0)

users_stai["submitdate"] = users_stai["data"].apply(lambda d: d["submitdate"])
users_stai["submitdate"] = pd.to_datetime(users_stai["submitdate"], infer_datetime_format=True).dt.date

for engage in range(1,21):
    if engage < 10:
        col_name = "STAI[SQ00{}]".format(engage)
    else:
        col_name = "STAI[SQ0{}]".format(engage)
    users_stai[col_name] = users_stai["data"].apply(lambda d: d[col_name])

users_stai.drop(['data'], axis=1, inplace=True)

print("We have {} survey responses for the STAI (Y1-form) scale.".format(users_stai.shape[0]))

users_stai.head()

# convert 5-likert to 4-likert scale (mistakenly the STAI scale was distributed with a 5-likert, but it's originally 4-likert)
def convert_5_to_4_likert(x):
    return (4 - 1) * (x - 1) / (5 - 1) + 1

users_stai.iloc[:, 3:] = users_stai.iloc[:, 3:].apply(lambda x: convert_5_to_4_likert(x))
users_stai

def proper_round(num, dec=0):
    num = str(num)[:str(num).index('.')+dec+2]
    if num[-1]>='5':
      a = num[:-2-(not dec)]       # integer part
      b = int(num[-2-(not dec)])+1 # decimal part
      return float(a)+b**(-dec+1) if a and b == 10 else float(a+str(b))
    return float(num[:-1])

for col in users_stai.iloc[:, 3:].columns:
    users_stai[col] = users_stai[col].apply(lambda x: proper_round(x))
users_stai

# Based on the scoring document: https://oml.eular.org/sysModules/obxOML/docs/id_150/State-Trait-Anxiety-Inventory.pdf, some questions are reversed in STAI
stai_reversed = ["STAI[SQ001]", "STAI[SQ002]", "STAI[SQ005]", "STAI[SQ008]", "STAI[SQ010]", "STAI[SQ011]", "STAI[SQ015]", "STAI[SQ016]", "STAI[SQ019]", "STAI[SQ020]"]
for col in users_personality.columns:
    # inversing scores for the minus keyed items; the plus keyed items stay as is
    if col in stai_reversed:
        users_stai[col] = users_stai[col].apply(lambda score: inverse_score(score, 1, 4))
users_stai

# to calculate the total stress score simply sum per row
users_stai['stai_stress'] = users_stai.iloc[:, 3:].sum(axis=1)
users_stai

"""Score interpretation.
Range of scores for each subtest is 20–80, the higher score indicating greater anxiety. A cut point of 39–40 has been suggested to detect clinically significant symptoms for the S-Anxiety scale (9, 10); however, other studies have suggested a higher cut score of 54–55 for older adults (11). Normative values are available in the manual (12) for adults, college students, and psychiatric samples. To this author's knowledge, no cut scores have been validated for rheumatic disease populations.

[https://onlinelibrary.wiley.com/doi/full/10.1002/acr.20561](Source)

**Warning:** The above assumes that both S-STAI and T-STAI were administered, but we administered only the S-STAI.
"""

users_stai['stai_stress'].hist()

mean_stai = users_stai['stai_stress'].mean()
std_stai = users_stai['stai_stress'].std()

def get_stai_category(score):
    if score < mean_stai-0.5*std_stai:
        return "Below average"
    if score > mean_stai+0.5*std_stai:
        return "Above average"
    return "Average"

users_stai['stai_stress_category'] = users_stai['stai_stress'].apply(lambda score: get_stai_category(score))
users_stai

users_stai.drop_duplicates(subset=["user_id", "submitdate"], inplace=True, keep="last")
print("We have {} survey responses for the STAI (Y1-form) scale after duplicates elimination.".format(users_stai.shape[0]))

users_stai.stai_stress_category.hist()

print("Average value on the S-STAI scale: {} - Standard deviation on the S-STAI scale: {}".format(mean_stai, std_stai))

users_stai = users_stai[users_stai.columns.drop(list(users_stai.filter(regex='STAI\[SQ')))]
users_stai

# test for random user
users_stai.sort_values(["user_id", "submitdate"], inplace=True)
users_stai[users_stai.user_id == ObjectId("621e2eaf67b776a2406b14ac")].plot("submitdate", "stai_stress")
plt.xticks(rotation=90)

users_stai.to_csv("data/surveys/stai.csv")
users_stai.to_pickle("data/surveys/stai.pkl")

"""### Integrate PANAS scale"""

users_panas = pd.DataFrame(columns=["user_id", "data"])
for user in users:
    user_data = pd.DataFrame(list(
        db.surveys.find({"$and": [
            {"type": "panas"},
            {"user_id": user}
        ]},
            {"_id": 0}
        )
    ))

    users_panas = pd.concat([users_panas, user_data], axis=0)

users_panas["submitdate"] = users_panas["data"].apply(lambda d: d["submitdate"])
users_panas["submitdate"] = pd.to_datetime(users_panas["submitdate"], infer_datetime_format=True).dt.date

for p in range(1,21):
    if p < 10:
        col_name = "P1[SQ00{}]".format(p)
    else:
        col_name = "P1[SQ0{}]".format(p)
    users_panas[col_name] = users_panas["data"].apply(lambda d: d[col_name])

users_panas.drop(['data'], axis=1, inplace=True)

print("We have {} survey responses for the STAI (Y1-form) scale.".format(users_panas.shape[0]))
users_panas.drop_duplicates(subset=["user_id", "submitdate"], inplace=True, keep="last")
print("We have {} survey responses for the STAI (Y1-form) scale after duplicates elimination.".format(users_panas.shape[0]))

users_panas

positive = ["P1[SQ001]", "P1[SQ003]", "P1[SQ005]", "P1[SQ009]", "P1[SQ010]", "P1[SQ012]", "P1[SQ014]", "P1[SQ016]", "P1[SQ017]", "P1[SQ019]"]
negative = ["P1[SQ002]", "P1[SQ004]", "P1[SQ006]", "P1[SQ007]", "P1[SQ008]", "P1[SQ011]", "P1[SQ013]", "P1[SQ015]", "P1[SQ018]", "P1[SQ020]"]
users_panas["positive_affect_score"] = users_panas[positive].sum(axis=1)
users_panas["negative_affect_score"] = users_panas[negative].sum(axis=1)
users_panas

print("Mean positive affect score is {} (SD={}) compared to original sample mean of 33.3 (SD=7.2)".format(users_panas["positive_affect_score"].mean(), users_panas["positive_affect_score"].std()))
print("Mean negative affect score is {} (SD={}) compared to original sample mean of 17.4 (SD=6.2)".format(users_panas["negative_affect_score"].mean(), users_panas["negative_affect_score"].std()))
print("Scores can range from 10 – 50, with higher scores representing higher levels of positive or negative affect, respectively.")

users_panas["positive_affect_score"].describe()

users_panas["negative_affect_score"].describe()

users_panas = users_panas[users_panas.columns.drop(list(users_panas.filter(regex='P1\[SQ')))]
users_panas

users_panas.to_csv("data/surveys/panas.csv")
users_panas.to_pickle("data/surveys/panas.pkl")

