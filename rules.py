# rules.py
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, theilslopes
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
import config

FEATURES = ["precip_lag1", "precip_lag2", "precip_lag3",
            "max_lag1", "max_lag2", "max_lag3",
            "min_lag1", "min_lag2", "min_lag3",
            "precip_roll3", "month", "dayofyear", "station_cat"]


def make_features(df):
    df = df.sort_values(["Station", config.DATE_COL]).copy()
    df["month"] = df[config.DATE_COL].dt.month
    df["dayofyear"] = df[config.DATE_COL].dt.dayofyear

    g = df.groupby("Station", group_keys=False)
    for lag in (1, 2, 3):
        df[f"precip_lag{lag}"] = g["Precipitation"].shift(lag)
        df[f"max_lag{lag}"] = g["Max"].shift(lag)
        df[f"min_lag{lag}"] = g["Min"].shift(lag)
    df["precip_roll3"] = g["Precipitation"].transform(
        lambda s: s.shift(1).rolling(3).mean())
    df["station_cat"] = df["Station"].astype("category").cat.codes
    return df


def split_time(df, cutoff="2004-01-01"):
    train = df[df[config.DATE_COL] < cutoff].dropna(
        subset=FEATURES + [config.TARGET_COL])
    test = df[df[config.DATE_COL] >= cutoff].dropna(
        subset=FEATURES + [config.TARGET_COL])
    return train, test


def fit_pooled(train):
    m = RandomForestClassifier(**config.MODEL_PARAMS)
    m.fit(train[FEATURES], train[config.TARGET_COL])
    return m


def fit_per_station(train):
    models = {}
    for st, g in train.groupby("Station"):
        m = RandomForestClassifier(**config.MODEL_PARAMS)
        m.fit(g[FEATURES], g[config.TARGET_COL])
        models[st] = m
    return models


def left_out_test(df, held_out, cutoff="2004-01-01"):
    tr = df[(df[config.DATE_COL] < cutoff) & (df["Station"] != held_out)]
    te = df[(df[config.DATE_COL] >= cutoff) & (df["Station"] == held_out)]
    tr = tr.dropna(subset=FEATURES + [config.TARGET_COL])
    te = te.dropna(subset=FEATURES + [config.TARGET_COL])
    if len(tr) == 0 or len(te) == 0:
        return {"held_out": held_out, "pooled_f1": float("nan"),
                "per_station_f1": float("nan"), "n_test": len(te)}
    pooled = RandomForestClassifier(**config.MODEL_PARAMS).fit(
        tr[FEATURES], tr[config.TARGET_COL])
    per = RandomForestClassifier(**config.MODEL_PARAMS).fit(
        tr[FEATURES], tr[config.TARGET_COL])
    return {
        "held_out": held_out,
        "pooled_f1": f1_score(te[config.TARGET_COL], pooled.predict(te[FEATURES])),
        "per_station_f1": f1_score(te[config.TARGET_COL], per.predict(te[FEATURES])),
        "n_test": len(te),
    }


def mann_kendall(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) < 3:
        return {"tau": np.nan, "p": np.nan}
    tau, p = kendalltau(np.arange(len(x)), x)
    return {"tau": float(tau), "p": float(p)}


def sens_slope(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) < 3:
        return {"slope": np.nan, "lo": np.nan, "hi": np.nan}
    slope, intercept, lo, hi = theilslopes(x)
    return {"slope": float(slope), "lo": float(lo), "hi": float(hi)}


def annual_trend(df, station):
    g = df[df["Station"] == station].copy()
    g["year"] = g[config.DATE_COL].dt.year
    annual = g.groupby("year")["Precipitation"].sum(min_count=1).dropna()
    return {"mk": mann_kendall(annual.values),
            "sen": sens_slope(annual.values),
            "n_years": int(len(annual))}
