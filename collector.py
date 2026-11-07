# collector.py
import json
import numpy as np
import pandas as pd
import config


def load_raw(path):
    """Load the single CSV. Coerces Date; strips station names."""
    df = pd.read_csv(path, skip_blank_lines=True, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    df[config.DATE_COL] = pd.to_datetime(df[config.DATE_COL], errors="coerce")
    df["Station"] = df["Station"].astype(str).str.strip().str.upper()
    df = df[df["Station"].isin(config.STATIONS)].copy()
    return df


def _mad_flag(series, k=config.MAD_K):
    """|x - median| > k * 1.4826 * MAD. NaN-safe."""
    s = series.dropna()
    if len(s) < 5:
        return pd.Series(False, index=series.index)
    med = s.median()
    mad = (s - med).abs().median()
    if mad == 0:
        return pd.Series(False, index=series.index)
    return ((series - med).abs() > k * 1.4826 * mad).fillna(False)


def quality_control(df):
    """Return (clean_df, audit_log). Never raises on bad values."""
    log = []
    out = []
    for station, g in df.groupby("Station", sort=False):
        g = g.copy()

        # 1) placeholder detection (Min == Max)
        for col in ["Max", "Min"]:
            if col in g and g[col].notna().any():
                eq = (g["Max"] == g["Min"]).mean()
                if eq > config.PLACEHOLDER_RATIO:
                    n = int(g[col].notna().sum())
                    g[col] = np.nan
                    log.append({"station": station, "column": col,
                                "rule": "placeholder_min_eq_max",
                                "n_changed": n,
                                "details": f"Min==Max in {eq:.0%} of rows"})

        # 2) non-numeric -> NaN
        for col in ["Precipitation", "Max", "Min"]:
            if col in g:
                before = g[col].copy()
                g[col] = pd.to_numeric(g[col], errors="coerce")
                g[col] = g[col].replace([np.inf, -np.inf], np.nan)
                bad = before.notna() & g[col].isna()
                if bad.any():
                    log.append({"station": station, "column": col,
                                "rule": "non_numeric_to_nan",
                                "n_changed": int(bad.sum()), "details": ""})

        # 3) physical plausibility
        for col, lo, hi in [
            ("Precipitation", config.MIN_PLAUSIBLE_PRECIP, config.MAX_PLAUSIBLE_PRECIP),
            ("Max", config.MIN_PLAUSIBLE_TEMP, config.MAX_PLAUSIBLE_TEMP),
            ("Min", config.MIN_PLAUSIBLE_TEMP, config.MAX_PLAUSIBLE_TEMP),
        ]:
            if col in g:
                bad = g[col].notna() & ((g[col] < lo) | (g[col] > hi))
                if bad.any():
                    g.loc[bad, col] = np.nan
                    log.append({"station": station, "column": col,
                                "rule": "range_plausibility",
                                "n_changed": int(bad.sum()),
                                "details": f"outside [{lo}, {hi}]"})

        # 4) robust MAD outliers
        for col in ["Precipitation", "Max", "Min"]:
            if col in g and g[col].notna().sum() >= 5:
                mask = _mad_flag(g[col])
                if mask.any():
                    g.loc[mask, col] = np.nan
                    log.append({"station": station, "column": col,
                                "rule": "mad_outlier",
                                "n_changed": int(mask.sum()),
                                "details": f"k={config.MAD_K}"})

        # 5) Min > Max
        if {"Min", "Max"} <= set(g):
            bad = g["Min"].notna() & g["Max"].notna() & (g["Min"] > g["Max"])
            if bad.any():
                g.loc[bad, ["Min", "Max"]] = np.nan
                log.append({"station": station, "column": "Min/Max",
                            "rule": "min_gt_max",
                            "n_changed": int(bad.sum()), "details": ""})

        # 6) target must be 0/1
        if config.TARGET_COL in g:
            t = pd.to_numeric(g[config.TARGET_COL], errors="coerce")
            bad = t.notna() & ~t.isin([0, 1])
            if bad.any():
                log.append({"station": station, "column": config.TARGET_COL,
                            "rule": "target_not_binary",
                            "n_changed": int(bad.sum()), "details": ""})
            g[config.TARGET_COL] = t.where(t.isin([0, 1]))

        out.append(g)

    clean = pd.concat(out, ignore_index=True)
    clean = clean.sort_values(["Station", config.DATE_COL]).reset_index(drop=True)
    return clean, log


def save_audit(log, path="qc_log.json"):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2)
