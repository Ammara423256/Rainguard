# test_agent.py
import numpy as np
import pandas as pd
import config
import collector
import rules
import report


def test_mad_flags_outlier():
    # realistic spread so MAD > 0
    s = pd.Series([10, 11, 10, 12, 11, 10, 1000])
    mask = collector._mad_flag(s, k=config.MAD_K)
    assert mask.iloc[-1], "1000 should be flagged as outlier"
    assert not mask.iloc[:-1].any(), "normal values should not be flagged"
    
def test_placeholder_min_eq_max_nulls_column():
    df = pd.DataFrame({
        "Date": pd.date_range("2000-01-01", periods=20),
        "Station": "MURREE",
        "Precipitation": 0.0,
        "Max": 58.0,
        "Min": 58.0,
        "Raintomorrow": 0,
    })
    clean, log = collector.quality_control(df)
    assert clean["Max"].isna().all(), "Max should be nulled"
    rules_seen = {e["rule"] for e in log}
    assert "placeholder_min_eq_max" in rules_seen

def test_never_crashes_on_bad_values():
    df = pd.DataFrame({
        "Date": ["2000-01-01", "garbage", "2000-01-03"],
        "Station": "MABAD",
        "Precipitation": ["abc", None, 5],
        "Max": [10, 12, 11],
        "Min": [1, 2, 3],
        "Raintomorrow": [0, 1, 2],
    })
    clean, log = collector.quality_control(df)          # must not raise
    assert len(clean) == 3
    # 2 should have become NaN; remaining values must be 0 or 1
    valid = clean["Raintomorrow"].dropna()
    assert valid.isin([0, 1]).all()
    assert clean["Raintomorrow"].isna().sum() == 1


def test_no_label_leakage():
    raw = pd.DataFrame({
        "Date": pd.date_range("2000-01-01", periods=5),
        "Station": "MABAD",
        "Precipitation": [1, 2, 3, 4, 5],
        "Max": [10, 10, 10, 10, 10],
        "Min": [5, 5, 5, 5, 5],
        "Raintomorrow": [0, 0, 0, 0, 0],
    })
    feat = rules.make_features(raw)
    # lag1 at row t must equal raw precip at t-1
    assert np.isnan(feat["precip_lag1"].iloc[0])
    assert feat["precip_lag1"].iloc[1] == 1
    assert feat["precip_lag1"].iloc[2] == 2
    # no future-derived column
    assert not any("lead" in c for c in feat.columns)


def test_placeholder_temperature_handled():
    df = pd.DataFrame({
        "Date": pd.date_range("2000-01-01", periods=10),
        "Station": "MABAD",
        "Precipitation": 0.0,
        "Max": [-999] * 10,
        "Min": [-999] * 10,
        "Raintomorrow": 0,
    })
    clean, _ = collector.quality_control(df)
    assert clean["Max"].isna().all()
    assert clean["Min"].isna().all()


def test_tier_mapping():
    assert report.tier_for(0.10) == "Normal"
    assert report.tier_for(0.60) == "Watch"
    assert report.tier_for(0.80) == "Warning"
    assert report.tier_for(0.99) == "Heavy"


if __name__ == "__main__":
    import traceback
    tests = [v for k, v in list(globals().items())
             if k.startswith("test_") and callable(v)]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
            passed += 1
        except Exception:
            print(f"FAIL {t.__name__}")
            traceback.print_exc()
    print(f"\n{passed}/{len(tests)} passed")
