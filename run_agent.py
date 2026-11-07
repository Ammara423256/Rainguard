# run_agent.py
import argparse
import config
import collector
import rules
import report


def main(csv_path):
    print(f"Loading {csv_path} ...")
    raw = collector.load_raw(csv_path)
    print(f"  raw rows: {len(raw):,}")
    print(raw.groupby("Station").size().to_string())

    clean, log = collector.quality_control(raw)
    collector.save_audit(log, "qc_log.json")
    print(f"\nQC applied. Clean rows: {len(clean):,}")
    for e in log:
        print(f"   {e['station']:<7} {e['column']:<13} {e['rule']:<24} "
              f"n={e['n_changed']:<5} {e['details']}")
    clean.to_csv("clean.csv", index=False)

    feat = rules.make_features(clean)
    train, test = rules.split_time(feat)
    print(f"\ntrain rows={len(train):,}  test rows={len(test):,}")

    pooled = rules.fit_pooled(train)
    per = rules.fit_per_station(train)

    # predict the most recent row per station
    latest = (feat.dropna(subset=[c for c in rules.FEATURES if c != "station_cat"])
              .sort_values(config.DATE_COL)
              .groupby("Station").tail(1))
    station_probs = {}
    for _, row in latest.iterrows():
        st = row["Station"]
        X = row[rules.FEATURES].to_frame().T
        station_probs[st] = float(pooled.predict_proba(X)[0, 1])

    status = report.write_status(station_probs, "status.json")
    report.write_chart(station_probs, "chart.png")
    print("\nAlert tiers:")
    for st, info in status["stations"].items():
        print(f"   {st:<7} p={info['prob']:.3f}  {info['tier']:<8} "
              f"-> {info['action']}")
    print("\nWrote status.json, chart.png, qc_log.json, clean.csv")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="Data_Input (2).csv")
    main(p.parse_args().csv)
