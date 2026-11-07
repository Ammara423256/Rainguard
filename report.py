# report.py
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config


def tier_for(prob):
    """Map a probability to the highest tier whose min is <= prob."""
    chosen = "Normal"
    for name, spec in config.ALERT_TIERS.items():
        if prob >= spec["min"]:
            chosen = name
    return chosen


def write_status(station_probs, path="status.json"):
    """station_probs: dict {station: probability}."""
    out = {"stations": {}}
    for st, p in station_probs.items():
        t = tier_for(p)
        out["stations"][st] = {
            "prob": round(float(p), 4),
            "tier": t,
            "action": config.ALERT_TIERS[t]["action"],
        }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    return out


def write_chart(station_probs, path="chart.png"):
    stations = list(station_probs)
    probs = [station_probs[s] for s in stations]
    colors = []
    for p in probs:
        t = tier_for(p)
        colors.append({"Normal": "#4caf50", "Watch": "#ffc107",
                       "Warning": "#ff9800", "Heavy": "#f44336"}[t])
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(stations, probs, color=colors)
    ax.set_ylim(0, 1)
    ax.set_ylabel("P(rain tomorrow)")
    ax.set_title("Rain-Guard daily risk")
    for i, p in enumerate(probs):
        ax.text(i, p + 0.02, f"{p:.2f}", ha="center")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
