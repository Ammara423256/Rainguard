# Rain-Guard — Daily Rainfall-Risk Agent

## 1. Setup

One command:

    python run_agent.py --csv "Data_Input (2).csv"

Produces:

- `status.json` — current tier per station
- `chart.png`   — one bar chart of P(rain tomorrow) per station, coloured by tier
- `qc_log.json` — every QC change (station, column, rule, count)
- `clean.csv`   — data after QC

Tests:

    python test_agent.py     # 6/6 pass

Seed: `8456` (`SEED` in `config.py`). Used for all `RandomForestClassifier(random_state=...)` calls and any random split.

## 2. Stations

- **MABAD** — moderate rainfall regime
- **MANGLA** — placeholder `Min == Max` era in 1971 (flagged and nulled by QC)
- **MURREE** — placeholder `Min == Max` throughout the raw file (flagged and nulled by QC)

The `Min`/`Max` columns for MANGLA (1971) and MURREE (all years) are **placeholders**: `Min == Max` in >90% of rows. `config.PLACEHOLDER_RATIO = 0.90` triggers nulling of those columns and a `placeholder_min_eq_max` entry in `qc_log.json`.

## 3. Perceive & QC (collector.py)

- Loads the single CSV, coerces dates (`errors="coerce"`), strips station names.
- Six QC rules, **all thresholds in `config.py`**: placeholder, non-numeric, range plausibility, robust MAD outlier (k = 5, 1.4826 factor), Min > Max, target not 0/1.
- Logs every change to `qc_log.json` (station, column, rule, count, details).
- **Never raises** on bad values — NaN-safe throughout.

## 4. Reason (rules.py)

### 4.1 Forecast — Random Forest

Features per station: `Precipitation`, `Max`, `Min` at lags 1–3, a 3-day rolling mean, month, day-of-year, station code. Label = `Raintomorrow`. Time split: **train < 2004, test ≥ 2004** (5,964 test rows).

| Model | F1 (2004+) |
|---|---|
| Pooled (all 3 stations) | **0.510** |
| MABAD per-station | 0.582 |
| MANGLA per-station | 0.165 |
| MURREE per-station | 0.564 |

**Which generalises better and why:** the pooled model is **more robust** — it is immune to per-station collapse. MANGLA's per-station F1 falls to 0.17 because its own training rows include the placeholder Min/Max era; the pooled model borrows clean signal from MABAD and MURREE and never collapses. On clean stations the per-station model is slightly better, but the pooled model is the deployment choice.

### 4.2 Left-out-station test

Train on 2 stations, predict the third (held out entirely, never seen in training):

| Held out | Pooled F1 | n_test |
|---|---|---|
| MABAD | 0.221 | 1,951 |
| MANGLA | 0.413 | 2,039 |
| MURREE | 0.418 | 1,974 |

MABAD is the hardest to transfer to — its rainfall regime differs most. **MANGLA held out (0.41) is far better than MANGLA's own per-station model (0.17)**, confirming MANGLA's placeholder training data is the problem, not the station's signal. The pooled model generalises, but not equally — spatial heterogeneity matters.

### 4.3 Trend — Mann–Kendall + Sen's slope

**What is tested:** whether annual total precipitation has a monotonic trend over 1971–2012. Null: no trend.

| Station | n_years | Kendall τ | p-value | Sen slope (mm/yr) |
|---|---|---|---|---|
| MABAD | 42 | +0.066 | 0.537 | +2.98 |
| MANGLA | 42 | −0.143 | 0.183 | −2.90 |
| MURREE | 42 | −0.108 | 0.314 | −4.84 |

**Result:** no station shows a significant trend at α = 0.05. MABAD trends slightly up, MANGLA/MURREE slightly down, but all p > 0.05.

**Assumption of MK:** observations are independent — no autocorrelation between years. Annual aggregation usually satisfies this, but multi-year cycles (ENSO) can violate it.

**Caveat of running on three stations at once:** family-wise false-positive rate at α = 0.05 across 3 tests is ≈ 14% (1 − 0.95³), not 5%. A Bonferroni correction (α_adj = 0.0167) leaves all three still non-significant. We report uncorrected p-values and note this caveat.

## 5. Act (report.py)

Alert tiers in `config.py`, thresholds set from the 60th / 80th / 95th percentiles of pooled-RF predicted probability on the **training set** (1971–2003):

| Tier | Min prob | Action |
|---|---|---|
| Normal | 0.00 | No action. Log and monitor. |
| Watch | 0.31 | Notify duty officer; pre-stage sandbags. |
| Warning | 0.50 | Alert local response team; check drains. |
| Heavy | 0.69 | **Escalate to district authority (HUMAN APPROVAL required).** |

Writes `status.json` (per-station tier + probability + action) and `chart.png`.

## 6. Evaluation and guardrails (test_agent.py)

Six automated tests, all passing:

1. `test_mad_flags_outlier` — a realistic outlier is flagged, normal values are not.
2. `test_placeholder_min_eq_max_nulls_column` — MURREE-pattern data triggers the placeholder rule.
3. `test_never_crashes_on_bad_values` — strings, NaNs and out-of-range target do not raise.
4. **`test_no_label_leakage`** — asserts `precip_lag1` at row t equals raw precip at t−1, and no `lead*` column exists. This test would fail if a future-row feature were introduced.
5. `test_placeholder_temperature_handled` — a `-999` temperature is nulled.
6. `test_tier_mapping` — probability → tier mapping is correct.

### Alert scorecard — Heavy tier, 2005–2012 (n = 5,323, positives = 1,473)

| Config | Threshold | TP | FP | FN | TN | Hit | FA | Miss |
|---|---|---|---|---|---|---|---|---|
| Raw RF | 0.69 | 148 | 14 | 1,325 | 3,836 | 10.0% | 0.4% | 90.0% |
| Calibrated | 0.80 | 190 | 25 | 1,283 | 3,825 | 12.9% | 0.6% | 87.1% |

The Heavy tier is high-precision / low-recall: it catches only the model's most confident rain days. **This is deliberate** — see §7.

### Human-approval action

The Heavy-tier action ("Escalate to district authority") requires human approval. Justification: (1) Heavy alerts are rare (~5% of days) and high-impact; (2) the cost is asymmetric — missing a real flood is worse than a false alarm, yet the scorecard shows the tier is not reliable enough to act alone; (3) automation bias means operators over-trust automated recommendations, so irreversible decisions must remain with a human. The agent does continuous surveillance; the irreversible decision stays with a human.

## 7. AI-use note

Portions of this project (code scaffolding, README structure) were drafted with an AI assistant. All model choices, thresholds, and interpretations were reviewed and verified against the data. Every number in this README is reproducible by running `python run_agent.py --csv "Data_Input (2).csv"` and `python test_agent.py`.