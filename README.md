# MAG Energy Solutions — FTR Opportunity Selector

A data-driven algorithm for identifying profitable Financial Transmission Right (FTR) opportunities in electricity markets, built for the CSD × MAG Energy Solutions Data Challenge 2026.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the selector (produces opportunities.csv)
python main.py --start-month 2020-01 --end-month 2023-12

# 3. (Optional) Run offline backtesting on development data
python backtest.py --start-month 2020-06 --end-month 2023-12
```

---

## Project Structure

```
mag_energy/
├── main.py              # Entry point — produces opportunities.csv
├── backtest.py          # Offline evaluation on historical data
├── requirements.txt
├── README.md
├── opportunities.csv    # Generated output
├── data/
│   ├── costs/           # Monthly exposure costs (Parquet)
│   ├── prices/          # Realized hourly prices (Parquet)
│   ├── sim_monthly/     # Monthly simulations — 3 scenarios (Parquet)
│   └── sim_daily/       # Daily simulations — 3 scenarios (Parquet)
└── src/
    ├── data_loader.py   # I/O layer — year-by-year Parquet loading
    ├── profitability.py # Computes PR_o, C_o, Profit(o) = PR_o - C_o
    ├── feature_builder.py # Feature engineering from all data sources
    └── selector.py      # Scoring + selection (10–100 per month)
```

---

## Approach

### Problem Framing

At the **7th day of month M**, we must select 10–100 opportunities `(EID, M+1, PEAKID)` likely to satisfy `PR_o - C_o > 0` once month M+1 concludes.

Since fewer than 5% of opportunities are historically profitable, the core challenge is **precision-recall trade-off under severe class imbalance**.

### Step 1 — Historical Profitability ("Answer Key")

`profitability.py` computes the ground truth for past months:

```
PR_o = Σ PRICEREALIZED  (hours in month, matching PEAKID)
Profit(o) = PR_o - C_o
```

Missing prices and costs are treated as **zero** (implicit zero rule). This gives us historical win rates per `(EID, PEAKID)` — the base signal for forward selection.

### Step 2 — Feature Engineering

`feature_builder.py` constructs a feature matrix at decision time using **only legally available data**:

| Feature group | Source | Signal |
|---|---|---|
| `sm_PSM_mean` | Monthly sims (M+1) | Forward simulated price — primary predictive signal |
| `sm_ACTIVATIONLEVEL_mean` | Monthly sims (M+1) | Opportunity intensity |
| `sm_PSM_scenario_std` | Monthly sims (M+1) | Scenario disagreement (uncertainty proxy) |
| `sd_PSD_mean` | Daily sims (days 1–7 of M) | Short-term refined forecast |
| `hist_profit_rate_6m` | Past realized data (≤ M) | Fraction of last 6 months profitable |
| `hist_mean_profit_6m` | Past realized data (≤ M) | Average historical profit magnitude |

### Step 3 — Scoring & Selection

`selector.py` min-max normalizes each feature and computes a **weighted composite score**:

```
score = 0.35 × sim_price  +  0.25 × activation  +  0.20 × hist_rate
      + 0.15 × hist_profit  +  0.05 × scenario_agreement
```

Top-K candidates (default K=50) are selected, clamped to [10, 100].

### Anti-Leakage Enforcement

All filtering is performed in `main.py` before features are built:

- Monthly sims: `MONTH ≤ M+1` ✓
- Daily sims: `DATETIME ≤ 8th of M at 00:00:00` ✓  
- Historical prices/costs: `MONTH ≤ M` ✓
- **Never used**: M+1 realized prices, M+1 costs, M prices after day 7

---

## CLI Reference

```
python main.py --start-month YYYY-MM --end-month YYYY-MM [--target-k N] [--output FILE]

Arguments:
  --start-month   First TARGET_MONTH to predict (e.g. 2020-01)
  --end-month     Last TARGET_MONTH to predict  (e.g. 2023-12)
  --target-k      Desired selections per month, 10–100 (default: 50)
  --output        Output CSV path (default: opportunities.csv)
```

---

## Output Format

`opportunities.csv` contains three columns:

| Column | Format | Example |
|---|---|---|
| `TARGET_MONTH` | YYYY-MM | `2022-07` |
| `PEAK_TYPE` | ON / OFF | `ON` |
| `EID` | string | `EID_0042` |

Constraints enforced: no duplicates, 10 ≤ count per month ≤ 100.

---

## Performance Evaluation (Backtesting)

Run `backtest.py` to evaluate on the 2020–2023 development data using the same F1 and net profit metrics as the competition. Results printed to stdout; no leakage — every month's labels are excluded from the features used to predict it.

---

## Next Steps / Tuning

1. **Weight optimization**: Run grid search over `selector.DEFAULT_WEIGHTS` using `backtest.py` F1 as objective.
2. **Lookback window**: Test 3, 6, 12 month horizons for `hist_profit_rate`.
3. **Supervised model**: Once sufficient labeled months accumulate, train a LightGBM classifier on the feature matrix — target variable is `IS_PROFITABLE`.
4. **Scenario weighting**: Treat the 3 SCENARIOID scenarios differently (e.g., ensemble with learned weights).
