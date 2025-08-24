# EUR/USD Asia Sweep + BOS Backtester

This repository contains a **single-purpose Python backtest script** for EUR/USD built around an
**Asia-session sweep → BOS (break of structure) → 2R/3R partials** model. It ingests minute/3‑minute
CSV data, scans each trading day for an Asia-session high/low sweep, confirms a BOS via recent
fractal levels, simulates a trade with risk-based sizing, and writes reports (PnL, withdrawals,
equity curve, trades).

> Script entrypoint: `EUR_test_backup.py`

---

## What it does (high-level)

1. **Loads historical data** from `2024-2025.csv` and converts timestamps to **Europe/Amsterdam**.
2. For each day:
   - Computes **Asia session** (01:00–08:00 Amsterdam) high/low.
   - Looks for a **sweep** of that high/low between **08:00–12:00 Amsterdam**.
   - Searches for a **BOS**: a 3‑minute candle close beyond the most recent qualifying fractal
     (high for upside BOS, low for downside BOS).
   - **Enters on the BOS candle close** and sets stop to the day’s opposite extreme observed
     from 08:00 to the BOS candle.
   - Manages the position with **partials**: +2R partial, then move stop to BE for the remainder,
     and target +3R final.
3. Tracks **equity**, **withdrawals**, **monthly/annual PnL**, **liquidations** (balance ≤ 90k → reset to 100k),
   and prints a summary + writes CSV outputs.

**Note:** Some filters (ATR filter, strict entry window, weekday skipping) are present in the code but
commented out. See the **Feature flags & optional filters** section.

---

## File/Folder Structure

```
project/
├─ EUR_test_backup.py
├─ 2024-2025.csv                # input data (see format below)
├─ reports/                     # auto-created
│  ├─ monthly_pnl.csv
│  ├─ trades_full.csv
│  ├─ trades_lite.csv
│  ├─ equity_curve.csv
│  ├─ withdrawals_yearly.csv
│  └─ withdrawals_total.txt
├─ charts/
│  ├─ equity_curve.png
│  └─ html/                     # per-trade HTML charts (feature currently commented)
```

---

## Requirements

- Python 3.10+
- Packages:
  - `pandas`
  - `pytz`
  - `matplotlib`
  - `plotly` (only needed if you re‑enable HTML trade charts)
- The script will create `reports/` and `charts/html/` if missing.

Install:

```bash
pip install pandas pytz matplotlib plotly
```

---

## Input data format

The script expects a CSV named **`2024-2025.csv`** with columns (case-sensitive as below **before** renaming):

- `Gmt time` (e.g., `2025-08-01 09:00:00`)
- `Open`, `High`, `Low`, `Close`
- `Volume`

The script **renames** these to `time, open, high, low, close, volume` and then parses `time` as
**naive** UTC-like values that are first localized to `Etc/GMT-3` and then converted to
`Europe/Amsterdam`. Adjust if your data’s timezone differs.

> If you want to backtest a different time range, name your file accordingly or edit the path at the bottom of the script.

---

## How to run

```bash
python EUR_test_backup.py
```

You’ll see console logs including per-day Asia H/L, sweep/BOS events, entries, partial/SL hits,
withdrawals, and a final trade summary.

Outputs are saved into `reports/` and `charts/` (see **File/Folder Structure**).

---

## Strategy logic (exact rules)

### Sessions
- **Asia session:** 01:00–08:00 Amsterdam → compute **Asia High/Low**.
- **Sweep window:** 08:00–12:00 Amsterdam → detect **first breach** beyond Asia High or Low.

### BOS detection
- Build 3‑minute **fractal highs/lows** with a 5‑bar window (center bar is greater/less than both
  immediate two-left and two-right neighbors).
- After a sweep:
  - If the sweep broke **Asia Low** → look for a **BOS Up**: a candle close **above** the most
    recent qualifying **fractal high** (not earlier than 07:00 that day).
  - If the sweep broke **Asia High** → look for a **BOS Down**: a close **below** the most
    recent qualifying **fractal low** (not earlier than 07:00).

### Entry, Stop, Targets
- **Entry:** **at the BOS candle close**.
- **Stop:** Opposing extremum observed between **08:00** and the BOS candle:
  - BOS Up: stop = **lowest low** in that window.
  - BOS Down: stop = **highest high** in that window.
- **Position sizing:** risk-based:  
  `lot_size = RISK_PER_TRADE / (stop_distance * PIP_VALUE_PER_LOT)`
- **Targets & management:**
  - **+2R partial**, then move SL to **break-even** for remainder.
  - **+3R final** on the remaining size.

**Important implementation detail:** In the current code, the +2R partial realizes **60%** of
`RISK_PER_TRADE` (i.e., `2R * 0.6`) and leaves **40%** to run toward +3R. The console message still
mentions “80%” — update either the math or the message for consistency.

---

## Config & knobs

All config lives at the top of the file:

- `RISK_PER_TRADE` (default: `2000`) – USD risk per trade.
- `PIP_VALUE_PER_LOT` (default: `1.0`) – adjust to your symbol/contract.
- `ACCOUNT_BALANCE` (default: `100_000`) – start capital.
- **Liquidation logic:** when balance ≤ **90,000**, a liquidation event is counted and balance is
  reset to 100,000.
- `ENTRY_START_AMS` / `ENTRY_END_AMS` – **currently commented out**; enable to restrict the
  entry time in Amsterdam time.
- `SKIP_WEEKDAYS` – **currently not enforced**; un-comment the check to skip e.g. Monday/Thursday.
- `FEE_PER_LOT` – commission placeholder (currently `0.0`).

---

## Feature flags & optional filters

These exist in the file but are **commented out by default**:

- **ATR filter**: compute an ATR on 3‑minute bars, convert to % of price, and skip trades outside
  a band (e.g., 0.015%–0.08%). Re-enable by un-commenting the `calculate_atr` function and the
  checks around it.
- **Entry-window filter (Amsterdam)**: enforce `10:00–13:30` entries only.
- **Weekday skipping**: e.g., `{0, 3}` to skip Monday/Thursday.
- **Interactive HTML trade charts** with Plotly: see `plot_trade_chart_interactive` (commented).

---

## Outputs

- **Console summary**, including:
  - SL/TP counts, **liquidations**, day-of-week breakdown.
  - **Withdrawals** (see below) and **PnL** summaries.
- **Monthly PnL:** `reports/monthly_pnl.csv`
- **Yearly withdrawals:** `reports/withdrawals_yearly.csv`
- **Total withdrawals:** `reports/withdrawals_total.txt`
- **Equity curve data:** `reports/equity_curve.csv` and **plot** at `charts/equity_curve.png`
- **Trades:** `reports/trades_full.csv` (all columns) and `reports/trades_lite.csv` (compact)

### Withdrawal model
At the **turn of a month**, if balance is **above 100,000**, the excess is withdrawn, booked into:
- `withdrawals_by_month[YYYY-MM]` and
- `withdrawals_by_year[YYYY]`

The balance is then reset to **100,000**. If not above 100k, no withdrawal is recorded for that
month.

---

## Common tweaks

- **Change instrument**: adjust `PIP_VALUE_PER_LOT`, risk, and data feed.
- **Use 1‑minute vs 3‑minute**: the core loop operates on **3‑minute** bars for logic/entry. Ensure
  your input sampling aligns with the code or extend it to resample.
- **Adjust sessions**: change Asia window or sweep window to your market hypothesis.
- **Introduce structure/filters**: e.g., exclude Tuesdays/Wednesdays; add ADR/ATR guards; spread/fee
  modeling; direction filters.

---

## Troubleshooting

- **“No ATR values recorded.”** – ATR code is commented out; that’s expected.
- **“Skipped: BOS candle not found…”** – your data may not include the exact BOS timestamp; check
  timezones and granularity.
- **No trades logged** – likely no sweep+bos combos in the date range, or filters are too strict.
- **Massive position sizes** – verify `PIP_VALUE_PER_LOT` and symbol decimals for EUR/USD (0.0001 pip).

---

## Roadmap ideas

- Proper **spread/fee** modeling.
- Robust **data loader** with broker‑specific mapping and timezone flags.
- **Unit tests** for fractals, BOS finder, and PnL.
- **Parameterization** via CLI flags or a YAML config.
- Rich **HTML reports** and per-trade charts.
- **Performance** profiling and vectorization.

---

## License

Proprietary / Personal use by the author. Update this section if you plan to share.

