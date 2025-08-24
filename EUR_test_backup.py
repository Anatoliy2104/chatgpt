import pandas as pd
import pytz
import matplotlib.pyplot as plt
from datetime import datetime, time
import plotly.graph_objects as go
from datetime import timedelta
import os



AMS_TZ = pytz.timezone("Europe/Amsterdam")

# ---------------------------------------------------------------------------
#  STRATEGY CONSTANTS
# ---------------------------------------------------------------------------
ASIA_START_AMS = time(1, 0)
ASIA_END_AMS = time(8, 0)
LONDON_SWEEP_START_AMS = time(8, 0)
LONDON_SWEEP_END_AMS = time(10, 30)
# New York session defined but unused for now
NEW_YORK_START_AMS = time(14, 30)
NEW_YORK_END_AMS = time(17, 0)

ASIA_RANGE_MIN_PIPS = 12
ASIA_RANGE_MAX_PIPS = 38
ENTRY_TIMEOUT_BARS = 3               # 3 x 5-minute candles
SKIP_WEEKDAYS = {0}                  # Skip Mondays (0 = Monday)
RISK_PER_EQUITY = 0.01               # Risk 1% of equity per trade
DAILY_MAX_LOSSES = 1
DAILY_MAX_WINS = 2
PIP_VALUE_PER_LOT = 10.0             # EURUSD pip value per standard lot


def price_to_pips(price_diff: float) -> float:
    """Convert a price difference (e.g. 0.0001) to pips for EURUSD."""
    return price_diff * 10_000


def pips_to_price(pips: float) -> float:
    """Convert pips to a price difference for EURUSD."""
    return pips / 10_000


# Summary counters (money-mapped)
total_skips = 0

# --- trade logging ---------------------------------------------------------
trade_log = []        # one dict per closed trade
FEE_PER_LOT = 0.0     # set to your broker commission if


equity_curve = []
equity_time = []
monthly_pnl = {}
atr_values = []
withdrawals_by_month = {}
total_sl2 = 0
total_tp5 = 0
total_sl5 = 0
# Day-of-week stats
withdrawals_by_year = {}  # "YYYY" -> float
yearly_pnl = {}           # optional: "YYYY" -> float
total_tp1 = 0
total_tp2 = 0
total_sl = 0
day_stats = {i: {'SL2': 0, 'SL5': 0, 'TP5': 0} for i in range(7)}


ACCOUNT_BALANCE = 100_000  # Starting FTMO account balance
LIQUIDATION_COUNT = 0  # number of times account is liquidated
last_trade_month = None
os.makedirs("charts/html", exist_ok=True)
daily_wins = 0
daily_losses = 0
current_trade_day = None
# --------------------------------------------------------------------------
#  OUTPUT FOLDER FOR ALL REPORT FILES
# --------------------------------------------------------------------------
OUTPUT_DIR = "reports"
os.makedirs(OUTPUT_DIR, exist_ok=True)


'''def plot_trade_chart_interactive(df_3m, entry_time, exit_time, entry_price, sl, tp, result_label):
    df_plot = df_3m[(df_3m['time'] >= entry_time - timedelta(minutes=60)) &
                 (df_3m['time'] <= exit_time + timedelta(minutes=30))].copy()

    fig = go.Figure(data=[go.Candlestick(
        x=df_plot['time'],
        open=df_plot['open'],
        high=df_plot['high'],
        low=df_plot['low'],
        close=df_plot['close'],
        increasing_line_color='green',
        decreasing_line_color='red',
        showlegend=False
    )])

    y_bottom = min(entry_price, sl, tp)
    y_top = max(entry_price, sl, tp)

    fig.add_shape(type="rect",
                  x0=entry_time, x1=exit_time,
                  y0=y_bottom, y1=y_top,
                  line=dict(width=0), fillcolor="lightgray", opacity=0.3)

    for price, name, color in zip([entry_price, sl, tp], ['Entry', 'Stop Loss', 'TP1',],
                                  ['blue', 'red', 'orange', 'green']):
        fig.add_shape(type="line",
                      x0=entry_time, x1=exit_time,
                      y0=price, y1=price,
                      line=dict(color=color, dash="dot"))
        fig.add_annotation(x=entry_time, y=price, text=name, showarrow=False,
                           yshift=10 if price < y_top else -10,
                           font=dict(color=color))

    fig.update_layout(title=f"Trade on {entry_time.strftime('%Y-%m-%d %H:%M')} ({result_label})",
                      xaxis_title='Time', yaxis_title='Price',
                      xaxis_rangeslider_visible=False,
                      template='plotly_white')

    filename = entry_time.strftime("charts/html/%Y-%m-%d_%H-%M.html")
    fig.write_html(filename)'''



def find_fractal_highs_lows(df):
    fractal_highs = []
    fractal_lows = []
    for i in range(2, len(df) - 2):
        center = df.iloc[i]
        left = df.iloc[i - 2:i]
        right = df.iloc[i + 1:i + 3]

        if center['high'] > left['high'].max() and center['high'] > right['high'].max():
            fractal_highs.append({'time': center['time'], 'price': center['high']})

        if center['low'] < left['low'].min() and center['low'] < right['low'].min():
            fractal_lows.append({'time': center['time'], 'price': center['low']})

    return fractal_highs, fractal_lows


# def find_real_fvgs_custom(df):
#     real_fvgs = []
#     for i in range(2, len(df)):
#         c1 = df.iloc[i - 2]
#         c2 = df.iloc[i - 1]
#         c3 = df.iloc[i]
#
#         body_low = min(c2['open'], c2['close'])
#         body_high = max(c2['open'], c2['close'])
#
#         left_top = c1['high']
#         left_bottom = c1['low']
#         right_top = c3['high']
#         right_bottom = c3['low']
#
#         if left_top < body_high and right_bottom > body_low:
#             untouched_top = min(body_high, right_bottom)
#             untouched_bottom = max(body_low, left_top)
#             if untouched_top - untouched_bottom >= 0.4:
#                 real_fvgs.append({
#                     'time': c3['time'],
#                     'direction': 'bullish',
#                     'fvg_top': untouched_top,
#                     'fvg_bottom': untouched_bottom,
#                     'bos_candle_time': c3['time']
#                 })
#
#         elif left_bottom > body_low and right_top < body_high:
#             untouched_top = min(body_high, left_bottom)
#             untouched_bottom = max(body_low, right_top)
#             if untouched_top - untouched_bottom >= 0.4:
#                 real_fvgs.append({
#                     'time': c3['time'],
#                     'direction': 'bearish',
#                     'fvg_top': untouched_top,
#                     'fvg_bottom': untouched_bottom,
#                     'bos_candle_time': c3['time']
#                 })
#     return real_fvgs

'''def calculate_atr(df, period=10):
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift())
    low_close = abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr'''

def check_liquidation(current_time):
    global ACCOUNT_BALANCE, LIQUIDATION_COUNT
    if ACCOUNT_BALANCE <= 90_000:
        LIQUIDATION_COUNT += 1
        print(f"💥 ACCOUNT LIQUIDATED at {current_time} → Balance: ${ACCOUNT_BALANCE:.2f}")
        ACCOUNT_BALANCE = 100_000  # Reset account after liquidation
        print(f"🔁 New account created. Balance reset to ${ACCOUNT_BALANCE}")


def simulate_trade(df_3m, bos_time, fractal_time, direction, df_full, sweep_time, fractal_type,
                   asia_high, asia_low, asia_range_pips):
    """Simulate a single trade. Returns PnL or None if skipped."""
    global total_sl2, total_tp5, total_skips, ACCOUNT_BALANCE, last_trade_month, total_tp1, total_sl
    entry_window_end = bos_time + timedelta(minutes=5 * ENTRY_TIMEOUT_BARS)
    entry_rows = df_3m[(df_3m['time'] >= bos_time) & (df_3m['time'] <= entry_window_end)]
    if entry_rows.empty:
        print("    ⛔ Skipped: entry timeout")
        total_skips += 1
        return None
    entry_row = entry_rows.iloc[0]
    entry_price = entry_row['close']
    entry_time = entry_row['time']
    current_trade_month = entry_time.to_period("M")
    global last_trade_month
    if last_trade_month is None:
        last_trade_month = current_trade_month
    if current_trade_month != last_trade_month:
        month_str = str(last_trade_month)
        if ACCOUNT_BALANCE > 100_000:
            w = ACCOUNT_BALANCE - 100_000
            withdrawals_by_month[month_str] = withdrawals_by_month.get(month_str, 0) + w
            year_str = month_str.split('-')[0]
            withdrawals_by_year[year_str] = withdrawals_by_year.get(year_str, 0.0) + w
            print(f"🏦 Withdrawal for {month_str}: ${w:.2f}")
            ACCOUNT_BALANCE = 100_000
        else:
            print(f"📉 No withdrawal for {month_str}. Continue with ${ACCOUNT_BALANCE:.2f}")
        last_trade_month = current_trade_month
    print(f"    📥 ENTRY at {entry_time} | Price: {entry_price:.5f}")
    start_sl_window = bos_time.replace(hour=8, minute=0, second=0, microsecond=0)
    sl_range = df_3m[(df_3m['time'] >= start_sl_window) & (df_3m['time'] <= bos_time)]
    if sl_range.empty:
        print("    ⛔ Skipped: SL range empty")
        total_skips += 1
        return None
    if direction == 'BOS Down':
        sl = sl_range['high'].max()
        stop_distance = sl - entry_price
        tp = entry_price - 2 * stop_distance
    else:
        sl = sl_range['low'].min()
        stop_distance = entry_price - sl
        tp = entry_price + 2 * stop_distance
    if stop_distance <= 0:
        print("    ⛔ Skipped: invalid stop distance")
        total_skips += 1
        return None
    stop_distance_pips = price_to_pips(stop_distance)
    risk_per_trade = ACCOUNT_BALANCE * RISK_PER_EQUITY
    lot_size = risk_per_trade / (stop_distance * PIP_VALUE_PER_LOT)
    print(f"    📊 Lot size: {lot_size:.2f} (Stop distance: {stop_distance_pips:.1f} pips)")
    after_entry = df_3m[df_3m['time'] > entry_time]
    for _, row in after_entry.iterrows():
        if (direction == 'BOS Up' and row['low'] <= sl) or (direction == 'BOS Down' and row['high'] >= sl):
            pnl = -risk_per_trade
            ACCOUNT_BALANCE += pnl
            total_sl2 += pnl
            total_sl += 1
            month_key = row['time'].strftime('%Y-%m')
            monthly_pnl[month_key] = monthly_pnl.get(month_key, 0) + pnl
            trade_log.append({
                'entry_time': entry_time,
                'exit_time': row['time'],
                'direction': 'BUY' if direction == 'BOS Up' else 'SELL',
                'qty': lot_size,
                'entry_px': entry_price,
                'exit_px': sl,
                'sl_px': sl,
                'tp_px': tp,
                'gross_pnl': pnl,
                'fee': lot_size * FEE_PER_LOT,
                'net_pnl': pnl - lot_size * FEE_PER_LOT,
                'asia_high': asia_high,
                'asia_low': asia_low,
                'asia_range_pips': asia_range_pips,
            })
            return pnl
        if (direction == 'BOS Up' and row['high'] >= tp) or (direction == 'BOS Down' and row['low'] <= tp):
            pnl = risk_per_trade * 2
            ACCOUNT_BALANCE += pnl
            total_tp5 += pnl
            total_tp1 += 1
            month_key = row['time'].strftime('%Y-%m')
            monthly_pnl[month_key] = monthly_pnl.get(month_key, 0) + pnl
            trade_log.append({
                'entry_time': entry_time,
                'exit_time': row['time'],
                'direction': 'BUY' if direction == 'BOS Up' else 'SELL',
                'qty': lot_size,
                'entry_px': entry_price,
                'exit_px': tp,
                'sl_px': sl,
                'tp_px': tp,
                'gross_pnl': pnl,
                'fee': lot_size * FEE_PER_LOT,
                'net_pnl': pnl - lot_size * FEE_PER_LOT,
                'asia_high': asia_high,
                'asia_low': asia_low,
                'asia_range_pips': asia_range_pips,
            })
            return pnl
    print("    ⛔ Skipped: No SL or TP hit after entry")
    total_skips += 1
    return None

def detect_bos(df_3m, sweep_dir, sweep_time):
    fractal_highs, fractal_lows = find_fractal_highs_lows(df_3m)

    # Define minimum allowed time for fractals (07:00 Amsterdam)
    min_fractal_time = sweep_time.replace(hour=7, minute=0, second=0, microsecond=0)

    for i in range(2, len(df_3m) - 2):
        candle = df_3m.iloc[i]
        close_price = candle['close']
        candle_time = candle['time']

        if candle_time <= sweep_time:
            continue

        if sweep_dir == 'low':
            for f in reversed(fractal_highs):
                if f['time'] < candle_time and f['time'] >= min_fractal_time and close_price > f['price']:
                    print(
                        f"  ↳ BOS Up (✅ HIGH Fractal): candle closed above fractal HIGH {f['price']:.2f} (fract. at {f['time']}, BOS candle at {candle_time})")
                    return candle_time, 'BOS Up', f['time'], 'high'

        elif sweep_dir == 'high':
            for f in reversed(fractal_lows):
                if f['time'] < candle_time and f['time'] >= min_fractal_time and close_price < f['price']:
                    print(
                        f"  ↳ BOS Down (✅ LOW Fractal): candle closed below fractal LOW {f['price']:.2f} (fract. at {f['time']}, BOS candle at {candle_time})")
                    return candle_time, 'BOS Down', f['time'], 'low'

    print("  ⛔ No valid BOS found after sweep.")
    return None, None, None, None



def detect_sweep_and_bos(df_3m, df_5m):
    df_5m['date'] = df_5m['time'].dt.date
    global daily_wins, daily_losses, current_trade_day
    for day in df_5m['date'].unique():
        day_dt = pd.to_datetime(str(day)).tz_localize(AMS_TZ)
        if day_dt.weekday() in SKIP_WEEKDAYS:
            continue
        day_5m = df_5m[df_5m['date'] == day]
        day_3m = df_3m[df_3m['time'].dt.date == day]
        if len(day_5m) < 50:
            continue
        daily_wins = daily_losses = 0
        current_trade_day = day
        try:
            asia_start = AMS_TZ.localize(datetime.combine(day, ASIA_START_AMS))
            asia_end = AMS_TZ.localize(datetime.combine(day, ASIA_END_AMS))
            asia_data = day_5m[(day_5m['time'] >= asia_start) & (day_5m['time'] <= asia_end)]
            if asia_data.empty:
                continue
            asia_high = asia_data['high'].max()
            asia_low = asia_data['low'].min()
            asia_range_pips = price_to_pips(asia_high - asia_low)
            if asia_range_pips < ASIA_RANGE_MIN_PIPS or asia_range_pips > ASIA_RANGE_MAX_PIPS:
                print(f"🌙 {day} | Asia range {asia_range_pips:.1f} pips → skipped")
                continue
            print(f"🌙 {day} | Asia High: {asia_high:.5f}, Asia Low: {asia_low:.5f} ({asia_range_pips:.1f} pips)")
            sweep_start = AMS_TZ.localize(datetime.combine(day, LONDON_SWEEP_START_AMS))
            sweep_end = AMS_TZ.localize(datetime.combine(day, LONDON_SWEEP_END_AMS))
            df_sweep = day_5m[(day_5m['time'] >= sweep_start) & (day_5m['time'] <= sweep_end)]
            sweep_dir = sweep_time = None
            for _, row in df_sweep.iterrows():
                if row['high'] > asia_high:
                    sweep_dir, sweep_time = 'high', row['time']
                    break
                if row['low'] < asia_low:
                    sweep_dir, sweep_time = 'low', row['time']
                    break
            if sweep_dir is None:
                continue
            bos_time, bos_label, fractal_time, fractal_type = detect_bos(day_5m, sweep_dir, sweep_time)
            if bos_time is None:
                continue
            print(f"[{day}] Sweep {sweep_dir.upper()} at {sweep_time} | BOS at {bos_time}")
            pnl = simulate_trade(day_3m, bos_time, fractal_time, bos_label, day_3m.copy(), sweep_time, fractal_type,
                                 asia_high, asia_low, asia_range_pips)
            if pnl is not None:
                if pnl > 0:
                    daily_wins += 1
                elif pnl < 0:
                    daily_losses += 1
                if daily_losses >= DAILY_MAX_LOSSES or daily_wins >= DAILY_MAX_WINS:
                    print(f"   🔒 Daily guardrail hit ({daily_losses} losses, {daily_wins} wins).")
                    continue
        except Exception as e:
            print(f"Error on {day}: {e}")
            continue


def print_trade_summary():
    net_pnl = total_tp5 + total_sl5 + total_sl2
    print(f"""
📜 Trade Summary:
  ❌ SL full:     ${-total_sl2:,.2f}
  🟡 SL after 3R: ${total_sl5:,.2f}
  🏁 TP 5R final: ${total_tp5:,.2f}
  ⛘ Skipped:     {total_skips}
💰 Net P&L:      ${net_pnl:,.2f}
""")
    print("🕒 Breakdown by Day of Week:")
    days = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    for i in range(7):
        stats = day_stats[i]
        print(f" {days[i]}: SL2={stats['SL2']}, SL5={stats['SL5']}, TP5={stats['TP5']}")

    print(f"\n💣 Total Liquidations: {LIQUIDATION_COUNT}")

    if withdrawals_by_month:
        print("\n💸 Withdrawals:")
        for month, amt in sorted(withdrawals_by_month.items()):
            print(f" {month}: ${amt:,.2f}")
    if withdrawals_by_year:
        print("\n🏦 Withdrawals per Year:")
        for y, amt in sorted(withdrawals_by_year.items()):
            print(f" {y}: ${amt:,.2f}")
        # write CSV
        pd.Series(withdrawals_by_year).sort_index() \
            .to_csv(os.path.join(OUTPUT_DIR, "withdrawals_yearly.csv"), header=["withdrawn"])
        # total into a TXT for quick check
        with open(os.path.join(OUTPUT_DIR, "withdrawals_total.txt"), "w") as f:
            f.write(f"{sum(withdrawals_by_year.values()):.2f}\n")
        print("📄 withdrawals_yearly.csv and withdrawals_total.txt written to /reports")

    # === Yearly PnL (optional) ===
    if yearly_pnl:
        print("\n📅 PnL per Year:")
        for y, amt in sorted(yearly_pnl.items()):
            print(f" {y}: ${amt:,.2f}")
        pd.Series(yearly_pnl).sort_index() \
            .to_csv(os.path.join(OUTPUT_DIR, "yearly_pnl.csv"), header=["net_pnl"])
        print("📄 yearly_pnl.csv written to /reports")

    if equity_curve:
        plt.figure(figsize=(10, 5))
        plt.plot(equity_time, equity_curve, label='Equity Curve')
        plt.axhline(y=100_000, color='gray', linestyle='--', label='Start Balance')
        plt.title("Equity Curve")
        plt.xlabel("Date")
        plt.ylabel("Account Balance (USD)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("charts/equity_curve.png")
        # raw equity data for analysis
        pd.DataFrame({
            "timestamp": equity_time,
            "equity": equity_curve
        }).to_csv(os.path.join(OUTPUT_DIR, "equity_curve.csv"), index=False)
        print("📄 equity_curve.csv written to", OUTPUT_DIR)

        plt.close()

    # 📊 Count SL distance in pips
    sl_pips_below_4 = 0
    sl_pips_above_4 = 0

    for trade in trade_log:
        stop_distance_pips = abs(trade["entry_px"] - trade["sl_px"]) * 10_000
        if stop_distance_pips < 6:
            sl_pips_below_4 += 1
        else:
            sl_pips_above_4 += 1

    # 🧾 Print summary
    print("\n📏 Stop-Loss Distance Analysis (in Pips):")
    print("┌───────────────────────────────┬────────────────────┐")
    print(f"│ Trades with SL < 6 pips       │ {sl_pips_below_4:>18} │")
    print(f"│ Trades with SL ≥ 6 pips       │ {sl_pips_above_4:>18} │")
    print("└───────────────────────────────┴────────────────────┘")



# Load CSVs and run
if __name__ == '__main__':
    df_3m = pd.read_csv("2024-2025.csv")

    rename_map = {'Gmt time': 'time', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close',
                  'Volume': 'volume'}
    df_3m.rename(columns=rename_map, inplace=True)
    df_3m['time'] = pd.to_datetime(df_3m['time'], format='%Y-%m-%d %H:%M:%S')
    df_3m['time'] = df_3m['time'].dt.tz_localize('Etc/GMT-3').dt.tz_convert('Europe/Amsterdam')
    df_3m.sort_values('time', inplace=True)
    df_3m.reset_index(drop=True, inplace=True)
    df_5m = (df_3m.set_index('time')
                   .resample('5T')
                   .agg({'open': 'first', 'high': 'max', 'low': 'min',
                         'close': 'last', 'volume': 'sum'})
                   .dropna()
                   .reset_index())

    detect_sweep_and_bos(df_3m, df_5m)

    print("1-Minute Data Range:")
    print("Start:", df_3m['time'].min())
    print("End:  ", df_3m['time'].max())
    print("Total Days Covered:", (df_3m['time'].max() - df_3m['time'].min()).days)
    if atr_values:
        print(f"🧪 Average ATR %: {sum(atr_values) / len(atr_values):.4f}%")
    else:
        print("🧪 No ATR values recorded.")

    print_trade_summary()
    print("\n📅 Monthly P&L:")
    for month, pnl in sorted(monthly_pnl.items()):
        print(f"  {month}: ${pnl:,.2f}")
    if monthly_pnl:
        total_monthly_pnl = sum(monthly_pnl.values())
        print(f"\n📆 Total Monthly PnL: ${total_monthly_pnl:,.2f}")
    if withdrawals_by_month:
        total_withdrawals = sum(withdrawals_by_month.values())
        print(f"\n💸 Total Withdrawals: ${total_withdrawals:,.2f}")
    # --- save monthly P&L -----------------------------------------------------
    # ---------- SAVE MONTHLY P&L ---------------------------------------------
    # ---------- SAVE MONTHLY P&L ---------------------------------------------
    if monthly_pnl:
        # build a sorted Series and write it
        monthly_series = pd.Series(monthly_pnl).sort_index()
        monthly_series.to_csv("reports/monthly_pnl.csv", header=["net_pnl"])
        print("📄 monthly_pnl.csv written to /reports")
    else:
        print("ℹ️ No monthly P&L to save.")


    # --- save trade logs -------------------------------------------------------
    if trade_log:
        df_trades = pd.DataFrame(trade_log)
        df_trades.to_csv(os.path.join(OUTPUT_DIR, "trades_full.csv"), index=False)
        df_trades[["entry_time", "exit_time", "direction", "net_pnl"]] \
            .to_csv(os.path.join(OUTPUT_DIR, "trades_lite.csv"), index=False)

        print(f"\nSaved {len(df_trades)} trades to trades_full.csv "
              f"and trades_lite.csv")
    else:
        print("\nNo trades were logged – check filters/time-range.")


