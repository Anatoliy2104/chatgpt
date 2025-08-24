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
skip_log = []  # optional diagnostic info

# --- trade logging ---------------------------------------------------------
trade_log = []        # one dict per closed trade
FEE_PER_LOT = 0.0     # set to your broker commission if


equity_curve = []
equity_time = []
monthly_pnl = {}
atr_values = []
withdrawals_by_month = {}
# P&L tracking
total_pnl = 0.0
# Day-of-week stats
withdrawals_by_year = {}  # "YYYY" -> float
yearly_pnl = {}           # optional: "YYYY" -> float
total_tp1_hits = 0
total_tp2_hits = 0
total_stop_hits = 0
day_stats = {i: {'TP1': 0, 'TP2': 0, 'SL': 0} for i in range(7)}

# Profit-factor / R-multiple tracking
gross_wins = 0.0
gross_losses = 0.0
total_r_multiple = 0.0
trade_count = 0

# Session stats (v1 only London)
session_stats = {'LDN': {'wins': 0, 'losses': 0, 'total': 0}}


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


def simulate_trade(day_3m, confirm_row, sweep_dir, sweep_price, asia_high, asia_low, asia_range_pips):
    """Execute trade based on confirmation candle."""
    global total_skips, ACCOUNT_BALANCE, total_pnl, total_tp1_hits, total_tp2_hits, total_stop_hits

    confirm_time = confirm_row['time']
    entry_deadline = confirm_time + timedelta(minutes=5 * ENTRY_TIMEOUT_BARS)

    # Preferred entry: midpoint of confirmation candle body
    body_mid = (confirm_row['open'] + confirm_row['close']) / 2
    if sweep_dir == 'low':
        entry_levels = [('mid', body_mid), ('asia', asia_low)]
    else:
        entry_levels = [('mid', body_mid), ('asia', asia_high)]

    window = day_3m[(day_3m['time'] > confirm_time) & (day_3m['time'] <= entry_deadline)]
    entry_price = entry_time = entry_label = None
    for _, row in window.iterrows():
        if sweep_dir == 'low':
            for label, level in entry_levels:
                if row['low'] <= level:
                    entry_price = level
                    entry_time = row['time']
                    entry_label = label
                    break
        else:
            for label, level in entry_levels:
                if row['high'] >= level:
                    entry_price = level
                    entry_time = row['time']
                    entry_label = label
                    break
        if entry_price is not None:
            break

    if entry_price is None:
        print("    ⛔ Skipped: entry timeout")
        total_skips += 1
        return None
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

    if sweep_dir == 'low':
        stop_price = sweep_price - pips_to_price(1.5)
        price_risk = entry_price - stop_price
        opp_boundary = asia_high
    else:
        stop_price = sweep_price + pips_to_price(1.5)
        price_risk = stop_price - entry_price
        opp_boundary = asia_low
    initial_stop = stop_price

    stop_pips = price_to_pips(price_risk)
    if stop_pips <= 0:
        print("    ⛔ Skipped: invalid stop distance")
        total_skips += 1
        skip_log.append({'time': confirm_time, 'reason': 'invalid_stop'})
        return None

    risk_capital = ACCOUNT_BALANCE * RISK_PER_EQUITY
    lot_size = risk_capital / (stop_pips * PIP_VALUE_PER_LOT)
    if lot_size <= 0 or lot_size > 100:
        print("    ⛔ Skipped: absurd lot size")
        total_skips += 1
        skip_log.append({'time': confirm_time, 'reason': 'lot_size'})
        return None

    tp1 = entry_price + pips_to_price(stop_pips) if sweep_dir == 'low' else entry_price - pips_to_price(stop_pips)

    dist_to_boundary = (opp_boundary - entry_price) if sweep_dir == 'low' else (entry_price - opp_boundary)
    r_boundary = dist_to_boundary / pips_to_price(stop_pips)
    r2 = min(max(r_boundary, 1.8), 2.5)
    tp2 = entry_price + pips_to_price(r2 * stop_pips) if sweep_dir == 'low' else entry_price - pips_to_price(r2 * stop_pips)

    print(f"    📥 ENTRY {entry_label} at {entry_time} | Price: {entry_price:.5f} | Stop {stop_price:.5f}")

    after_entry = day_3m[day_3m['time'] > entry_time]
    hit_tp1 = False
    pnl_total = 0.0
    outcome = None
    exit_time = entry_time
    exit_price = entry_price

    for _, row in after_entry.iterrows():
        if not hit_tp1:
            if sweep_dir == 'low':
                if row['low'] <= stop_price:
                    pnl_total = -risk_capital
                    exit_price = stop_price
                    exit_time = row['time']
                    outcome = 'SL'
                    break
                if row['high'] >= tp1:
                    pnl_total += risk_capital / 2
                    hit_tp1 = True
                    stop_price = entry_price
            else:
                if row['high'] >= stop_price:
                    pnl_total = -risk_capital
                    exit_price = stop_price
                    exit_time = row['time']
                    outcome = 'SL'
                    break
                if row['low'] <= tp1:
                    pnl_total += risk_capital / 2
                    hit_tp1 = True
                    stop_price = entry_price
        else:
            if sweep_dir == 'low':
                if row['low'] <= stop_price:
                    exit_price = stop_price
                    exit_time = row['time']
                    outcome = 'TP1_BE'
                    break
                if row['high'] >= tp2:
                    profit2 = risk_capital / 2 * r2
                    pnl_total += profit2
                    exit_price = tp2
                    exit_time = row['time']
                    outcome = 'TP2'
                    break
            else:
                if row['high'] >= stop_price:
                    exit_price = stop_price
                    exit_time = row['time']
                    outcome = 'TP1_BE'
                    break
                if row['low'] <= tp2:
                    profit2 = risk_capital / 2 * r2
                    pnl_total += profit2
                    exit_price = tp2
                    exit_time = row['time']
                    outcome = 'TP2'
                    break

    if outcome is None:
        if hit_tp1:
            outcome = 'TP1_BE'
        else:
            print("    ⛔ Skipped: No SL or TP hit after entry")
            total_skips += 1
            return None

    ACCOUNT_BALANCE += pnl_total
    total_pnl += pnl_total
    equity_curve.append(ACCOUNT_BALANCE)
    equity_time.append(exit_time)

    r_multiple = pnl_total / risk_capital if risk_capital else 0
    global gross_wins, gross_losses, total_r_multiple, trade_count, session_stats
    trade_count += 1
    total_r_multiple += r_multiple
    session_stats['LDN']['total'] += 1
    if pnl_total > 0:
        gross_wins += pnl_total
        session_stats['LDN']['wins'] += 1
    elif pnl_total < 0:
        gross_losses += pnl_total
        session_stats['LDN']['losses'] += 1

    month_key = exit_time.strftime('%Y-%m')
    monthly_pnl[month_key] = monthly_pnl.get(month_key, 0) + pnl_total

    if outcome == 'SL':
        total_stop_hits += 1
        day_stats[exit_time.weekday()]['SL'] += 1
    elif outcome == 'TP2':
        total_tp2_hits += 1
        day_stats[exit_time.weekday()]['TP2'] += 1
    else:
        total_tp1_hits += 1
        day_stats[exit_time.weekday()]['TP1'] += 1

    trade_log.append({
        'date': entry_time.date(),
        'session': 'LDN',
        'sweep_side': sweep_dir,
        'confirm_time': confirm_time,
        'entry_time': entry_time,
        'exit_time': exit_time,
        'direction': 'BUY' if sweep_dir == 'low' else 'SELL',
        'entry_px': entry_price,
        'sl_px': initial_stop,
        'tp1_px': tp1,
        'tp2_px': tp2,
        'exit_px': exit_price,
        'stop_pips': stop_pips,
        'risk_usd': risk_capital,
        'qty_lots': lot_size,
        'outcome': outcome,
        'r_multiple': r_multiple,
        'asia_high': asia_high,
        'asia_low': asia_low,
        'asia_range_pips': asia_range_pips,
        'gross_pnl': pnl_total,
        'fee': lot_size * FEE_PER_LOT,
        'net_pnl': pnl_total - lot_size * FEE_PER_LOT,
    })

    check_liquidation(exit_time)

    return pnl_total, outcome

def find_sweep(day_5m, asia_high, asia_low, day):
    sweep_start = AMS_TZ.localize(datetime.combine(day, LONDON_SWEEP_START_AMS))
    sweep_end = AMS_TZ.localize(datetime.combine(day, LONDON_SWEEP_END_AMS))
    window = day_5m[(day_5m['time'] >= sweep_start) & (day_5m['time'] <= sweep_end)]
    for _, row in window.iterrows():
        if row['high'] > asia_high:
            return 'high', row['time'], row['high']
        if row['low'] < asia_low:
            return 'low', row['time'], row['low']
    return None, None, None



def find_confirmation(day_5m, sweep_dir, sweep_time, asia_high, asia_low):
    confirm_end = sweep_time.replace(hour=LONDON_SWEEP_END_AMS.hour,
                                     minute=LONDON_SWEEP_END_AMS.minute,
                                     second=0,
                                     microsecond=0)
    after_sweep = day_5m[(day_5m['time'] > sweep_time) & (day_5m['time'] <= confirm_end)]
    highest = float('-inf')
    lowest = float('inf')
    for _, row in after_sweep.iterrows():
        body = abs(row['close'] - row['open'])
        rng = row['high'] - row['low']
        if rng == 0:
            continue
        cond_body = body >= 0.5 * rng
        cond_range = asia_low < row['close'] < asia_high
        if sweep_dir == 'low':
            cond_struct = row['close'] > highest
            if cond_body and cond_range and cond_struct:
                return row
            highest = max(highest, row['high'])
        else:
            cond_struct = row['close'] < lowest
            if cond_body and cond_range and cond_struct:
                return row
            lowest = min(lowest, row['low'])
    print("  ⛔ No valid confirmation found after sweep.")
    return None


def run_backtest(df_3m, df_5m):
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
            sweep_dir, sweep_time, sweep_price = find_sweep(day_5m, asia_high, asia_low, day)
            if sweep_dir is None:
                continue
            confirm_row = find_confirmation(day_5m, sweep_dir, sweep_time, asia_high, asia_low)
            if confirm_row is None:
                continue
            if daily_losses >= DAILY_MAX_LOSSES or daily_wins >= DAILY_MAX_WINS:
                print(f"   🔒 Daily guardrail already hit ({daily_losses} losses, {daily_wins} wins).")
                continue
            print(f"[{day}] Sweep {sweep_dir.upper()} at {sweep_time} | Confirm at {confirm_row['time']}")
            pnl, outcome = simulate_trade(day_3m, confirm_row, sweep_dir, sweep_price, asia_high, asia_low, asia_range_pips)
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
    net_pnl = total_pnl
    print(f"""
📜 Trade Summary:
  ✅ TP1 hits: {total_tp1_hits}
  🏁 TP2 hits: {total_tp2_hits}
  ❌ Stops:    {total_stop_hits}
  ⛘ Skipped:  {total_skips}
💰 Net P&L:   ${net_pnl:,.2f}
""")
    total_trades = len(trade_log)
    avg_r = total_r_multiple / trade_count if trade_count else 0
    pf = (gross_wins / abs(gross_losses)) if gross_losses < 0 else float('inf')
    win_rate = (sum(1 for t in trade_log if t['outcome'] != 'SL') / total_trades * 100) if total_trades else 0
    print(f"📈 Profit Factor: {pf:.2f} | Avg R: {avg_r:.2f} | Win%: {win_rate:.1f}%\n")

    print("🕒 Breakdown by Day of Week:")
    days = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    for i in range(7):
        stats = day_stats[i]
        print(f" {days[i]}: TP1={stats['TP1']}, TP2={stats['TP2']}, SL={stats['SL']}")

    print("\n🕛 Hit Rate by Session:")
    for sess, sstats in session_stats.items():
        wr = (sstats['wins'] / sstats['total'] * 100) if sstats['total'] else 0
        print(f" {sess}: {sstats['wins']}W/{sstats['losses']}L ({wr:.1f}% win)")

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

    run_backtest(df_3m, df_5m)

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


