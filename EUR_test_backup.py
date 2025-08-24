import pandas as pd
import pytz
import matplotlib.pyplot as plt
from datetime import datetime, time
import plotly.graph_objects as go
from datetime import timedelta
import os



AMS_TZ = pytz.timezone("Europe/Amsterdam")


# Summary counters (money-mapped)
  # 3R hit, then SL on rest (+$4 400)
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
#blocked_months = set()  # months where trading is stopped after hitting $104k
SKIP_WEEKDAYS = {0, 3}       # 0 = Monday, 3 = Thursday
ENTRY_START_AMS = (10, 0)
ENTRY_END_AMS   = (13, 30)# 13:30 UTC
# Day-of-week stats
withdrawals_by_year = {}  # "YYYY" -> float
yearly_pnl = {}           # optional: "YYYY" -> float
total_tp1 = 0
total_tp2 = 0
total_sl = 0
day_stats = {i: {'SL2': 0, 'SL5': 0, 'TP5': 0} for i in range(7)}


RISK_PER_TRADE = 2000  # USD
PIP_VALUE_PER_LOT = 1.0
ACCOUNT_BALANCE = 100_000  # Starting FTMO account balance
LIQUIDATION_COUNT = 0  # number of times account is liquidated
last_trade_month = None
os.makedirs("charts/html", exist_ok=True)
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


def simulate_trade(df_1m, bos_time, fractal_time, direction, df_3m, sweep_time, fractal_type):
    global total_sl2, total_tp5, total_sl5, total_skips, ACCOUNT_BALANCE, last_trade_month, total_tp1, total_tp2, total_sl


    entry_row = df_1m[df_1m['time'] == bos_time]
    if entry_row.empty:
        print(f"    ⛔ Skipped: BOS candle not found in df at {bos_time}")
        total_skips += 1
        return

    entry_price = entry_row['close'].iloc[0]
    entry_time = bos_time

    # ⛔ Stop trading if month is blocked
    current_trade_month = entry_time.to_period("M")
    month_str = str(current_trade_month)

    '''if month_str in blocked_months:
        return  # skip this trade

    # 💰 Check if account exceeds $104,000 → withdraw and block month
    if ACCOUNT_BALANCE > 108_000:
        excess = ACCOUNT_BALANCE - 100_000
        withdrawals_by_month[month_str] = withdrawals_by_month.get(month_str, 0) + excess
        ACCOUNT_BALANCE = 100_000
        blocked_months.add(month_str)
        print(f"💸 Balance exceeded $104k → Withdrawal: ${excess:.2f}, trading stopped for {month_str}")
        return  # stop this trade'''



    # 🔄 Monthly withdrawal and balance reset logic
    current_trade_month = entry_time.to_period("M")
    if last_trade_month is None:
        last_trade_month = current_trade_month

    if current_trade_month != last_trade_month:
        month_str = str(last_trade_month)
        if ACCOUNT_BALANCE > 100_000:
            withdrawal = ACCOUNT_BALANCE - 100_000
            withdrawals_by_month[month_str] = withdrawals_by_month.get(month_str, 0) + withdrawal
            year_str = month_str.split("-")[0]  # "YYYY"
            withdrawals_by_year[year_str] = withdrawals_by_year.get(year_str, 0.0) + withdrawal
            print(f"🏦 Withdrawal for {month_str}: ${withdrawal:.2f}")
            ACCOUNT_BALANCE = 100_000
        else:
            print(f"📉 No withdrawal for {month_str}. Continue with ${ACCOUNT_BALANCE:.2f}")
        last_trade_month = current_trade_month
        #blocked_months.discard(str(last_trade_month))  # ✅ Resume trading next month
    # -------------------------------------------------------------------
    #  ENTRY-WINDOW filter – Amsterdam clock
    # -------------------------------------------------------------------
    '''entry_local = entry_time.astimezone(AMS_TZ)
    hm = (entry_local.hour, entry_local.minute)

    if hm < ENTRY_START_AMS or hm > ENTRY_END_AMS:
        print(f"    ⛔ Skipped: entry {entry_local.strftime('%H:%M')} AMS "
              f"outside 10:00–13:30 window")
        total_skips += 1
        return'''

    print(f"    📥 ENTRY at {entry_time} | Price: {entry_price:.2f}")

    if not entry_time:
        print("    ⛔ Skipped: Entry not triggered.")
        total_skips += 1
        return

    if entry_time.hour >= 13:
        print(f"    ⛔ Skipped: Entry time {entry_time.strftime('%H:%M')} is too late (after 11:00)")
        total_skips += 1
        return

    '''atr_series = calculate_atr(df_3m[df_3m['time'] <= entry_time])
    if atr_series.dropna().empty:
        print("    ⛔ Skipped due to insufficient ATR data")
        total_skips += 1
        return'''

    '''latest_atr = atr_series.dropna().iloc[-1]
    latest_atr = float(latest_atr)
    scaled_atr = latest_atr / entry_price * 100  # ATR as percentage of price
    atr_values.append(scaled_atr)
    if scaled_atr > 0.08 or scaled_atr < 0.015:
        print(f"    ⛔ Skipped due to ATR filter: {scaled_atr:.4f}%")
        total_skips += 1
        return'''

    # window_df = df_3m[(df_3m['time'] >= sweep_time) & (df_3m['time'] <= entry_time)] use this for checking candle behavior between sweep and entry

    start_sl_window = bos_time.replace(hour=8, minute=0, second=0, microsecond=0)
    sl_range = df_3m[(df_3m['time'] >= start_sl_window) & (df_3m['time'] <= bos_time)]

    if sl_range.empty:
        print(f"    ⛔ Skipped: SL range empty between 08:00 and BOS at {bos_time}")
        total_skips += 1
        return

    if direction == 'BOS Down':
        sl = sl_range['high'].max()
        stop_distance = sl - entry_price
        tp = entry_price - 2 * stop_distance
    else:
        sl = sl_range['low'].min()
        stop_distance = entry_price - sl
        tp = entry_price + 2 * stop_distance


    if stop_distance <= 0:
        print(f"    ⛔ Skipped: Invalid stop distance (stop={sl:.2f}, entry={entry_price:.2f})")
        total_skips += 1
        return
    stop_distance_pips = stop_distance * 10_000  # For EUR/USD, 1 pip = 0.0001

    print(f"    🎯 TP set at {tp:.2f} (1.5R)")
    lot_size = RISK_PER_TRADE / (stop_distance * PIP_VALUE_PER_LOT)
    print(f"    📊 Lot size: {lot_size:.2f} (Stop distance: {stop_distance:.2f} / {stop_distance_pips:.1f} pips)")

    tp_2r = entry_price + 2 * stop_distance if direction == 'BOS Up' else entry_price - 2 * stop_distance
    tp_3r = entry_price + 3 * stop_distance if direction == 'BOS Up' else entry_price - 3 * stop_distance
    sl_original = sl
    be_triggered = False
    position_remaining = 1.0  # 100% at start

    after_entry = df_3m[df_3m['time'] > entry_time].copy()
    for _, row in after_entry.iterrows():
        # ✅ Take partial profit at 2R
        if not be_triggered:
            if (direction == 'BOS Up' and row['high'] >= tp_2r) or \
                    (direction == 'BOS Down' and row['low'] <= tp_2r):
                partial_profit = RISK_PER_TRADE * 2 * 0.6
                remaining_risk = RISK_PER_TRADE * 0.4

                # log partial TP
                equity_time.append(row['time'])
                equity_curve.append(ACCOUNT_BALANCE)
                day_stats[entry_time.weekday()]['TP5'] += 1
                month_key = row['time'].strftime('%Y-%m')
                monthly_pnl[month_key] = monthly_pnl.get(month_key, 0) + partial_profit

                print(f"    ✅ Partial TP HIT at {row['time']} → +{partial_profit:.2f} USD (80%)")
                total_tp1 += 1
                # update SL and TP
                sl = entry_price
                tp = tp_3r
                be_triggered = True
                position_remaining = 0.4

                ACCOUNT_BALANCE += partial_profit
                check_liquidation(row['time'])


        # ❌ SL HIT (either full or after BE move)
        if (direction == 'BOS Up' and row['low'] <= sl) or \
                (direction == 'BOS Down' and row['high'] >= sl):
            if not be_triggered:
                # full SL
                total_sl2 += -RISK_PER_TRADE
                pnl = -RISK_PER_TRADE
                ACCOUNT_BALANCE -= RISK_PER_TRADE
                check_liquidation(row['time'])
                print(f"    ❌ Full SL HIT at {row['time']} → -{RISK_PER_TRADE:.2f} USD")
                total_sl += 1
                print(f"    💰 Account Balance after BE: ${ACCOUNT_BALANCE:.2f}")

            else:
                # BE SL on remaining 30%
                print(f"    🟦 SL to BE HIT at {row['time']} → $0 (on 20%)")
                print(f"    💰 Account Balance after BE: ${ACCOUNT_BALANCE:.2f}")
                pnl = 0.0

            equity_time.append(row['time'])
            equity_curve.append(ACCOUNT_BALANCE)
            day_stats[entry_time.weekday()]['SL2'] += 1
            month_key = row['time'].strftime('%Y-%m')
            monthly_pnl[month_key] = monthly_pnl.get(month_key, 0) + pnl


            label = "SL" if pnl < 0 else "BE"
            #plot_trade_chart_interactive(df_3m, entry_time, row['time'], entry_price, sl_original, tp, label)

            trade_log.append({
                "entry_time": entry_time,
                "exit_time": row['time'],
                "direction": "BUY" if direction == "BOS Up" else "SELL",
                "qty": lot_size * position_remaining,
                "entry_px": entry_price,
                "exit_px": sl,
                "sl_px": sl_original,
                "tp_px": tp,
                "gross_pnl": pnl,
                "fee": lot_size * position_remaining * FEE_PER_LOT,
                "net_pnl": pnl - lot_size * position_remaining * FEE_PER_LOT,
            })
            return

        # ✅ Full TP (3R on remaining 30%)
        if be_triggered and (
                (direction == 'BOS Up' and row['high'] >= tp) or
                (direction == 'BOS Down' and row['low'] <= tp)
        ):
            profit = RISK_PER_TRADE * 3 * 0.4

            ACCOUNT_BALANCE += profit
            check_liquidation(row['time'])
            equity_time.append(row['time'])
            equity_curve.append(ACCOUNT_BALANCE)
            day_stats[entry_time.weekday()]['TP5'] += 1
            month_key = row['time'].strftime('%Y-%m')
            monthly_pnl[month_key] = monthly_pnl.get(month_key, 0) + profit
            print(f"    ✅ TP HIT at {row['time']} → +{profit:.2f} USD (30%)")
            total_tp2 += 1
            print(f"    💰 Account Balance after trade: ${ACCOUNT_BALANCE:.2f}")

            #plot_trade_chart_interactive(df_3m, entry_time, row['time'], entry_price, sl_original, tp, "TP3")
            trade_log.append({
                "entry_time": entry_time,
                "exit_time": row['time'],
                "direction": "BUY" if direction == "BOS Up" else "SELL",
                "qty": lot_size * 0.4,
                "entry_px": entry_price,
                "exit_px": tp,
                "sl_px": sl_original,
                "tp_px": tp,
                "gross_pnl": profit,
                "fee": lot_size * 0.4 * FEE_PER_LOT,
                "net_pnl": profit - lot_size * 0.4 * FEE_PER_LOT,
            })
            return

    # If neither SL nor TP hit
    print("    ⛔ Skipped: No SL or TP hit after entry")
    total_skips += 1



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



def detect_sweep_and_bos(df_3m):
    df_3m['date'] = df_3m['time'].dt.date
    for day in df_3m['date'].unique():
        day_dt = pd.to_datetime(str(day)).tz_localize(AMS_TZ)
        #if day_dt.weekday() in SKIP_WEEKDAYS:  # {0, 3}
          #  continue  # skip the entire day
        day_3m = df_3m[df_3m['date'] == day]
        if len(day_3m) < 100:
            continue
        try:
            # 1️⃣ Define Asia session
            asia_start = AMS_TZ.localize(datetime.combine(day, time(1, 0)))
            asia_end   = AMS_TZ.localize(datetime.combine(day, time(8, 0)))
            asia_data = day_3m[(day_3m['time'] >= asia_start) & (day_3m['time'] <= asia_end)]

            if asia_data.empty:
                continue

            asia_high = asia_data['high'].max()
            asia_low  = asia_data['low'].min()

            print(f"🌙 {day} | Asia High: {asia_high:.2f}, Asia Low: {asia_low:.2f}")

            # 2️⃣ Define sweep detection window (Frankfurt + London open)
            sweep_start = AMS_TZ.localize(datetime.combine(day, time(8, 0)))
            sweep_end   = AMS_TZ.localize(datetime.combine(day, time(12, 0)))
            df_sweep = day_3m[(day_3m['time'] >= sweep_start) & (day_3m['time'] <= sweep_end)]

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

            bos_time, bos_label, fractal_time, fractal_type = detect_bos(day_3m, sweep_dir, sweep_time)
            if bos_time is None:
                continue

            print(f"[{day}] Sweep {sweep_dir.upper()} at {sweep_time} | BOS at {bos_time}")
            full_after_sweep = day_3m.copy()
            simulate_trade(day_3m, bos_time, fractal_time, bos_label, full_after_sweep, sweep_time, fractal_type)

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

    detect_sweep_and_bos(df_3m)

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


