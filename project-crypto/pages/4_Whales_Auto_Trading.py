import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

st.set_page_config(page_title="Whales & Auto-Trading", page_icon="🐋", layout="wide")
st.title("🐋 Whale Detection & Auto-Trading Simulation")
st.caption("Auto-runs continuously. Uses volume z-score + price impact to decide BUY/SELL/HOLD.")

# State
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = {
        'cash': 10000.0,
        'position': 0.0,
        'entry': None,
        'equity': [10000.0],
        'timestamps': [datetime.now()],
        'trades': []
    }

def generate_tick(prev_price):
    drift = np.random.normal(0, 0.002)
    price = max(100.0, prev_price * (1 + drift))
    volume = np.random.lognormal(mean=6, sigma=0.4)
    # 3% chance whale
    whale = np.random.rand() < 0.03
    impact = 0.0
    if whale:
        direction = 1 if np.random.rand() < 0.5 else -1
        impact = direction * np.random.uniform(0.02, 0.06)
        price *= (1 + impact)
        volume *= np.random.uniform(8, 15)
    return price, volume, impact

def step_sim():
    p = st.session_state.portfolio
    last_price = p['trades'][-1]['price'] if p['trades'] else 30000.0
    price, volume, impact = generate_tick(last_price)

    # Maintain rolling window for z-score
    if 'vol_hist' not in p:
        p['vol_hist'] = []
        p['price_hist'] = []
    p['vol_hist'].append(volume)
    p['price_hist'].append(price)
    if len(p['vol_hist']) > 48:
        p['vol_hist'] = p['vol_hist'][-48:]
        p['price_hist'] = p['price_hist'][-48:]

    vol_z = 0
    if len(p['vol_hist']) > 10:
        vol_z = (volume - np.mean(p['vol_hist'])) / (np.std(p['vol_hist']) + 1e-9)

    price_change = 0 if len(p['price_hist']) < 2 else (p['price_hist'][-1] - p['price_hist'][-2]) / p['price_hist'][-2]
    whale_detected = (vol_z > 3.0 and abs(price_change) > 0.01)
    signal = 'HOLD'
    if whale_detected:
        signal = 'BUY' if price_change > 0 else 'SELL'

    # Trading rules
    if signal == 'BUY' and p['position'] == 0.0:
        # buy 20%
        amount = p['cash'] * 0.2
        qty = amount / price
        p['cash'] -= amount
        p['position'] += qty
        p['entry'] = price
        p['trades'].append({'time': datetime.now(), 'side': 'BUY', 'price': price, 'qty': qty})
    elif signal == 'SELL' and p['position'] > 0.0:
        # sell only if profitable
        if price > (p['entry'] or price):
            proceeds = p['position'] * price
            p['cash'] += proceeds
            p['trades'].append({'time': datetime.now(), 'side': 'SELL', 'price': price, 'qty': p['position']})
            p['position'] = 0.0
            p['entry'] = None

    equity = p['cash'] + p['position'] * price
    p['equity'].append(equity)
    p['timestamps'].append(datetime.now())

    return price, volume, vol_z, price_change, signal

col1, col2, col3, col4 = st.columns(4)
price, volume, vol_z, price_change, signal = step_sim()

with col1:
    st.metric("Last Price", f"${price:,.2f}", f"{price_change*100:.2f}%")
with col2:
    st.metric("Volume", f"{volume:,.0f}")
with col3:
    st.metric("Vol Z-Score", f"{vol_z:.2f}")
with col4:
    st.metric("AI Signal", signal)

p = st.session_state.portfolio
left, right = st.columns([2,1])
with left:
    st.subheader("Equity Curve")
    df_eq = pd.DataFrame({'time': p['timestamps'], 'equity': p['equity']})
    st.line_chart(df_eq.set_index('time'))
with right:
    st.subheader("Portfolio")
    st.metric("Cash", f"${p['cash']:,.2f}")
    st.metric("Position (BTC)", f"{p['position']:.6f}")
    st.metric("Equity", f"${p['equity'][-1]:,.2f}")

st.subheader("Recent Trades")
st.dataframe(pd.DataFrame(p['trades'][-10:]), use_container_width=True)

# Auto-refresh
from streamlit_autorefresh import st_autorefresh
st_autorefresh(interval=1500, limit=None, key="auto_sim")


