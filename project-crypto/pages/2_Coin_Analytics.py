import streamlit as st
from utils.data import get_coins, get_ohlcv
import pandas as pd

st.set_page_config(page_title="Coin Analytics", page_icon="📈", layout="wide")
st.title("📈 Coin Analytics")

coins = get_coins()
symbols = [c['symbol'] for c in coins]

default_symbol = st.session_state.get('selected_symbol', 'BTC' if 'BTC' in symbols else (symbols[0] if symbols else 'BTC'))
symbol = st.selectbox("Select coin", options=symbols or ['BTC'], index=(symbols.index(default_symbol) if default_symbol in symbols else 0))

ohlcv = get_ohlcv(symbol, days=60)
df = pd.DataFrame(ohlcv)

left, right = st.columns([2,1])
with left:
    st.subheader(f"{symbol} Price (Last {len(df)} Days)")
    st.line_chart(df.set_index('date')['close'])
with right:
    st.subheader("Summary")
    if len(df) >= 2:
        change = (df['close'].iloc[-1] - df['close'].iloc[-2]) / df['close'].iloc[-2] * 100
    else:
        change = 0.0
    st.metric("24h Change", f"{change:.2f}%")
    st.metric("Volume (last)", f"{df['volume'].iloc[-1]:,.0f}")
    # Market cap proxy using coins dataset
    coin_info = next((c for c in coins if c['symbol'] == symbol), None)
    st.metric("Market Cap", f"${coin_info['market_cap']:,}" if coin_info else "N/A")

st.subheader("OHLCV Table")
st.dataframe(df.tail(30), use_container_width=True)


