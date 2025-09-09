import streamlit as st
from utils.data import get_coins
import os

st.set_page_config(page_title="All Coins", page_icon="🪙", layout="wide")
st.title("🪙 All Coins - Crypto Dashboard")
st.caption("Top coins from local dataset. Click a coin to open analytics page.")

coins = get_coins()

search = st.text_input("Search coin", "")
filtered = [c for c in coins if search.lower() in (c['name']+c['symbol']).lower()]

grid_cols = st.columns(4)
for i, coin in enumerate(filtered):
    with grid_cols[i % 4]:
        with st.container(border=True):
            logo_path = os.path.join('assets', 'coins', coin.get('logo',''))
            st.image(logo_path if os.path.exists(logo_path) else "https://via.placeholder.com/64", width=48)
            st.subheader(f"{coin['name']} ({coin['symbol']})")
            st.metric("Price", f"${coin['price']:,}")
            st.metric("Market Cap", f"${coin['market_cap']:,}")
            st.metric("Volume (24h)", f"${coin['volume']:,}")
            if st.button("Open Analytics", key=f"open_{coin['symbol']}"):
                st.switch_page("pages/2_Coin_Analytics.py")
                st.session_state['selected_symbol'] = coin['symbol']


