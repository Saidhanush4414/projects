import json
import os
from datetime import datetime, timedelta
import random
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(ROOT_DIR, 'data')
ASSETS_DIR = os.path.join(ROOT_DIR, 'assets')
ICON_DIR = os.path.join(ROOT_DIR, 'icon')


def load_json(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_coins():
    """Return coins from the richest available source.

    Priority:
    1) CoinMarketCap CSV at project root: coinmarketcap_06122017.csv
    2) JSON at data/coins.json
    """
    # 1) Try CoinMarketCap CSV at project root
    cmc_csv = os.path.join(ROOT_DIR, 'coinmarketcap_06122017.csv')
    if os.path.exists(cmc_csv):
        try:
            # Many dumps have an unnamed index column; drop it via index_col=0
            df = pd.read_csv(cmc_csv, index_col=0)
            # Normalize column names that might vary in case/format
            cols = {c.lower(): c for c in df.columns}
            def col(name, fallback=None):
                return cols.get(name.lower(), fallback)

            symbol_col = col('symbol')
            name_col = col('name')
            price_col = col('price_usd')
            market_cap_col = col('market_cap_usd')
            volume_col = col('24h_volume_usd')
            change24_col = col('percent_change_24h')
            rank_col = col('rank')

            # Filter rows with essential fields
            required = [symbol_col, name_col, price_col]
            df = df.dropna(subset=[c for c in required if c is not None])

            # Sort by rank if available, else by market cap desc
            if rank_col and rank_col in df.columns:
                # Some ranks may be strings; coerce to numeric
                df[rank_col] = pd.to_numeric(df[rank_col], errors='coerce')
                df = df.sort_values(by=rank_col, ascending=True, na_position='last')
            elif market_cap_col and market_cap_col in df.columns:
                df = df.sort_values(by=market_cap_col, ascending=False)

            coins = []
            for _, row in df.iterrows():
                def safe_get(column_name, default=None):
                    if column_name is None or column_name not in df.columns:
                        return default
                    val = row[column_name]
                    # Convert NaNs to default
                    if pd.isna(val):
                        return default
                    return val

                symbol_val = str(safe_get(symbol_col, '')).upper()
                name_val = str(safe_get(name_col, ''))

                # Resolve icon filename by trying common patterns
                icon_filename = None
                if os.path.isdir(ICON_DIR):
                    candidates = [
                        f"{symbol_val}.svg",
                        f"{symbol_val.lower()}.svg",
                        f"{name_val}.svg",
                        f"{name_val.lower()}.svg",
                        f"{name_val.replace(' ', '-').lower()}.svg",
                    ]
                    for cand in candidates:
                        path = os.path.join(ICON_DIR, cand)
                        if os.path.exists(path):
                            icon_filename = cand
                            break

                coins.append({
                    'symbol': symbol_val,
                    'name': name_val,
                    'price': float(safe_get(price_col, 0.0)),
                    'market_cap': float(safe_get(market_cap_col, 0.0)) if market_cap_col else None,
                    'volume': float(safe_get(volume_col, 0.0)) if volume_col else None,
                    'change_24h': float(safe_get(change24_col, 0.0)) if change24_col else 0.0,
                    'logo': (f"/icon/{icon_filename}" if icon_filename else None)
                })

            # Keep a reasonable number for UI; return all if desired by frontend
            return coins
        except Exception as e:
            print(f"Error loading CoinMarketCap CSV: {e}")

    # 2) Fallback to JSON file
    json_path = os.path.join(DATA_DIR, 'coins.json')
    if os.path.exists(json_path):
        return load_json(json_path)['coins']
    return []


def get_news():
    # Prefer CSV seed if available
    seed_csv = os.path.join(DATA_DIR, 'news_seed.csv')
    if os.path.exists(seed_csv):
        try:
            df = pd.read_csv(seed_csv)
            records = []
            for _, r in df.iterrows():
                records.append({
                    'id': int(r.get('id', 0)) if not pd.isna(r.get('id')) else 0,
                    'title': str(r.get('title', '')),
                    'date': str(r.get('date', '')),
                    'source': str(r.get('source', '')),
                    'coins': str(r.get('coins', '')),
                    'source_ref': str(r.get('source_ref', '')),
                })
            return records
        except Exception as e:
            print(f"Failed to read news_seed.csv: {e}")
    # Fallback to legacy json
    path = os.path.join(DATA_DIR, 'news.json')
    if os.path.exists(path):
        return load_json(path)['news']
    return []


def expand_news_to_count(news: list[dict], target: int = 500) -> list[dict]:
    if not news:
        return []
    import itertools
    out = []
    counter = 1
    cycler = itertools.cycle(news)
    while len(out) < target:
        base = next(cycler).copy()
        base['id'] = counter
        # Keep original title but append a copy index to ensure uniqueness
        base['title'] = f"{base.get('title', '')}"
        out.append(base)
        counter += 1
    return out[:target]


def filter_news_by_coin(news: list[dict], coin_symbol_or_name: str | None) -> list[dict]:
    if not coin_symbol_or_name:
        return news
    key = (coin_symbol_or_name or '').strip().lower()
    filtered = []
    for item in news:
        coins_field = (item.get('coins') or '')
        parts = [p.strip().lower() for p in str(coins_field).replace(',', ';').split(';') if p.strip()]
        if key in parts:
            filtered.append(item)
        else:
            # Try name match in title as weaker fallback
            if key and key in (item.get('title') or '').lower():
                filtered.append(item)
    return filtered


def generate_synthetic_ohlcv(days: int = 60, start_price: float = 30000.0):
    timestamps = [datetime.now() - timedelta(days=days - i) for i in range(days)]
    price = start_price
    rows = []
    rng = np.random.default_rng(42)
    base_vol = 1_000_000
    for i in range(days):
        drift = rng.normal(0, 0.01)
        price *= np.exp(drift)
        vol = base_vol * (1 + rng.normal(0, 0.2))
        o = price * (1 + rng.normal(0, 0.005))
        c = price * (1 + rng.normal(0, 0.005))
        h = max(o, c) * (1 + rng.uniform(0, 0.01))
        l = min(o, c) * (1 - rng.uniform(0, 0.01))
        rows.append({
            'date': timestamps[i].strftime('%Y-%m-%d'),
            'open': float(o),
            'high': float(h),
            'low': float(l),
            'close': float(c),
            'volume': float(max(vol, 1.0)),
        })
    return rows


def get_ohlcv(symbol: str, days: int = 60):
    # Look for csv in data/<symbol>.csv else generate
    csv_path = os.path.join(DATA_DIR, f'{symbol.upper()}.csv')
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        return df.to_dict(orient='records')
    return generate_synthetic_ohlcv(days=days, start_price=30000 if symbol.upper() == 'BTC' else 2000)


