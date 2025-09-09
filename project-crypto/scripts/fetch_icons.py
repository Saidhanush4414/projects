import os
import time
import unicodedata
import re
import requests

import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.data import get_coins  # noqa: E402


ICON_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'icon')


def slugify(value: str) -> str:
    value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore').decode('ascii')
    value = re.sub(r'[^a-zA-Z0-9\-\_\s]', '', value).strip().lower()
    value = re.sub(r'[\s_]+', '-', value)
    return value


def ensure_icon_dir():
    os.makedirs(ICON_DIR, exist_ok=True)


def already_has_icon(symbol: str) -> bool:
    candidates = [f"{symbol.upper()}.svg", f"{symbol.lower()}.svg"]
    return any(os.path.exists(os.path.join(ICON_DIR, c)) for c in candidates)


def try_download(url: str, dest_path: str, timeout: float = 8.0) -> bool:
    try:
        resp = requests.get(url, timeout=timeout)
        if resp.status_code == 200 and resp.headers.get('content-type', '').lower().startswith('image/svg'):
            with open(dest_path, 'wb') as f:
                f.write(resp.content)
            return True
        return False
    except Exception:
        return False


def build_candidate_urls(symbol: str, name: str):
    sym_u = symbol.upper()
    sym_l = symbol.lower()
    name_slug = slugify(name)

    # 1) spothq/cryptocurrency-icons (color)
    base_spothq = 'https://raw.githubusercontent.com/spothq/cryptocurrency-icons/master/svg/color'
    yield f"{base_spothq}/{sym_u}.svg"
    yield f"{base_spothq}/{sym_l}.svg"

    # 2) spothq/cryptocurrency-icons (black)
    base_spothq_bw = 'https://raw.githubusercontent.com/spothq/cryptocurrency-icons/master/svg/black'
    yield f"{base_spothq_bw}/{sym_u}.svg"
    yield f"{base_spothq_bw}/{sym_l}.svg"

    # 3) simple-icons by name slug
    base_simple = 'https://raw.githubusercontent.com/simple-icons/simple-icons/develop/icons'
    if name_slug:
        yield f"{base_simple}/{name_slug}.svg"

    # 4) common alternates: remove spaces, hyphens, and try lowercase
    alt = name_slug.replace('-', '') if name_slug else ''
    if alt:
        yield f"{base_simple}/{alt}.svg"


def fetch_missing_icons(max_icons: int | None = None, sleep_between: float = 0.15):
    ensure_icon_dir()
    coins = get_coins()
    downloaded = 0
    skipped = 0
    for coin in coins:
        symbol = (coin.get('symbol') or '').strip()
        name = (coin.get('name') or '').strip()
        if not symbol:
            continue

        if already_has_icon(symbol):
            skipped += 1
            continue

        dest = os.path.join(ICON_DIR, f"{symbol.upper()}.svg")
        success = False
        for url in build_candidate_urls(symbol, name):
            if try_download(url, dest):
                print(f"Downloaded {symbol}: {url}")
                success = True
                downloaded += 1
                break

        if not success:
            print(f"No icon found for {symbol} ({name})")

        if max_icons is not None and downloaded >= max_icons:
            break

        time.sleep(sleep_between)

    print(f"Done. Downloaded: {downloaded}, Existing: {skipped}")


if __name__ == '__main__':
    # Optional: parse simple args
    import argparse
    parser = argparse.ArgumentParser(description='Fetch missing cryptocurrency SVG icons into the icon/ folder.')
    parser.add_argument('--max', type=int, default=None, help='Max number of icons to download in this run')
    parser.add_argument('--sleep', type=float, default=0.15, help='Delay between requests to be polite')
    args = parser.parse_args()
    fetch_missing_icons(max_icons=args.max, sleep_between=args.sleep)


