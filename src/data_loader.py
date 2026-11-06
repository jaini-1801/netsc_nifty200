"""Download once, cache locally, clean, and compute log returns."""
import os
import numpy as np
import pandas as pd
import requests
import yfinance as yf
from . import config as C


def load_universe():
    f = f"{C.DATA_DIR}/nifty200_list.csv"

    def valid(path):
        try:
            d = pd.read_csv(path)
            return {"Symbol", "Industry"}.issubset(d.columns) and len(d) > 100
        except Exception:
            return False

    if os.path.exists(f) and not valid(f):
        os.remove(f)                                   # discard a bad cached file

    if not os.path.exists(f):
        urls = [C.UNIVERSE_URL,
                "https://archives.nseindia.com/content/indices/ind_nifty200list.csv"]
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
                   "Accept": "text/csv,*/*", "Referer": "https://www.niftyindices.com/"}
        for u in urls:
            try:
                r = requests.get(u, headers=headers, timeout=30)
                if r.ok and len(r.content) > 1000:
                    open(f, "wb").write(r.content)
                    if valid(f):
                        break
                    os.remove(f)
            except requests.RequestException:
                continue
        if not os.path.exists(f):
            raise SystemExit("Could not download the Nifty-200 list. Download it in your browser "
                             f"from {C.UNIVERSE_URL} and save it as {f}, then rerun.")

    df = pd.read_csv(f)
    df["Ticker"] = df["Symbol"].str.strip() + ".NS"
    return df


def load_prices(tickers):
    f = f"{C.DATA_DIR}/prices.csv"
    if os.path.exists(f) and os.path.getsize(f) < 10_000:
        os.remove(f)                                   # discard a bad cached file
    if not os.path.exists(f):
        px = yf.download(tickers, start=C.START, end=C.END, auto_adjust=True, progress=False)["Close"]
        if px.empty or px.shape[1] < 100:
            raise SystemExit(f"Yahoo returned only {px.shape} - check internet / run: pip install -U yfinance")
        px.to_csv(f)
    return pd.read_csv(f, index_col=0, parse_dates=True)


def clean_prices(px):
    """Drop stocks with too many gaps (e.g. recent IPOs), then fill short gaps."""
    keep = px.columns[px.isna().mean() <= C.MAX_MISSING]
    dropped = sorted(set(px.columns) - set(keep))
    px = px[keep].ffill(limit=3).dropna()
    return px, dropped


def log_returns(px):
    """R_i(t) = ln(P_i(t) / P_i(t-1))  -- exactly as the assignment specifies."""
    return np.log(px / px.shift(1)).dropna()


def get_data():
    uni = load_universe()
    px, dropped = clean_prices(load_prices(uni["Ticker"].tolist()))
    ret = log_returns(px)
    ret.to_csv(f"{C.DATA_DIR}/returns.csv")
    sector = dict(zip(uni["Ticker"], uni["Industry"]))
    return ret, sector, dropped
