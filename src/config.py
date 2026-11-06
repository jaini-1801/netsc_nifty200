"""Central settings. Change values here, not inside the modules."""
START = "2023-11-01"      # ~3 years of daily data (state this in the report)
END = "2026-11-01"
R_MAIN = 0.5              # main correlation threshold (choose after looking at the sweep)
R_SWEEP = [round(0.30 + 0.05 * i, 2) for i in range(12)]   # 0.30 ... 0.85
MAX_MISSING = 0.05        # drop stocks with more than 5% missing prices
SEED = 42
UNIVERSE_URL = "https://www.niftyindices.com/IndexConstituent/ind_nifty200list.csv"
HEAVYWEIGHTS = ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK"]
DATA_DIR, FIG_DIR, TAB_DIR = "data", "out/figures", "out/tables"
