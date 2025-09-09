"""
WhaleTrader - local demo project (no paid APIs required)

Features:
- Demo data generator (synthetic crypto candle data + synthetic "whale trades")
- OR load your own CSV (OHLCV with timestamp)
- Feature engineering (returns, rolling vol, volume zscore, vwap, imbalance)
- Labeling whale events (volume spike + price impact)
- Train a classifier (RandomForest) to predict whale events n-steps ahead
- Simulated trading/backtest using model predictions (paper trading)
- Simple performance metrics (total return, max drawdown, sharpe-like)

Requirements (pip install):
pandas numpy scikit-learn matplotlib

Run:
python whale_trader.py
"""

import os
import math
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")


# ----------------------------
# Config
# ----------------------------
CONFIG = {
    "use_synthetic_data": True,      # If False -> attempts to load data/csv_path
    "csv_path": "data/BTCUSDT_1h.csv",
    "label_forward_steps": 3,        # predict whale within next N candles
    "volume_zscore_window": 24,      # hours
    "price_ret_window": 24,
    "train_test_split": 0.2,
    "random_state": 42,
    "initial_cash": 10000.0,
    "position_risk_fraction": 0.02,  # fraction of wallet per trade
    "stop_loss_pct": 0.02,
    "take_profit_pct": 0.04,
}


# ----------------------------
# Synthetic data generator for demo
# ----------------------------
def generate_synthetic_ohlcv(num_candles=24*90, start_price=30000):
    """
    Generates synthetic OHLCV hourly data with occasional 'whale' trades
    Whale trade = sudden large volume + immediate price jump (up or down).
    Returns DataFrame with columns: datetime, open, high, low, close, volume
    """
    rng = np.random.default_rng(CONFIG["random_state"])
    timestamps = [datetime.now() - timedelta(hours=num_candles - i) for i in range(num_candles)]
    price = start_price
    rows = []
    base_vol = 1000.0
    for i in range(num_candles):
        # small random walk in log-price
        drift = rng.normal(0, 0.001)
        price *= math.exp(drift)
        # baseline volume with diurnal-ish pattern
        vol = base_vol * (1 + 0.2 * math.sin(2 * math.pi * (i % 24) / 24)) * (1 + rng.normal(0, 0.1))
        is_whale = rng.random() < 0.005  # ~0.5% candles are whale events
        if is_whale:
            spike = rng.uniform(10, 60)   # 10x to 60x volume
            vol *= spike
            # price impact
            direction = 1 if rng.random() < 0.5 else -1
            impact = rng.uniform(0.01, 0.08) * direction
            price *= (1 + impact)
        # build OHLC as small variations around price
        o = price * (1 + rng.normal(0, 0.002))
        c = price * (1 + rng.normal(0, 0.002))
        h = max(o, c) * (1 + rng.uniform(0, 0.001))
        l = min(o, c) * (1 - rng.uniform(0, 0.001))
        rows.append({"datetime": timestamps[i], "open": o, "high": h, "low": l, "close": c, "volume": max(vol, 1.0)})
    df = pd.DataFrame(rows).sort_values("datetime").reset_index(drop=True)
    return df


# ----------------------------
# Load data
# ----------------------------
def load_data():
    if CONFIG["use_synthetic_data"]:
        print("Generating synthetic data...")
        df = generate_synthetic_ohlcv()
    else:
        if not os.path.exists(CONFIG["csv_path"]):
            raise FileNotFoundError(f"CSV not found at {CONFIG['csv_path']}. Change path or enable synthetic data.")
        print(f"Loading data from {CONFIG['csv_path']} ...")
        df = pd.read_csv(CONFIG["csv_path"], parse_dates=["datetime"])
        df = df.sort_values("datetime").reset_index(drop=True)
    return df


# ----------------------------
# Feature engineering & labeling
# ----------------------------
def engineer_features_and_labels(df):
    df = df.copy()
    # basic price features
    df["return_1"] = df["close"].pct_change()
    df["log_return_1"] = np.log(df["close"] / df["close"].shift(1))
    # rolling volatility and volume z-score
    w = CONFIG["volume_zscore_window"]
    df["vol_roll_mean"] = df["volume"].rolling(w, min_periods=1).mean()
    df["vol_roll_std"] = df["volume"].rolling(w, min_periods=1).std().fillna(0.0)
    df["vol_zscore"] = (df["volume"] - df["vol_roll_mean"]) / (df["vol_roll_std"] + 1e-9)
    # price momentum
    df["ret_24"] = df["close"].pct_change(CONFIG["price_ret_window"])
    # vwap-like (here simple)
    df["typical_price"] = (df["high"] + df["low"] + df["close"]) / 3
    df["vwap_24"] = (df["typical_price"] * df["volume"]).rolling(24, min_periods=1).sum() / (df["volume"].rolling(24, min_periods=1).sum() + 1e-9)
    df["vwap_diff"] = (df["close"] - df["vwap_24"]) / (df["vwap_24"] + 1e-9)
    # price impact estimate: intraperiod return
    df["intraperiod_ret"] = (df["close"] - df["open"]) / (df["open"] + 1e-9)
    # label: whale event if volume zscore > threshold and intraperiod_ret magnitude large
    # We label a whale "event" at time t if this candle had large volume and price impact.
    vol_threshold = 4.0
    impact_threshold = 0.01  # 1% intraperiod move
    df["is_whale_event"] = ((df["vol_zscore"] > vol_threshold) & (df["intraperiod_ret"].abs() > impact_threshold)).astype(int)
    # Create predict-forward label: if any whale event occurs in next N candles
    N = CONFIG["label_forward_steps"]
    df["future_whale"] = df["is_whale_event"].shift(-1).rolling(window=N, min_periods=1).max().shift(-(N-1)).fillna(0).astype(int)
    # features to use
    feature_cols = ["return_1", "log_return_1", "vol_zscore", "ret_24", "vwap_diff", "intraperiod_ret"]
    # fill nans
    df[feature_cols] = df[feature_cols].fillna(0.0)
    return df, feature_cols


# ----------------------------
# Train classifier
# ----------------------------
def train_model(df, feature_cols):
    X = df[feature_cols]
    y = df["future_whale"]
    # drop last N rows which cannot be labeled
    N = CONFIG["label_forward_steps"]
    X = X.iloc[:-N]
    y = y.iloc[:-N]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=CONFIG["train_test_split"], random_state=CONFIG["random_state"], stratify=y)
    # Use RandomForest (fast, robust)
    clf = RandomForestClassifier(n_estimators=200, random_state=CONFIG["random_state"], class_weight="balanced")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_proba_full = clf.predict_proba(X_test)
    # Handle case where only one class is present
    if y_proba_full.shape[1] == 1:
        y_proba = y_proba_full[:, 0]
        print("Warning: Only one class present in predictions")
    else:
        y_proba = y_proba_full[:, 1]
    print("Classification report:")
    print(classification_report(y_test, y_pred))
    try:
        print("ROC AUC: ", roc_auc_score(y_test, y_proba))
    except:
        print("ROC AUC: cannot compute (maybe only one class present).")
    return clf


# ----------------------------
# Backtesting / Simulator
# ----------------------------
class Simulator:
    def __init__(self, df, model, feature_cols):
        self.df = df.reset_index(drop=True)
        self.model = model
        self.feature_cols = feature_cols
        self.cash = CONFIG["initial_cash"]
        self.position = 0.0   # amount of coin held
        self.position_entry_price = None
        self.equity_curve = []

    def step(self, i):
        row = self.df.loc[i]
        X = row[self.feature_cols].values.reshape(1, -1)
        pred_proba_full = self.model.predict_proba(X)
        # Handle case where only one class is present
        if pred_proba_full.shape[1] == 1:
            pred_proba = pred_proba_full[0, 0]
        else:
            pred_proba = pred_proba_full[0, 1]
        # threshold can be tuned; using 0.5 default
        buy_signal = pred_proba > 0.5
        # If we have no position and model signals whale-buying, open long
        if buy_signal and self.position <= 0:
            # size in USD we risk: fraction of wallet
            allocation = self.cash * CONFIG["position_risk_fraction"]
            # buy at open price of next candle (simulate)
            price = row["open"]
            qty = allocation / price
            if qty > 0:
                self.position += qty
                self.cash -= qty * price
                self.position_entry_price = price
                # print(f"BUY at {price:.2f}, qty={qty:.6f}")
        # Exit conditions: stop-loss or take-profit evaluated on this candle's high/low
        if self.position > 0:
            current_price = row["close"]
            entry = self.position_entry_price
            if entry is None:
                entry = current_price
            ret = (current_price - entry) / (entry + 1e-9)
            if ret <= -CONFIG["stop_loss_pct"] or ret >= CONFIG["take_profit_pct"]:
                # sell all
                price = row["open"]  # assume we exit at open of this candle (conservative)
                self.cash += self.position * price
                # print(f"SELL at {price:.2f}, pnl={(price-entry)*self.position:.2f}")
                self.position = 0.0
                self.position_entry_price = None
        # compute equity
        eq = self.cash + self.position * row["close"]
        self.equity_curve.append(eq)

    def run(self):
        # skip initial rows until features stable
        start = max(30, CONFIG["volume_zscore_window"])
        for i in range(start, len(self.df)-CONFIG["label_forward_steps"]):
            self.step(i)
        return pd.Series(self.equity_curve)


# ----------------------------
# Utility metrics
# ----------------------------
def compute_metrics(equity_series):
    returns = equity_series.pct_change().fillna(0)
    total_return = equity_series.iloc[-1] / equity_series.iloc[0] - 1.0
    # simple max drawdown
    cum_max = equity_series.cummax()
    drawdown = (equity_series - cum_max) / cum_max
    max_dd = drawdown.min()
    # simple Sharpe (annualization approximate for hourly series)
    if len(returns) > 1:
        sharpe = (returns.mean() / (returns.std() + 1e-9)) * math.sqrt(24*365)
    else:
        sharpe = 0.0
    return {"total_return": total_return, "max_drawdown": max_dd, "sharpe": sharpe}


# ----------------------------
# Main pipeline
# ----------------------------
def main():
    df = load_data()
    df, feature_cols = engineer_features_and_labels(df)
    # Quick check: count whale events
    whale_count = int(df["is_whale_event"].sum())
    print(f"Detected {whale_count} whale-labeled candles in dataset (historical).")
    # Train model
    model = train_model(df, feature_cols)
    # Run simulator
    sim = Simulator(df, model, feature_cols)
    equity = sim.run()
    if len(equity) == 0:
        print("Nothing simulated (too short dataset). Increase data length.")
        return
    equity.index = range(len(equity))
    metrics = compute_metrics(equity)
    print("Simulation metrics:")
    print(metrics)
    # Plot equity curve
    plt.figure(figsize=(10, 5))
    plt.plot(equity.values)
    plt.title("Equity Curve (simulated)")
    plt.xlabel("Steps")
    plt.ylabel("Equity (USD)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Show confusion matrix and feature importances on the training data (for resume display)
    # feature importance
    importances = model.feature_importances_
    fi = pd.Series(importances, index=feature_cols).sort_values(ascending=False)
    print("\nTop feature importances:")
    print(fi)
    # If you want to export model or equity results:
    os.makedirs("output", exist_ok=True)
    equity.to_csv("output/equity_curve.csv", index=False)
    fi.to_csv("output/feature_importances.csv")
    print("Saved equity_curve.csv and feature_importances.csv to ./output/")

if __name__ == "__main__":
    main()
