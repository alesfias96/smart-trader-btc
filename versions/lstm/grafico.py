import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggiunge indicatori tecnici:
    - RSI(14)
    - MACD(12,26,9) (solo MACD line)
    - Bollinger Width (20,2)
    - Returns logaritmico
    """
    df = df.copy()
    close = df["Close"].astype(float)

    # Returns (log)
    df["Returns"] = np.log(close / close.shift(1))

    # RSI(14) - Wilder
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    # Wilder smoothing (EMA con alpha=1/14)
    period = 14
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()

    rs = avg_gain / (avg_loss.replace(0.0, np.nan))
    df["RSI"] = 100.0 - (100.0 / (1.0 + rs))

    # MACD (12, 26, 9) - usiamo la MACD line (EMA12-EMA26)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    # Signal e Histogram non servono se nel progetto usi solo MACD line    

    # Bollinger Bands (20,2) e Width
    length = 20
    ma = close.rolling(length).mean()
    std = close.rolling(length).std(ddof=0)
    upper = ma + 2.0 * std
    lower = ma - 2.0 * std
    df["BB_Width"] = (upper - lower) / close

    df.dropna(inplace=True)
    return df

ticker = "BTC-USD"

data = yf.download(ticker, start="2023-01-01", end="2025-12-01")

df = data[["Close"]].copy()

df = add_indicators(df)

df.dropna(inplace=True)

print(df.head())

fig, axes = plt.subplots(
    nrows=4, ncols=1,
    sharex=True,
    figsize=(14, 10),
    gridspec_kw={"height_ratios": [2.2, 1.2, 1.2, 1.2], "hspace": 0.08}
)

# --- 1) Close + Bollinger Width ---
ax0 = axes[0]
ax0.plot(df.index, df["Close"], color="red", linewidth=1.8, label="Close")
ax0.set_ylabel("Price")

ax0b = ax0.twinx()
ax0b.plot(df.index, df["BB_Width"], color="tab:purple", linewidth=1.4, label="Bollinger Width")
ax0b.set_ylabel("BB Width")

# Legenda combinata (due assi)
lines0, labels0 = ax0.get_legend_handles_labels()
lines0b, labels0b = ax0b.get_legend_handles_labels()
ax0.legend(lines0 + lines0b, labels0 + labels0b, loc="upper left")

# --- 2) RSI ---
ax1 = axes[1]
ax1.plot(df.index, df["RSI"], color="tab:blue", linewidth=1.4, label="RSI")
ax1.axhline(70, color="gray", linestyle="--", linewidth=1)
ax1.axhline(30, color="gray", linestyle="--", linewidth=1)
ax1.set_ylim(0, 100)
ax1.set_ylabel("RSI")
ax1.legend(loc="upper left")

# --- 3) MACD ---
ax2 = axes[2]
ax2.plot(df.index, df["MACD"], color="tab:green", linewidth=1.4, label="MACD")
ax2.axhline(0, color="gray", linestyle="--", linewidth=1)
ax2.set_ylabel("MACD")
ax2.legend(loc="upper left")

# --- 4) Log Returns ---
ax3 = axes[3]
ax3.bar(df.index, df["Returns"], color="tab:orange", linewidth=1.2, label="Log Returns")
ax3.axhline(0, color="gray", linestyle="--", linewidth=1)
ax3.set_ylabel("Returns")
ax3.legend(loc="upper left")

axes[-1].set_xlabel("Time")
plt.show()
