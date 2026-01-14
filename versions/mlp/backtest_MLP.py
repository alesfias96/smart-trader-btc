from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import torch
import yfinance as yf

from config import CFG

from core_rl import (
    Net,
    download_df,
    add_sentiment_column,
    load_scaler,
)

# Sentiment viene aggiunto (placeholder in training; reale in live dal controller)
ALL_FEATURE_COLS = ["Close", "SMA_7", "SMA_30", "Returns"]


@dataclass
class BacktestResult:
    history: np.ndarray
    actions: np.ndarray  # 0 hold, 1 buy, 2 sell
    prices: np.ndarray
    dates: np.ndarray

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ticker", default=getattr(CFG, "TICKER", "BTC-USD"))
    p.add_argument("--start", default=getattr(CFG, "BACKTEST_START", None))
    p.add_argument("--end", default=getattr(CFG, "BACKTEST_END", None))
    p.add_argument("--initial-balance", type=float, default=getattr(CFG, "PAPER_START_CAPITAL", 10000.0))
    p.add_argument("--commission", type=float, default=getattr(CFG, "COMMISSION", 0.001))
    p.add_argument("--export-csv", default=None, help="Percorso CSV per salvare equity curve (opzionale)")
    return p.parse_args()

def _default_dates_if_missing(start: Optional[str], end: Optional[str]) -> tuple[str, str]:
    if start and end:
        return start, end
    # fallback: ultimi 365 giorni fino a oggi
    today = date.today()
    s = today - timedelta(days=365)
    return str(s), str(today)

def download_test_df(ticker: str, start: str, end: str) -> pd.DataFrame:
    df = download_df(ticker=ticker, start=start, end=end)
    df = add_sentiment_column(df[ALL_FEATURE_COLS])  # placeholder 0.0 (storico sentiment non disponibile)
    return df

def run_backtest_greedy(
    model: torch.nn.Module,
    df: pd.DataFrame,
    scaler,
    window_size: int,
    initial_balance: float,
    commission: float,
) -> BacktestResult:
    # scala con lo scaler del training
    scaled = scaler.transform(df.values)

    balance = float(initial_balance)
    position_qty = 0.0  # quantità di asset
    history = []
    actions = []

    closes = np.asarray(df["Close"].to_numpy(), dtype=float).reshape(-1)
    dates = np.asarray(df.index.to_numpy()).reshape(-1)

    # loop: a ogni t, lo state è finestra [t:t+W]
    max_t = len(scaled) - window_size - 1
    if max_t <= 0:
        raise RuntimeError(f"Dati insufficienti dopo indicatori/dropna: righe={len(scaled)}, window={window_size}.")

    model.eval()
    with torch.no_grad():
        for t in range(max_t):
            window = scaled[t : t + window_size]  # (W,F)
            state = torch.tensor(window, dtype=torch.float32).reshape(1, -1)  # (1,W,F)

            q = model(state)  # (1,3)
            action = int(torch.argmax(q, dim=1).item())
            actions.append(action)

            # prezzo "corrente" = close dell'ultimo giorno della finestra
            current_price = float(np.asarray(closes[t + window_size]).item())

            if action == 2 and position_qty == 0.0:  # BUY
                position_qty = balance / current_price
                balance *= (1.0 - commission)

            elif action == 0 and position_qty > 0.0:  # SELL
                balance = position_qty * current_price
                balance *= (1.0 - commission)
                position_qty = 0.0

            portfolio_value = balance if position_qty == 0.0 else position_qty * current_price
            history.append(portfolio_value)

    return BacktestResult(
        history=np.array(history, dtype=float),
        actions=np.array(actions, dtype=int),
        prices=closes[window_size : window_size + len(history)],
        dates=dates[window_size : window_size + len(history)],
    )

def main():
    args = parse_args()

    start, end = _default_dates_if_missing(args.start, args.end)

    scaler = load_scaler(CFG.SCALER_PATH)

    # istanzia modello e carica pesi
    input_dim = CFG.WINDOW_SIZE * CFG.N_FEATURES

    model = Net(input_dim=input_dim, n_actions=3)
    state_dict = torch.load(CFG.MODEL_PATH)
    model.load_state_dict(state_dict)

    df_test = download_test_df(args.ticker, start, end)

    res = run_backtest_greedy(
        model=model,
        df=df_test,
        scaler=scaler,
        window_size=CFG.WINDOW_SIZE,
        initial_balance=args.initial_balance,
        commission=args.commission,
    )

    # metriche: se in core_rl c'è analyze_results la usiamo
    try:
        from core_rl import analyze_results  # type: ignore
        analyze_results(res.history, initial_balance=args.initial_balance)
    except Exception:
        # fallback minimale
        total_return = (res.history[-1] - args.initial_balance) / args.initial_balance * 100.0
        print(f"Final balance: {res.history[-1]:.2f}")
        print(f"Total return: {total_return:.2f}%")

    if args.export_csv:
        out = pd.DataFrame(
            {
                "Date": pd.to_datetime(np.asarray(res.dates).reshape(-1)),
                "Close": np.asarray(res.prices).reshape(-1),
                "Action": np.asarray(res.actions).reshape(-1),
                "Equity": np.asarray(res.history).reshape(-1),
            }
        )
        out.to_csv(args.export_csv, index=False)
        print(f"Salvato CSV: {args.export_csv}")

if __name__ == "__main__":
    main()


