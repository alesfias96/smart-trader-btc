# SMART TRADER (Bitcoin)

Trading research project focused on Bitcoin market data.
Goal: build a complete pipeline for feature engineering, ML experiments and strategy evaluation
using backtesting and standard performance metrics.

This repository is intended as a learning + portfolio project and does not provide financial advice.

## Goals
- Collect and store BTC market data in a SQL database
- Engineer meaningful features (indicators, returns, volatility, sentiment signals)
- Train and validate ML models responsibly (no leakage)
- Backtest strategies and evaluate performance with standard metrics

## Planned pipeline
1. Data ingestion (OHLCV + optional sentiment)
2. SQL storage
3. Feature engineering (RSI, MACD, Bollinger Bands, returns, etc.)
4. Baseline modeling
   - classification (next movement up/down)
   - or regression (future return)
5. Backtesting engine
6. Performance evaluation
   - Total return
   - Sharpe ratio
   - Maximum drawdown

## Tech stack
- Python (Pandas, NumPy)
- SQL
- scikit-learn / PyTorch (experiments)
- Jupyter Notebooks
- Matplotlib

## Project structure
```text
smart-trader-btc/
  sql/                  # schema.sql
  notebooks/
  src/                  # data pipeline, features, backtest utils
  data/                 # empty (dataset not included)
  reports/
  README.md
  requirements.txt
```

## Status
Work in progress.

## !Disclaimer
This project is for educational purposes only and is not financial advice.
