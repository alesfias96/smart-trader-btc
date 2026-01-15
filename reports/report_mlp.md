# Smart Trader BTC (DQN) — MLP Baseline Report

**Asset:** BTC-USD  
**Approach:** Deep Reinforcement Learning (DQN)  
**Stable Version:** `versions/mlp/`  
**Experimental Version:** `versions/lstm/` (WIP / unstable)

> Educational portfolio project. Not financial advice.

---

## 1) Overview

This repository implements a simplified RL trading agent for **BTC-USD** using a **Deep Q-Network (DQN)**.
The **MLP baseline** is the recommended stable version: it uses a feed-forward network with classic technical features (SMAs + returns).

Workflows:
- **Train:** `train.py` → saves `dqn_btc.pth` and `scaler.pkl` into `versions/mlp/outputs/`
- **Backtest:** `backtest_MLP.py` (or `backtest.py` depending on your file) → prints metrics and exports a CSV log
- **Live simulation:** `live_bot.py` → paper trading + optional Telegram + JSON logs

---

## 2) Data & Features (MLP)

Data is sourced from Yahoo Finance via `yfinance` (`BTC-USD`).

MLP feature set:
- Close
- SMA_7
- SMA_30
- Returns
- (optional) Sentiment placeholder (0.0)

---

## 3) Backtest Results (MLP)

Below are selected evaluation windows produced by the current MLP baseline run.

| Window | Total Return | Sharpe Ratio | Max Drawdown | Notes |
|---|---:|---:|---:|---|
| 2012–2025 | 16181.75% | 1.15 | -57.95% | Long window spanning multiple regimes; high drawdown risk |
| 2018–2021 | 417.91% | 1.26 | -48.24% | Strong performance in this regime; still high risk |
| 2021–2025 | 0.00% | 0.00 | 0.00% | Policy stayed mostly **HOLD** (no meaningful trades) |

**Interpretation**
- Performance is **regime-dependent**: results can be strong in certain market phases and collapse (HOLD-only) in others.
- High returns can coexist with large drawdowns: risk management is a key next step.

---

## 4) Limitations

- Simplified execution: liquidity/spread/slippage are not fully modeled.
- Transaction costs may be simplified.
- RL training is stochastic: results can vary across runs (initialization + exploration).
- No walk-forward evaluation yet (train/test rolling windows).

---

## 5) Next Steps (Roadmap)

Recommended improvements (see `docs/ROADMAP.md`):
- Walk-forward evaluation (rolling train/test)
- Add explicit costs (fees + slippage) and risk constraints (max DD stop, max position)
- Stabilize training (fixed seeds, multi-run averages)
- Upgrade algorithms (Double DQN, Dueling, Prioritized Replay)
- Improve sequence modeling (LSTM/Transformer) once the baseline is stable
