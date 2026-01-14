# Roadmap — Smart Trader BTC (DQN)

This roadmap outlines planned improvements for the project, with priorities and suggested milestones.
The goal is to evolve from a **baseline DQN (MLP)** to a more robust system with **temporal modeling (LSTM)** and future upgrades (e.g., Transformers, anomaly detection, safety/risk controls).

---

## Current Status

- **MLP (Baseline)**: working end-to-end pipeline (train → save model+scaler → backtest → live simulation).
- **LSTM (Improved)**: first major upgrade focused on capturing temporal dependencies.

---

## Milestone 1 — Reproducibility & Experiment Tracking (High Priority)

**Goal:** make results comparable and repeatable across runs.

- [ ] Fix random seeds (NumPy / PyTorch) for reproducibility (where applicable).
- [ ] Standardize output structure (`outputs/`) for:
  - trained models (`.pth`)
  - scalers (`.pkl`)
  - backtest trades (`.csv`)
  - live logs (`.json`)
- [ ] Save a lightweight run summary (JSON/CSV) after training/backtest:
  - version (mlp/lstm), date range, features, window_size
  - metrics (return, sharpe, max drawdown)
  - hyperparameters (episodes, gamma, lr, epsilon schedule)
- [ ] Add a consistent naming convention for artifacts:
  - `dqn_btc_<version>_<start>_<end>_<timestamp>.pth`
  - `scaler_<version>_<timestamp>.pkl`

---

## Milestone 2 — Better Evaluation (High Priority)

**Goal:** measure performance more realistically.

- [ ] Add transaction costs (fees + slippage).
- [ ] Add position sizing assumptions (fixed size, fraction of equity, etc.).
- [ ] Evaluate on multiple periods / regimes (bull, bear, range).
- [ ] Walk-forward evaluation:
  - train on A, test on B
  - slide window and repeat
- [ ] Add additional metrics:
  - win rate
  - profit factor
  - average trade return
  - exposure time
  - Calmar ratio

---

## Milestone 3 — Risk Management & Safety System (High Priority)

**Goal:** prevent dangerous behavior and reduce tail risk.

- [ ] Hard risk rules (guardrails):
  - max daily loss / max drawdown stop
  - max position size
  - cooldown after consecutive losses
- [ ] Volatility-aware constraints:
  - reduce exposure when volatility spikes
  - pause trading during abnormal conditions
- [ ] Data integrity checks:
  - missing / NaN values
  - sudden price gaps
  - out-of-order timestamps
- [ ] Fail-safe mode:
  - if model/scaler missing or corrupted → default to HOLD
  - if price feed unstable → default to HOLD

---

## Milestone 4 — Anomaly Detection & Anomaly Blocking (High Priority)

**Goal:** detect abnormal market situations and block trades.

- [ ] Rule-based anomaly flags:
  - extreme returns (z-score / MAD)
  - candle gap detection
  - abnormal volume (if available)
- [ ] ML-based anomaly detection (optional):
  - Isolation Forest
  - One-Class SVM
  - Autoencoder reconstruction error
- [ ] Trade blocking logic:
  - if anomaly flag is ON → do not trade (HOLD) or reduce risk
  - log anomalies to JSON and report frequency

---

## Milestone 5 — RL Algorithm Upgrades (Medium Priority)

**Goal:** improve stability and sample efficiency.

- [ ] Double DQN (reduce Q-value overestimation).
- [ ] Dueling DQN (better state-value vs advantage separation).
- [ ] Prioritized Experience Replay (PER).
- [ ] N-step returns.
- [ ] Soft/Target network update tuning.

---

## Milestone 6 — Feature Engineering & Market Regime Modeling (Medium Priority)

**Goal:** make the input state more informative and regime-aware.

- [ ] Confirm and standardize features across versions:
  - Close / Returns
  - RSI / MACD
  - Bollinger Band width
  - (optional) Sentiment
- [ ] Add regime indicators:
  - volatility regime (low/medium/high)
  - trend regime (ADX / moving average slope)
  - range vs breakout signals
- [ ] Separate feature scaling strategy for train vs live consistency.

---

## Milestone 7 — Transformer-Based Model (Medium / Research)

**Goal:** move beyond LSTM to attention-based temporal modeling.

- [ ] Start with a lightweight Transformer encoder for time-series windows.
- [ ] Compare vs LSTM on the same exact pipeline and features.
- [ ] Evaluate latency/CPU requirements for live inference.
- [ ] Add ablation tests:
  - attention depth, heads, window sizes
  - impact of each feature group

---

## Milestone 8 — Engineering Improvements (Medium)

**Goal:** improve maintainability and production-readiness without rewriting the algorithm.

- [ ] Unify CLI arguments across scripts (train/backtest/live).
- [ ] Add clear error messages and safe defaults (especially live).
- [ ] Improve logging format consistency across versions.
- [ ] Add a simple “smoke test” checklist in docs.

---

## Milestone 9 — Comparison Notebook & Report (Portfolio Priority)

**Goal:** create a clean, international, portfolio-ready comparison.

- [ ] `notebooks/compare_mlp_vs_lstm.ipynb`
  - load backtest CSVs / summaries
  - compare Return, Sharpe, Max Drawdown
  - plot equity curves
- [ ] `reports/` summary with figures:
  - performance overview
  - key findings
  - limitations and next steps

---

## Nice-to-Have (Later)

- [ ] Multi-asset support (ETH, SOL, etc.).
- [ ] Timeframe options (1h, 4h, daily).
- [ ] Model ensembling (signal voting).
- [ ] Continuous learning (carefully, with safeguards).

---

## Guiding Principles

- Keep **MLP** as the stable baseline reference.
- Treat **LSTM** as the first production-grade temporal upgrade.
- Add safety systems before adding complexity.
- Prefer measurable improvements (metrics + comparisons) over architecture hype.
