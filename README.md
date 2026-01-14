# SMART TRADER BTC (DQN) — Baseline MLP & Improved LSTM

A Reinforcement Learning trading project (Deep Q-Network) for **BTC-USD**.

This repository contains **two independent versions** of the agent:

- **MLP version (Baseline)** → `versions/mlp/`
- **LSTM version (Improved)** → `versions/lstm/`

Each version works standalone: you run scripts directly from its folder and all Python files must remain in that same directory.

> ⚠️ Disclaimer: This project is for educational purposes only. It is not financial advice.

---

## Versions

### MLP (Baseline)
The MLP version is the **base implementation** of the DQN trading agent.  
It is simpler and provides a reference point for comparisons and benchmarking.

Folder:
- `versions/mlp/`

### LSTM (Improved)
The LSTM version includes improvements aimed at better capturing **temporal dependencies** in market data.

Folder:
- `versions/lstm/`

> The LSTM version is intended as the next step after the baseline MLP and represents the first major upgrade of the project.
> It is based also on different features (RSI, MACD, BB_Width, Returns) instead of (SMA_7, SMA_30, Returns)

---

## Repository Structure

```text
smart-trader-btc/
├── versions/
│   ├── mlp/
│   │   ├── train.py
│   │   ├── backtest.py
│   │   ├── live_bot.py
│   │   ├── core_rl.py
│   │   ├── config.py
│   │   └── ... other .py files
│   └── lstm/
│       ├── train.py
│       ├── backtest.py
│       ├── live_bot.py
│       ├── core_rl.py
│       ├── config.py
│       └── ... other .py files
├── requirements.txt
└── README.md
```

---

## Installation

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## How to Run (Training → Backtest → Live)

Choose a version first:

- MLP: `versions/mlp/`
- LSTM: `versions/lstm/`

> Important rule: all Python files of the chosen version must remain in the same folder.

---

## 1) Training

Run `train.py` inside the selected version folder.

### MLP Training

```bash
cd versions/mlp
python train.py
```

### LSTM Training

```bash
cd versions/lstm
python train.py
```

### Training behavior

- Training runs **50 episodes** by default.
- Every **10 episodes**, the terminal prints:
  - total reward (for the last 10 episodes)
  - current epsilon (exploration rate)

To change the number of episodes, edit the parameter in `config.py`.

### Output files after training

At the end of training, two files are saved **in the same folder**:

- `dqn_btc.pth` → trained model weights
- `scaler.pkl` → fitted scaler

---

## 2) Backtesting

After training, run `backtest.py` inside the same version folder.

### Example

```bash
cd versions/mlp
python backtest.py
```

(or run the same from `versions/lstm/`)

### Backtest results

After a few seconds, the terminal prints:

- **Total Return (%)**
- **Sharpe Ratio**
- **Max Drawdown (%)**

### Backtest CSV output

The script also saves a CSV file containing all trades executed during the backtest period.

- If no output path is specified, the CSV is saved in the same folder where the script is executed.

### Custom date range (optional)

If supported by your `backtest.py`, you can specify a date range, for example:

```bash
python backtest.py --start 2017-01-01 --end 2019-01-01
```

> If your script uses different argument names, check the `argparse` section inside `backtest.py`.

---

## 3) Live Trading Simulation (paper trading)

Run `live_bot.py` to simulate live trading with a virtual balance.

### Example

```bash
cd versions/mlp
python live_bot.py
```

(or run the same from `versions/lstm/`)

### Live behavior

- Every **5 minutes**, the bot prints a message in the terminal containing:
  - action: **BUY / SELL / HOLD**
  - current BTC-USD price
  - current held quantity
- Uses a **virtual initial capital of $10,000**
- Saves a JSON file containing the live log

### Telegram notifications (optional)

To enable Telegram notifications, set the following parameters in `config.py`:

- `TELEGRAM_TOKEN`
- `TELEGRAM_CHAT_ID`

> ⚠️ Do not commit real tokens or personal chat IDs to GitHub.

---

## Configuration

All main parameters are located in each version’s `config.py`, including:

- number of training episodes
- state window size
- features used
- epsilon schedule (exploration)
- trading settings
- Telegram configuration (optional)

---

## Roadmap (Future Improvements)

Planned upgrades include:

- Transformer-based architecture
- anomaly detection & anomaly blocking
- safety / risk management system
- additional market regime detection / filters
