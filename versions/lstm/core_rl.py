# core_rl.py
"""
Deep RL core (DQN) per trading.

Versione coerente con gli altri moduli del progetto:
- Rete Q basata su LSTM: TradingLSTM
- Stato in forma sequenziale: (batch, window_size, n_features)
- Feature tecniche: Close, RSI, MACD, BB_Width, Returns
- Sentiment come 6ª feature (replicata sulla finestra) -> gestita dal controller in live,
  e in training inserita come placeholder (0.0) tramite add_sentiment_column()
- Normalizzazione con StandardScaler, salvata su scaler.pkl
- ReplayBuffer + epsilon schedule + target network update

Azioni:
0 = HOLD
1 = BUY
2 = SELL
"""

from __future__ import annotations

import math
import random
import collections
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

import yfinance as yf
from sklearn.preprocessing import StandardScaler
import joblib


# -------------------------
# Config (import dal progetto)
# -------------------------
from config import CFG


# -------------------------
# Feature schema
# -------------------------
TECH_FEATURE_COLS = ["Close", "RSI", "MACD", "BB_Width", "Returns"]
# Sentiment viene aggiunto (placeholder in training; reale in live dal controller)
ALL_FEATURE_COLS = TECH_FEATURE_COLS + ["Sentiment"]


# -------------------------
# Rete neurale (LSTM)
# -------------------------
class TradingLSTM(nn.Module):
    def __init__(self, n_features: int, hidden_dim: int, n_actions: int, n_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, window_size, n_features)
        h0 = torch.zeros(self.n_layers, x.size(0), self.hidden_dim, device=x.device)
        c0 = torch.zeros(self.n_layers, x.size(0), self.hidden_dim, device=x.device)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(out[:, -1, :])  # (batch, n_actions)


# Compatibilità: alcuni file importano Net dal core
Net = TradingLSTM


# -------------------------
# Replay Buffer
# -------------------------
Transition = collections.namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buf = collections.deque(maxlen=capacity)

    def push(self, state: torch.Tensor, action: int, reward: float, next_state: torch.Tensor, done: bool):
        self.buf.append(Transition(state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> List[Transition]:
        return random.sample(self.buf, batch_size)

    def __len__(self) -> int:
        return len(self.buf)


# -------------------------
# Feature engineering (indicatori)
# -------------------------
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


def add_sentiment_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Placeholder per training offline: sentiment non disponibile storicamente -> 0.0
    In live viene fornito dal controller e replicato sulla finestra.
    """
    df = df.copy()
    df["Sentiment"] = 0.0
    return df


# -------------------------
# Download / scaler
# -------------------------
def download_df(ticker: str, start: str, end: str) -> pd.DataFrame:
    data = yf.download(ticker, start=start, end=end, progress=False)
    df = data[["Close"]].copy()
    df = add_indicators(df)
    df = add_sentiment_column(df)
    return df


def fit_scaler(df: pd.DataFrame) -> Tuple[StandardScaler, np.ndarray]:
    feats = df[ALL_FEATURE_COLS].values
    scaler = StandardScaler()
    scaled = scaler.fit_transform(feats)
    return scaler, scaled


def save_scaler(scaler: StandardScaler, path: str) -> None:
    joblib.dump(scaler, path)


def load_scaler(path: str) -> StandardScaler:
    return joblib.load(path)


# -------------------------
# Stato per la rete (sequenziale)
# -------------------------
def get_state(scaled_data: np.ndarray, t: int, window_size: int, device: torch.device) -> torch.Tensor:
    """
    scaled_data: (N, n_features)
    ritorna: (1, window_size, n_features)
    """
    window = scaled_data[t : t + window_size]
    x = torch.tensor(window, dtype=torch.float32, device=device).unsqueeze(0)
    return x


# -------------------------
# Ambiente di mercato (reward + SL/TP)
# -------------------------
class MarketEnv:
    def __init__(self, prices: np.ndarray, sl_percent: float = 0.02, tp_percent: float = 0.05, end_on_stoploss: bool = True):
        self.prices = prices.astype(float)
        self.sl_percent = sl_percent
        self.tp_percent = tp_percent
        self.end_on_stoploss = end_on_stoploss
        self.reset()

    def reset(self):
        self.current_step = 0
        self.has_position = False
        self.entry_price = 0.0
        self.prev_price = float(self.prices[0])

    def _profit(self, current_price: float) -> float:
        return float(current_price - self.entry_price)

    def step(self, action: int) -> Tuple[float, bool]:
        current_price = float(self.prices[self.current_step])
        reward = 0.0
        done = False

        # 0 HOLD
        if action == 1 and not self.has_position:  # BUY
            self.entry_price = current_price
            self.has_position = True

        elif action == 2 and self.has_position:  # SELL
            reward = self._profit(current_price)
            self.has_position = False
            self.entry_price = 0.0

        # SL / TP
        if self.has_position:
            reward += (current_price - self.prev_price) / self.prev_price
            price_change = (current_price - self.entry_price) / self.entry_price
            if price_change <= -self.sl_percent:  # STOP LOSS
                reward = -2.0
                self.has_position = False
                self.entry_price = 0.0
                if self.end_on_stoploss:
                    done = True
            elif price_change >= self.tp_percent:  # TAKE PROFIT
                reward = 1.0
                self.has_position = False
                self.entry_price = 0.0

        self.current_step += 1
        if self.current_step >= len(self.prices) - 1:
            done = True
        self.prev_price = current_price

        return reward, done


# -------------------------
# Epsilon schedule
# -------------------------
def epsilon_by_step(step: int) -> float:
    return CFG.EPS_END + (CFG.EPS_START - CFG.EPS_END) * math.exp(-step / CFG.EPS_DECAY_STEPS)


# -------------------------
# DQN update
# -------------------------
def dqn_update(
    q_net: nn.Module,
    target_net: nn.Module,
    optimizer: optim.Optimizer,
    batch: List[Transition],
    gamma: float,
) -> float:
    device = next(q_net.parameters()).device

    states = torch.cat([tr.state for tr in batch], dim=0).to(device)  # (B,W,F)
    actions = torch.tensor([tr.action for tr in batch], dtype=torch.long, device=device).unsqueeze(1)  # (B,1)
    rewards = torch.tensor([tr.reward for tr in batch], dtype=torch.float32, device=device).unsqueeze(1)  # (B,1)
    next_states = torch.cat([tr.next_state for tr in batch], dim=0).to(device)  # (B,W,F)
    dones = torch.tensor([tr.done for tr in batch], dtype=torch.float32, device=device).unsqueeze(1)  # (B,1)

    q_sa = q_net(states).gather(1, actions)  # (B,1)

    with torch.no_grad():
        next_q_max = target_net(next_states).max(dim=1, keepdim=True).values  # (B,1)
        target = rewards + gamma * next_q_max * (1.0 - dones)

    loss = nn.MSELoss()(q_sa, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return float(loss.item())


# -------------------------
# Train + save
# -------------------------
def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_and_save() -> None:
    device = get_device()

    df = download_df(CFG.TICKER, CFG.START, CFG.END)
    scaler, scaled_data = fit_scaler(df)

    prices = df["Close"].to_numpy(dtype=float).reshape(-1)
    env = MarketEnv(prices, sl_percent=getattr(CFG, "SL_PERCENT", 0.02), tp_percent=getattr(CFG, "TP_PERCENT", 0.05))

    n_features = len(ALL_FEATURE_COLS)
    n_actions = 3

    q_net = TradingLSTM(n_features=n_features, hidden_dim=CFG.HIDDEN_DIM, n_actions=n_actions, n_layers=CFG.N_LAYERS).to(device)
    target_net = TradingLSTM(n_features=n_features, hidden_dim=CFG.HIDDEN_DIM, n_actions=n_actions, n_layers=CFG.N_LAYERS).to(device)
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(q_net.parameters(), lr=CFG.LR)
    buffer = ReplayBuffer(CFG.MEMORY_SIZE)

    global_step = 0

    for ep in range(CFG.EPISODES):
        env.reset()
        total_reward = 0.0

        state = get_state(scaled_data, 0, CFG.WINDOW_SIZE, device)

        for t in range(len(scaled_data) - CFG.WINDOW_SIZE - 1):
            eps = epsilon_by_step(global_step)

            if random.random() < eps:
                action = random.randint(0, 2)
            else:
                with torch.no_grad():
                    q_vals = q_net(state)
                    action = int(torch.argmax(q_vals, dim=1).item())

            reward, done = env.step(action)
            total_reward += reward

            next_state = get_state(scaled_data, t + 1, CFG.WINDOW_SIZE, device)
            buffer.push(state, action, reward, next_state, done)

            state = next_state
            global_step += 1

            if len(buffer) >= CFG.BATCH_SIZE:
                batch = buffer.sample(CFG.BATCH_SIZE)
                _ = dqn_update(q_net, target_net, optimizer, batch, CFG.GAMMA)

            if global_step % CFG.TARGET_UPDATE_EVERY == 0:
                target_net.load_state_dict(q_net.state_dict())

            if done:
                break

        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"Episode {ep+1}/{CFG.EPISODES} | reward={total_reward:.2f} | eps={epsilon_by_step(global_step):.3f}")

    torch.save(q_net.state_dict(), CFG.MODEL_PATH)
    save_scaler(scaler, CFG.SCALER_PATH)
    print(f"Salvati: {CFG.MODEL_PATH} e {CFG.SCALER_PATH}")


# -------------------------
# Live helper: ultimo stato tecnico normalizzato
# -------------------------
def get_latest_market_state(ticker: str, window_size: int, scaler_path: str) -> Tuple[List[float], float]:
    """
    Restituisce:
    - tech_flat: lista con window_size * len(TECH_FEATURE_COLS) valori scalati (SENZA sentiment!)
    - last_close: prezzo reale (non scalato)
    """
    data = yf.download(ticker, period="240d", interval="1d", progress=False)
    df = data[["Close"]].copy()
    df = add_indicators(df)

    last_close = float(df["Close"].iloc[-1])

    # Per trasformare con lo scaler allenato su ALL_FEATURE_COLS,
    # dobbiamo fornire anche la colonna Sentiment (qui 0.0 placeholder)
    df = add_sentiment_column(df)

    scaler = load_scaler(scaler_path)
    scaled = scaler.transform(df[ALL_FEATURE_COLS].values)

    if len(scaled) < window_size:
        raise ValueError(f"Dati insufficienti: servono almeno {window_size} righe dopo dropna(), ne ho {len(scaled)}.")

    window = scaled[-window_size:]  # (W, 6)

    # estraiamo SOLO le feature tecniche (prime len(TECH_FEATURE_COLS))
    tech_window = window[:, : len(TECH_FEATURE_COLS)]  # (W,5)
    tech_flat = tech_window.reshape(-1).astype(float).tolist()  # W*5

    return tech_flat, last_close


# -------------------------
# Backtest + analisi (Algoritmo 49 + 50)
# -------------------------
def run_backtest(
    model: nn.Module,
    data_test: pd.DataFrame,
    scaler: StandardScaler,
    window_size: int,
    initial_balance: float = 10000.0,
    commission: float = 0.001,
    device: Optional[torch.device] = None,
) -> np.ndarray:
    """
    Backtest su dati mai visti. Usa il modello in modalità greedy (argmax).
    """
    if device is None:
        device = next(model.parameters()).device

    df = data_test[["Close"]].copy()
    df = add_indicators(df)
    df = add_sentiment_column(df)

    scaled = scaler.transform(df[ALL_FEATURE_COLS].values)

    balance = float(initial_balance)
    position_qty = 0.0
    history = []

    for t in range(len(scaled) - window_size - 1):
        state = get_state(scaled, t, window_size, device)  # (1,W,F)
        with torch.no_grad():
            q_values = model(state)
            action = int(torch.argmax(q_values, dim=1).item())

        current_price = float(df["Close"].iloc[t + window_size])

        if action == 1 and position_qty == 0.0:  # BUY
            position_qty = balance / current_price
            balance -= balance * commission

        elif action == 2 and position_qty > 0.0:  # SELL
            balance = position_qty * current_price
            balance -= balance * commission
            position_qty = 0.0

        portfolio_value = balance if position_qty == 0.0 else position_qty * current_price
        history.append(portfolio_value)

    return np.array(history, dtype=float)


def analyze_results(history: np.ndarray, initial_balance: float) -> None:
    """
    (Algoritmo 50) Stampa:
    - Rendimento totale
    - Sharpe ratio (annualizzato, 252)
    - Max drawdown
    """
    if len(history) < 2:
        print("History troppo corta per analisi.")
        return

    returns = np.diff(history) / history[:-1]
    total_return = (history[-1] - initial_balance) / initial_balance * 100.0

    # evita divisione per 0
    if np.std(returns) == 0:
        sharpe_ratio = 0.0
    else:
        sharpe_ratio = float(np.mean(returns) / np.std(returns) * np.sqrt(252))

    peak = np.maximum.accumulate(history)
    drawdown = (history - peak) / peak
    max_drawdown = float(np.min(drawdown) * 100.0)

    print("--- Risultati Backtest ---")
    print(f"Rendimento Totale: {total_return:.2f}%")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Max Drawdown: {max_drawdown:.2f}%")
