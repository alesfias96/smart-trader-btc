# core_rl.py
import random
import collections
from typing import Tuple, List

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import joblib

from config import CFG


# -------------------------
# 1) Modello: Q-network
# -------------------------
class Net(nn.Module):
    def __init__(self, input_dim: int, n_actions: int = 3):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, n_actions)
        )

    def forward(self, x):
        return self.fc(x)


# -------------------------
# 2) Dati: download + feature
# -------------------------
def download_df(ticker: str, start: str, end: str) -> pd.DataFrame:
    data = yf.download(ticker, start=start, end=end)
    df = data[["Close"]].copy()
    df["SMA_7"] = df["Close"].rolling(7).mean()
    df["SMA_30"] = df["Close"].rolling(30).mean()
    df["Returns"] = np.log(df["Close"] / df["Close"].shift(1))
    df.dropna(inplace=True)
    return df


def add_sentiment_column(df: pd.DataFrame) -> pd.DataFrame:
    # Nel training offline mettiamo sentiment finto (0.0)
    # In live userai sentiment vero.
    df = df.copy()
    df["Sentiment"] = 0.0
    return df


def fit_scaler(df: pd.DataFrame) -> Tuple[MinMaxScaler, np.ndarray]:
    features = df[["Close", "SMA_7", "SMA_30", "Returns", "Sentiment"]].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(features)
    return scaler, scaled


def save_scaler(scaler: MinMaxScaler, path: str) -> None:
    joblib.dump(scaler, path)


def load_scaler(path: str) -> MinMaxScaler:
    return joblib.load(path)


def get_state(scaled_data: np.ndarray, t: int, window_size: int) -> torch.Tensor:
    """
    scaled_data: (N, n_features)
    ritorna tensor (1, window_size*n_features)
    """
    window = scaled_data[t:t + window_size]
    x = torch.tensor(window, dtype=torch.float32).reshape(1, -1)
    return x


# -------------------------
# 3) Ambiente di trading (paper reward)
# -------------------------
class MarketEnv:
    """
    Ambiente semplice:
    - action 2 = BUY => aggiunge un buy_price in inventory
    - action 0 = SELL => se inventory non vuota, reward = price - buy_price
    - action 1 = HOLD => reward 0
    """
    def __init__(self, prices: np.ndarray):
        self.prices = prices
        self.step_idx = 0
        self.inventory = []

    def reset(self):
        self.step_idx = 0
        self.inventory = []

    def step(self, action: int) -> Tuple[float, bool]:
        price = float(self.prices[self.step_idx])
        reward = 0.0

        if action == 2:  # BUY
            self.inventory.append(price)
        elif action == 0 and len(self.inventory) > 0:  # SELL
            buy_price = self.inventory.pop(0)
            reward = price - buy_price

        self.step_idx += 1
        done = self.step_idx >= (len(self.prices) - 1)
        return reward, done


# -------------------------
# 4) Replay Buffer
# -------------------------
Transition = collections.namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buf = collections.deque(maxlen=capacity)

    def push(self, *args):
        self.buf.append(Transition(*args))

    def sample(self, batch_size: int) -> List[Transition]:
        return random.sample(self.buf, batch_size)

    def __len__(self):
        return len(self.buf)


# -------------------------
# 5) Epsilon schedule
# -------------------------
def epsilon_by_step(step: int) -> float:
    # decay esponenziale “soft”
    # eps = eps_end + (eps_start - eps_end) * exp(-step/decay)
    import math
    return CFG.EPS_END + (CFG.EPS_START - CFG.EPS_END) * math.exp(-step / CFG.EPS_DECAY_STEPS)


# -------------------------
# 6) DQN update: QUELLO MANCANTE NELLA LEZIONE
# -------------------------
def dqn_update(
    q_net: Net,
    target_net: Net,
    optimizer: optim.Optimizer,
    batch: List[Transition],
    gamma: float
) -> float:
    """
    Implementazione standard DQN:
    target = r + gamma * max_a' Q_target(s', a') * (1 - done)
    loss = MSE(Q(s,a), target)
    """
    states = torch.cat([tr.state for tr in batch], dim=0)              # (B, input_dim)
    actions = torch.tensor([tr.action for tr in batch], dtype=torch.long).unsqueeze(1)  # (B,1)
    rewards = torch.tensor([tr.reward for tr in batch], dtype=torch.float32).unsqueeze(1)  # (B,1)
    next_states = torch.cat([tr.next_state for tr in batch], dim=0)     # (B, input_dim)
    dones = torch.tensor([tr.done for tr in batch], dtype=torch.float32).unsqueeze(1)   # (B,1)

    # Q(s,a) predetto
    q_values = q_net(states).gather(1, actions)  # (B,1)

    # max Q_target(s',a')
    with torch.no_grad():
        next_q_max = target_net(next_states).max(dim=1, keepdim=True).values  # (B,1)
        target = rewards + gamma * next_q_max * (1.0 - dones)

    loss = nn.MSELoss()(q_values, target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return float(loss.item())


# -------------------------
# 7) Training end-to-end + salvataggio
# -------------------------
def train_and_save():
    df = download_df(CFG.TICKER, CFG.START, CFG.END)
    df = add_sentiment_column(df)
    scaler, scaled_data = fit_scaler(df)

    prices = df["Close"].to_numpy(dtype=float).reshape(-1)
    env = MarketEnv(prices)

    input_dim = CFG.WINDOW_SIZE * CFG.N_FEATURES
    q_net = Net(input_dim=input_dim, n_actions=3)
    target_net = Net(input_dim=input_dim, n_actions=3)
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(q_net.parameters(), lr=CFG.LR)
    buffer = ReplayBuffer(CFG.MEMORY_SIZE)

    global_step = 0

    for ep in range(CFG.EPISODES):
        env.reset()
        total_reward = 0.0

        # stato iniziale
        state = get_state(scaled_data, 0, CFG.WINDOW_SIZE)

        for t in range(len(scaled_data) - CFG.WINDOW_SIZE - 1):
            eps = epsilon_by_step(global_step)

            # epsilon-greedy
            if random.random() < eps:
                action = random.randint(0, 2)
            else:
                with torch.no_grad():
                    action = int(torch.argmax(q_net(state), dim=1).item())

            # step env (reward/done)
            reward, done = env.step(action)
            total_reward += reward

            next_state = get_state(scaled_data, t + 1, CFG.WINDOW_SIZE)

            buffer.push(state, action, reward, next_state, done)

            state = next_state
            global_step += 1

            # update rete se abbastanza esperienze
            if len(buffer) >= CFG.BATCH_SIZE:
                batch = buffer.sample(CFG.BATCH_SIZE)
                _loss = dqn_update(q_net, target_net, optimizer, batch, CFG.GAMMA)

            # update target network periodico
            if global_step % CFG.TARGET_UPDATE_EVERY == 0:
                target_net.load_state_dict(q_net.state_dict())

            if done:
                break

        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"Episode {ep+1}/{CFG.EPISODES} | Total reward (profit): {total_reward:.2f} | eps={epsilon_by_step(global_step):.3f}")

    # Salvataggi
    torch.save(q_net.state_dict(), CFG.MODEL_PATH)
    save_scaler(scaler, CFG.SCALER_PATH)
    print(f"\nSalvati: {CFG.MODEL_PATH} e {CFG.SCALER_PATH}")


# -------------------------
# 8) Funzione live: ultimi dati -> stato tecnico normalizzato
# -------------------------
def get_latest_market_state(ticker: str, window_size: int, scaler_path: str) -> Tuple[List[float], float]:
    """
    Restituisce:
    - tech_flat: lista lunga window_size*(N_FEATURES-1) (qui: 5*4=20)
    - last_close: prezzo attuale (non normalizzato) per logging
    """
    # scarica abbastanza giorni per SMA30
    data = yf.download(ticker, period="90d", interval="1d")
    df = data[["Close"]].copy()
    df["SMA_7"] = df["Close"].rolling(7).mean()
    df["SMA_30"] = df["Close"].rolling(30).mean()
    df["Returns"] = np.log(df["Close"] / df["Close"].shift(1))
    df.dropna(inplace=True)

    # --- FIX DEFINITIVO per FutureWarning pandas ---
    close_col = df["Close"]

    # Se per qualche motivo è una DataFrame (1 colonna), estrai la Series
    if hasattr(close_col, "iloc") and hasattr(close_col, "shape") and len(close_col.shape) > 1:
        close_col = close_col.iloc[:, 0]

    last_close = float(close_col.iloc[-1])

    # sentiment in live lo aggiungiamo DOPO (qui mettiamo 0 nella colonna sentiment per far scalare coerente)
    df["Sentiment"] = 0.0

    scaler = load_scaler(scaler_path)
    feats = df[["Close", "SMA_7", "SMA_30", "Returns", "Sentiment"]].values
    scaled = scaler.transform(feats)

    # prendi ultime window_size righe e togli l’ultima colonna (Sentiment) perché lo mettiamo separatamente
    window = scaled[-window_size:]  # (window_size, 5)
    tech_only = window[:, :4]       # (window_size, 4)
    tech_flat = tech_only.reshape(-1).astype(float).tolist()  # 20 valori

    return tech_flat, last_close

def analyze_results(history: np.ndarray, initial_balance: float) -> None:
    """
    Stampa:
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
