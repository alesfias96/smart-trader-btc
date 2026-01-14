# config.py
from dataclasses import dataclass

@dataclass
class Config:
    TICKER: str = "BTC-USD"
    START: str = "2023-01-01"
    END: str = "2025-12-01"

    WINDOW_SIZE: int = 5
    # 4 feature tecniche + 1 sentiment
    N_FEATURES: int = 5

    # Training DQN
    EPISODES: int = 50
    BATCH_SIZE: int = 64
    MEMORY_SIZE: int = 5000
    GAMMA: float = 0.99
    LR: float = 1e-3

    EPS_START: float = 1.0
    EPS_END: float = 0.05
    EPS_DECAY_STEPS: int = 5000  # più alto = decay più lento
    TARGET_UPDATE_EVERY: int = 200  # step

    # Artefatti salvati
    MODEL_PATH: str = "dqn_btc.pth"
    SCALER_PATH: str = "scaler.pkl"

    # Live / paper trading
    LIVE_TRADING: bool = False  # lascia False
    PAPER_START_CAPITAL: float = 10000.0
    CYCLE_SECONDS: int = 300  # 5 minuti

    # Telegram (facoltativo)
    TELEGRAM_TOKEN: str = ""   # metti token o lascia vuoto
    TELEGRAM_CHAT_ID: str = "" # metti chat_id o lascia vuoto

CFG = Config()
