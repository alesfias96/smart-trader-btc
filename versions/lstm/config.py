# config.py
from dataclasses import dataclass

@dataclass
class Config:
    # Dataset
    TICKER: str = "BTC-USD"
    START: str = "2023-01-01"
    END: str = "2025-12-01"

    # Stato
    WINDOW_SIZE: int = 5
    # 5 feature tecniche + 1 sentiment = 6
    N_FEATURES: int = 6

    # Modello
    HIDDEN_DIM: int = 64
    N_LAYERS: int = 2

    # Training DQN
    EPISODES: int = 50
    BATCH_SIZE: int = 64
    MEMORY_SIZE: int = 5000
    GAMMA: float = 0.99
    LR: float = 1e-3

    EPS_START: float = 1.0
    EPS_END: float = 0.05
    EPS_DECAY_STEPS: int = 5000
    TARGET_UPDATE_EVERY: int = 200

    # SL/TP (training env)
    SL_PERCENT: float = 0.02
    TP_PERCENT: float = 0.05

    # Artefatti
    MODEL_PATH: str = "dqn_btc.pth"
    SCALER_PATH: str = "scaler.pkl"

    # Live / paper
    LIVE_TRADING: bool = False
    PAPER_START_CAPITAL: float = 10000.0
    CYCLE_SECONDS: int = 300

    # Telegram
    TELEGRAM_TOKEN: str = ""
    TELEGRAM_CHAT_ID: str = ""

CFG = Config()
