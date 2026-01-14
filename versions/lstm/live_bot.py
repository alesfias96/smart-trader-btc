# live_bot.py
import time
import torch

from config import CFG
from core_rl import Net
from controller import DQNWrapper, MasterController
from sentiment_module import SentimentModule


def main():
    model = Net(n_features=CFG.N_FEATURES, hidden_dim=CFG.HIDDEN_DIM, n_actions=3, n_layers=CFG.N_LAYERS)
    model.load_state_dict(torch.load(CFG.MODEL_PATH, map_location="cpu"))
    model.eval()

    dqn = DQNWrapper(model)
    sentiment = SentimentModule()

    controller = MasterController(CFG.TICKER, dqn, sentiment)
    controller.is_running = True

    while controller.is_running:
        controller.run_safe_cycle()
        time.sleep(CFG.CYCLE_SECONDS)


if __name__ == "__main__":
    main()
