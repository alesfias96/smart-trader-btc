# live_bot.py
import time
import torch

from config import CFG
from core_rl import Net
from controller import DQNWrapper, MasterController
from real_sentiment import RealSentimentModule


def main():
    input_dim = CFG.WINDOW_SIZE * CFG.N_FEATURES
    model = Net(input_dim=input_dim, n_actions=3)
    model.load_state_dict(torch.load(CFG.MODEL_PATH, map_location="cpu"))
    model.eval()

    dqn = DQNWrapper(model)
    sentiment = RealSentimentModule()

    controller = MasterController(CFG.TICKER, dqn, sentiment)
    controller.is_running = True

    while controller.is_running:
        controller.run_safe_cycle()
        time.sleep(CFG.CYCLE_SECONDS)


if __name__ == "__main__":
    main()
