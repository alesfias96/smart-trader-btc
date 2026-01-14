# controller.py
import time
import requests
import torch

from config import CFG
from core_rl import Net, get_latest_market_state
from fusion import fusion_modulator
from telegram_utils import invia_segnale_telegram
from logging_utils import registra_operazione


ACTION_ID_TO_STR = {0: "HOLD", 1: "BUY", 2: "SELL"}
ACTION_STR_TO_SIGNAL = {"SELL": -1.0, "HOLD": 0.0, "BUY": 1.0}


class DQNWrapper:
    def __init__(self, model: Net):
        self.model = model
        self.model.eval()

    def predict_action(self, state_seq):
        """
        state_seq: torch.Tensor shape (1, window_size, n_features)
        """
        with torch.no_grad():
            q = self.model(state_seq)
        a = int(torch.argmax(q, dim=1).item())
        return ACTION_ID_TO_STR[a]


class MasterController:
    def __init__(self, ticker: str, dqn: DQNWrapper, sentiment_module):
        self.ticker = ticker
        self.dqn = dqn
        self.sentiment_module = sentiment_module
        self.is_running = False

        # Paper trading
        self.capital = CFG.PAPER_START_CAPITAL
        self.position = 0.0  # quantità BTC simulata

    def fetch_full_state(self):
        # 1) stato tecnico (W*5) + prezzo reale
        tech_flat, last_close = get_latest_market_state(self.ticker, CFG.WINDOW_SIZE, CFG.SCALER_PATH)

        # 2) sentiment e reliability
        sent = float(self.sentiment_module.get_current_sentiment(self.ticker))
        rel = float(self.sentiment_module.get_reliability(self.ticker))

        # 3) ricostruisci finestra (W, 5 tech) e aggiungi sentiment come 6ª colonna
        W = CFG.WINDOW_SIZE
        tech_per_day = 5  # Close, RSI, MACD, BB_Width, Returns

        tech_mat = []
        for i in range(W):
            row5 = tech_flat[i * tech_per_day : (i + 1) * tech_per_day]
            tech_mat.append(row5 + [sent])  # append sentiment

        state_seq = torch.tensor(tech_mat, dtype=torch.float32).unsqueeze(0)  # (1,W,6)
        return state_seq, last_close, sent, rel

    def run_cycle(self):
        state_seq, last_close, sent, rel = self.fetch_full_state()

        # decisione tecnica del DQN
        tech_action = self.dqn.predict_action(state_seq)
        tech_signal = ACTION_STR_TO_SIGNAL[tech_action]

        # fusione con sentiment
        fused = fusion_modulator(tech_signal, sent, rel)

        # azione finale
        if fused > 0.2:
            action = "BUY"
        elif fused < -0.2:
            action = "SELL"
        else:
            action = "HOLD"

        # paper trading semplice (no commissioni)
        if action == "BUY":
            spend = self.capital * 0.10
            if spend > 0:
                qty = spend / last_close
                self.position += qty
                self.capital -= spend

        elif action == "SELL":
            if self.position > 0:
                self.capital += self.position * last_close
                self.position = 0.0

        registra_operazione(action, last_close, self.capital, self.position)

        if action != "HOLD":
            msg = (
                f"🤖 *Segnale IA*\n"
                f"Asset: *{self.ticker}*\n"
                f"Prezzo: `${last_close:,.2f}`\n"
                f"Sentiment: `{sent:+.3f}` (rel {rel:.2f})\n"
                f"Azione: **{action}**\n"
                f"Capitale: `${self.capital:,.2f}` | Posizione: `{self.position:.6f}` BTC"
            )
            invia_segnale_telegram(msg)

        print(f"[LIVE] {action} @ {last_close:.2f} | cap={self.capital:.2f} pos={self.position:.6f} sent={sent:+.3f}")

    def run_safe_cycle(self):
        try:
            self.run_cycle()
        except requests.exceptions.ConnectionError:
            print("⚠️ Connessione giù. Riprovo tra 60s...")
            time.sleep(60)
        except Exception as e:
            invia_segnale_telegram(f"🚨 CRASH BOT: {str(e)}")
            self.is_running = False
            raise
