# sentiment_module.py
"""
Sentiment module (live).

Obiettivo:
- Esporre un'interfaccia semplice usata dal controller:
  - get_current_sentiment(ticker) -> float
  - get_reliability(ticker) -> float

Include protezione base "sentiment poisoning" (Algoritmo 53):
- Se il sentiment corrente cambia troppo rispetto alla media recente, abbassa la reliability.
"""

from collections import deque
from typing import Deque, Tuple

# Algoritmo 53
def check_news_anomaly(new_sentiment: float, historical_avg_sentiment: float) -> bool:
    threshold = 3.0
    if abs(new_sentiment - historical_avg_sentiment) > threshold:
        print("ATTENZIONE: Possibile attacco di Sentiment Poisoning rilevato!")
        return False
    return True


class SentimentModule:
    def __init__(self, history_len: int = 50):
        self.history: Deque[float] = deque(maxlen=history_len)

        # prova a usare il modulo reale, se disponibile
        self._real = None
        try:
            from real_sentiment import RealSentimentModule  # type: ignore
            self._real = RealSentimentModule()
        except Exception:
            self._real = None

        # cache ultimo valore
        self._last_sent = 0.0
        self._last_rel = 0.2

    def _fetch(self, ticker: str) -> Tuple[float, float]:
        if self._real is None:
            # fallback: nessuna news -> neutro
            return 0.0, 0.2

        sent, rel = self._real.get_sentiment_and_reliability(ticker)
        return float(sent), float(rel)

    def get_current_sentiment(self, ticker: str) -> float:
        sent, rel = self._fetch(ticker)

        # controllo anomalia vs media storica
        if len(self.history) >= 5:
            avg = sum(self.history) / len(self.history)
            ok = check_news_anomaly(sent, avg)
            if not ok:
                # se sospetto: "congela" sentiment verso neutro
                rel = min(rel, 0.1)
                sent = 0.0

        self.history.append(sent)
        self._last_sent = sent
        self._last_rel = rel
        return sent

    def get_reliability(self, ticker: str) -> float:
        # Assicura che siano aggiornati anche quando viene chiamato dopo sentiment
        return float(self._last_rel)
