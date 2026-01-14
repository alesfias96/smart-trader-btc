# sentiment_module.py
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

class SentimentModule:
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()

    def score_text(self, text: str) -> float:
        return self.analyzer.polarity_scores(text)["compound"]

    def get_current_sentiment(self, ticker: str) -> float:
        """
        Versione DEMO: usa frasi “finte”.
        Quando vuoi, la sostituiamo con news vere da una API.
        """
        news = [
            "Bitcoin ETF approved, market reacts positively",
            "Regulatory uncertainty increases, investors cautious",
            "Institutional adoption of BTC grows steadily"
        ]
        scores = [self.score_text(n) for n in news]
        return sum(scores) / max(len(scores), 1)

    def get_reliability(self, ticker: str) -> float:
        """
        DEMO: affidabilità fissa alta.
        In reale: dipende da quante news hai, coerenza, fonti, ecc.
        """
        return 0.9
