# real_sentiment.py
import time
import statistics
from typing import List, Tuple

import feedparser
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


class RealSentimentModule:
    """
    Estrae headline da RSS crypto e stima:
    - sentiment (media compound VADER)
    - reliability: più news + meno varianza => più affidabile
    """

    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()

        self.feeds = [
            "http://feeds.feedburner.com/CoinDesk",
            "http://cointelegraph.com/rss",
            "http://bitcoinmagazine.com/.rss/full",
        ]

    def _score(self, text: str) -> float:
        return float(self.analyzer.polarity_scores(text)["compound"])

    def _fetch_headlines(self) -> List[str]:
        headlines: List[str] = []
        for url in self.feeds:
            try:
                feed = feedparser.parse(url)
                for entry in feed.entries[:20]:
                    title = getattr(entry, "title", "")
                    if title:
                        headlines.append(str(title))
            except Exception:
                continue
        return headlines

    def get_sentiment_and_reliability(self, ticker: str = "") -> Tuple[float, float]:
        # ticker qui non serve, ma lo teniamo per compatibilità
        headlines = self._fetch_headlines()

        if not headlines:
            return 0.0, 0.2

        scores = [self._score(h) for h in headlines]
        sentiment = float(statistics.mean(scores))

        if len(scores) > 1:
            variance = float(statistics.pvariance(scores))
            reliability = min(1.0, len(scores) / 10.0) * (1.0 / (1.0 + variance))
        else:
            reliability = 0.2

        return sentiment, float(reliability)
