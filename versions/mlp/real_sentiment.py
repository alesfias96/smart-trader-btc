import feedparser
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import time
import statistics

class RealSentimentModule:
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()

        # RSS feed affidabili (puoi aggiungere altri)
        self.feeds = [
            "http://feeds.feedburner.com/CoinDesk",
            "http://cointelegraph.com/rss",
            "http://bitcoinmagazine.com/.rss/full"
        ]

    def _fetch_headlines(self, limit_per_feed=5):
        headlines = []

        for url in self.feeds:
            try:
                feed = feedparser.parse(url)
                for entry in feed.entries[:limit_per_feed]:
                    title = entry.get("title", "")
                    if title:
                        headlines.append(title)
            except Exception:
                continue

        return headlines

    def _score(self, text: str) -> float:
        return self.analyzer.polarity_scores(text)['compound']

    def get_current_sentiment(self, ticker: str):
        headlines = self._fetch_headlines()

        # Se non arrivano news -> sentiment neutro e poco affidabile
        if len(headlines) == 0:
            return 0.0, 0.0

        scores = [self._score(h) for h in headlines]

        sentiment = statistics.mean(scores)

        # Reliability: più news + meno dispersione = più affidabile
        if len(scores) > 1:
            variance = statistics.pvariance(scores)
            reliability = min(1.0, len(scores) / 10) * (1 / (1 + variance))
        else:
            reliability = 0.2

        return sentiment, reliability


