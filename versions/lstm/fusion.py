# fusion.py
def fusion_modulator(tech_signal: float, sentiment_score: float, reliability_factor: float) -> float:
    if reliability_factor > 0.8:
        return (tech_signal * 0.6) + (sentiment_score * 0.4)
    return tech_signal