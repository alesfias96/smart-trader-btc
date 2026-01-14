# telegram_utils.py
import requests
from config import CFG

def invia_segnale_telegram(messaggio: str) -> None:
    if not CFG.TELEGRAM_TOKEN or not CFG.TELEGRAM_CHAT_ID:
        # Telegram non configurato: non fare nulla
        return

    url = f"https://api.telegram.org/bot{CFG.TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": CFG.TELEGRAM_CHAT_ID,
        "text": messaggio,
        "parse_mode": "Markdown",
    }
    requests.post(url, data=payload, timeout=15)
