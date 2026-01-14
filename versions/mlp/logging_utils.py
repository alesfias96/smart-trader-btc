# logging_utils.py
import json
import datetime

def registra_operazione(azione: str, prezzo: float, capitale_virtuale: float, posizione: float) -> None:
    log = {
        "timestamp": str(datetime.datetime.now()),
        "azione": azione,
        "prezzo": prezzo,
        "capitale_residuo": capitale_virtuale,
        "posizione": posizione
    }
    with open("paper_trading_log.json", "a", encoding="utf-8") as f:
        f.write(json.dumps(log) + "\n")
