import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 1. Scarichiamo i dati storici (ultimi 2 anni)
ticker = "BTC-USD"
data = yf.download(ticker, start="2023-01-01", end="2025-12-01")

# 2. Pulizia dei dati: teniamo solo il prezzo di chiusura 'Close'
df = data[['Close']].copy()

# 3. Feature Engineering: Creiamo degli indicatori per l'IA
# Media mobile a 7 giorni e a 30 giorni
df['SMA_7'] = df['Close'].rolling(window=7).mean()
df['SMA_30'] = df['Close'].rolling(window=30).mean()

# Calcoliamo il rendimento logaritmico (più facile da processare dalla rete)
df['Returns'] = np.log(df['Close'] / df['Close'].shift(1))

# Rimuoviamo i valori nulli creati dalle medie mobili
df.dropna(inplace=True)

print(df.head())

# Da attaccare al file precedente [Progetto_2.py] se si vuole farlo funzionare per il progetto.

plt.figure(figsize=(12,6))
plt.plot(df['Close'], label='Prezzo di Chiusura', alpha=0.5)
plt.plot(df['SMA_7'], label='Trend Breve (7gg)', color='red')
plt.plot(df['SMA_30'], label='Trend Lungo (30gg)', color='green')
plt.title(f"Dataset di Addestramento per l'Agente IA - {ticker}")
plt.legend()
plt.show()

