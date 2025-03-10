import requests
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

def get_historical_data(coin_id, days):
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    params = {
        'vs_currency': 'usd',
        'days': days
    }
    response = requests.get(url, params=params)
    if response.status_code != 200:
        raise Exception(f"Erro na requisição à API: {response.status_code}")
    data = response.json()
    prices = data['prices']
    df = pd.DataFrame(prices, columns=['timestamp', 'price'])
    df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.drop('timestamp', axis=1)
    return df

coin_id = 'bitcoin'
days = 365
try:
    df = get_historical_data(coin_id, days)
    print("Dados coletados com sucesso!")
    print(df.head())  
except Exception as e:
    print(e)
    exit()

df['next_day_price'] = df['price'].shift(-1)
df['target'] = (df['next_day_price'] > df['price']).astype(int)
df = df.dropna()

if len(df) == 0:
    print("Erro: Nenhum dado válido após o pré-processamento.")
    exit()

X = df[['price']]
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f'Acurácia do modelo: {accuracy:.2f}')

importances = model.feature_importances_
plt.bar(X.columns, importances)
plt.title('Importância das Variáveis')
plt.show()