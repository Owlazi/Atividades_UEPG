
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1)  
y = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0])  

x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.3, random_state=42)


modelo = LogisticRegression()


modelo.fit(x_treino, y_treino)

y_previsto = modelo.predict(x_teste)

acuracia = accuracy_score(y_teste, y_previsto)
print(f"Acurácia do modelo: {acuracia * 100:.2f}%")

horas_exercicio = 2  
previsao = modelo.predict([[horas_exercicio]])
print(f"Para {horas_exercicio} horas de exercício por semana, o risco de doença cardíaca é: {'Sim' if previsao[0] == 1 else 'Não'}")