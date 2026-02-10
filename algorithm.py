import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris

# Carga de Datos
iris = load_iris()
df = pd.DataFrame(iris.data, columns = iris.feature_names)
df['target'] = iris.target

# Analisis y Representacion de los atributos
sns.pairplot(df, hue = 'target', palette = 'viridis')
plt.savefig('grafica_iris_color.png')
print("Grafica guardada como 'grafica_iris_color.png'")

X = df.drop('target', axis = 1)

# Utilizacion de la organiacion K-Means
model = KMeans(n_clusters = 3, random_state = 42)
df['cluster'] = model.fit_predict(df)

print("Algoritmo aplicado de forma correcta (creo)")