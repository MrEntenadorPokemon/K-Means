import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris

# Carga de Datos
iris = load_iris()
df = pd.DataFrame(iris.data, columns = iris.feature_names)
df['target'] = iris.target # Antes de la aplicacion de K-Means

# Aplicacion del metodo K-Means
models = KMeans(n_clusters = 3, random_state = 42)
df['cluster'] = models.fit_predict(df.drop('target', axis = 1))

# Creacion de graficas antes y despues
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Grafica antes de K-Means
sns.scatterplot(
    data = df,
    x = 'sepal length (cm)',
    y = 'sepal width (cm)',
    hue = 'target',
    palette = 'viridis',
    ax = axes[0]
)
axes[0].set_title('Antes de K-Means')

# Grafica despues de K-Means
sns.scatterplot(
    data = df,
    x = 'sepal length (cm)',
    y = 'sepal width (cm)',
    hue = 'cluster',
    palette = 'viridis',
    s = 100,
    ax = axes[1]
)
axes[1].set_title('Despues de K-Means')

# Centralizacion de los Clusters
centers = models.cluster_centers_
axes[1].scatter(centers[:, 0], centers[:, 1], c='red', s=200, marker='X', label='Centroides')
axes[1].set_title('Despues de K-Means con Centroides')

# Creacion de la grafica comparativa en imagen png
plt.tight_layout()
plt.savefig('grafica_iris_comparativa.png')
print("Grafica comparativa guardada como 'grafica_iris_comparativa.png'")