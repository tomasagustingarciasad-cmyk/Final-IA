import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Cargar el CSV
df = pd.read_csv("base_datos/features_imagenes.csv")

# Ver las primeras filas para confirmar que se cargó bien
print("Primeras 5 filas del dataset:")
print(df.head())

print("\nInformación del dataset:")
print(df.info())

# Crear la gráfica de hu1, hu2 y REDONDEZ (en lugar de aspect_ratio)
plt.figure(figsize=(12, 8))

# Gráfico 1: hu1 vs hu2 coloreado por clase
plt.subplot(2, 2, 1)
for clase in df['clase'].unique():
    mask = df['clase'] == clase
    plt.scatter(df[mask]['hu1'], df[mask]['hu2'], label=clase, alpha=0.7)
plt.xlabel('hu1')
plt.ylabel('hu2')
plt.title('hu1 vs hu2 por Clase')
plt.legend()
plt.grid(True, alpha=0.3)

# Gráfico 2: hu1 vs REDONDEZ coloreado por clase
plt.subplot(2, 2, 2)
for clase in df['clase'].unique():
    mask = df['clase'] == clase
    plt.scatter(df[mask]['hu1'], df[mask]['redondez'], label=clase, alpha=0.7)
plt.xlabel('hu1')
plt.ylabel('redondez')
plt.title('hu1 vs Redondez por Clase')
plt.legend()
plt.grid(True, alpha=0.3)

# Gráfico 3: hu2 vs REDONDEZ coloreado por clase
plt.subplot(2, 2, 3)
for clase in df['clase'].unique():
    mask = df['clase'] == clase
    plt.scatter(df[mask]['hu2'], df[mask]['redondez'], label=clase, alpha=0.7)
plt.xlabel('hu2')
plt.ylabel('redondez')
plt.title('hu2 vs Redondez por Clase')
plt.legend()
plt.grid(True, alpha=0.3)

# Gráfico 4: Distribución de valores (ahora con REDONDEZ)
plt.subplot(2, 2, 4)
features = ['hu1', 'hu2', 'redondez']  # Cambiado aquí
df[features].boxplot()
plt.title('Distribución de Características')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# Estadísticas descriptivas por clase (ahora con REDONDEZ)
print("\nEstadísticas por clase:")
for clase in df['clase'].unique():
    print(f"\n--- {clase} ---")
    clase_data = df[df['clase'] == clase]
    print(f"hu1: {clase_data['hu1'].mean():.3f} ± {clase_data['hu1'].std():.3f}")
    print(f"hu2: {clase_data['hu2'].mean():.3f} ± {clase_data['hu2'].std():.3f}")
    print(f"redondez: {clase_data['redondez'].mean():.3f} ± {clase_data['redondez'].std():.3f}")  # Cambiado aquí



    from mpl_toolkits.mplot3d import Axes3D

# Gráfica 3D de hu1, hu2 y REDONDEZ
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Colores para cada clase
colors = {'Arandela': 'red', 'Tuerca': 'blue', 'Tornillo': 'green', 'Clavo': 'orange'}

for clase in df['clase'].unique():
    mask = df['clase'] == clase
    ax.scatter(df[mask]['hu1'], 
               df[mask]['hu2'], 
               df[mask]['redondez'],  # Cambiado aquí
               label=clase, 
               alpha=0.7,
               s=60)

ax.set_xlabel('hu1')
ax.set_ylabel('hu2')
ax.set_zlabel('redondez')  # Cambiado aquí
ax.set_title('Gráfica 3D: hu1 vs hu2 vs Redondez')
ax.legend()

plt.show()