import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import rankdata
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../evaluacion/python')))
from do_ICM import do_ICM

tipo = sys.argv[1]

# Leer el archivo CSV
df = pd.read_csv('../../../entrenamiento/audio/results/resultados_train.csv')

if tipo == "mio":
    modelos = [
        'facebook/wav2vec2-large-xlsr-53',
        'google/vivit-b-16x2-kinetics400',
        'FacebookAI/xlm-roberta-large_transcripcion',
        'FacebookAI/xlm-roberta-large_texto_video',
        #'FacebookAI/xlm-roberta-large_original'
    ]
 
    
if tipo == "original":
    modelos = [
        'facebook/wav2vec2-large-xlsr-53',
        'google/vivit-b-16x2-kinetics400',
        #'FacebookAI/xlm-roberta-large_transcripcion',
        #'FacebookAI/xlm-roberta-large_texto_video',
        'FacebookAI/xlm-roberta-large_original'
    ]
  

if tipo == "todos":
    modelos = [
        'facebook/wav2vec2-large-xlsr-53',
        'google/vivit-b-16x2-kinetics400',
        'FacebookAI/xlm-roberta-large_transcripcion',
        'FacebookAI/xlm-roberta-large_texto_video',
        'FacebookAI/xlm-roberta-large_original'
    ]
   

# Ground truth
ground_truth = df['ground_truth'].values

# Función para aplicar Borda clásico
def borda_clasico(df, modelos):
    """
    Aplica el método Borda clásico basado en rankings
    """
    n_muestras = len(df)
    
    # Matriz para almacenar los puntos Borda
    puntos_borda = np.zeros(n_muestras)
    
    # Matriz para almacenar rankings de cada modelo
    rankings_modelos = np.zeros((n_muestras, len(modelos)))
    
    # Para cada modelo, obtener el ranking
    for i, modelo in enumerate(modelos):
        scores = df[modelo].values
        
        # Ranking: mayor score = posición 1
        # El parámetro method='max' asigna el mismo ranking a valores iguales
        rankings = rankdata(-scores, method='max')  # -scores para ordenar descendente
        rankings_modelos[:, i] = rankings
        
        # Convertir ranking a puntos: posición 1 = n_muestras puntos
        puntos = n_muestras - rankings + 1
        puntos_borda += puntos
    
    return puntos_borda, rankings_modelos

def calcular_metricas(y_pred):
    ids = df["id_exist"].values
    icm, icm_norm, f1 = do_ICM(y_pred, ids, "borda_"+tipo+".json")
    
    return {
        'icm': icm,
        'icm_norm': icm_norm,
        'f1': f1
    }

# Función para encontrar el mejor umbral sobre los puntos Borda
def encontrar_mejor_umbral_borda(puntos_borda, ground_truth, n_umbrales=10000):
    """
    Encuentra el mejor umbral sobre los puntos Borda
    """
    umbrales = np.linspace(puntos_borda.min(), puntos_borda.max(), n_umbrales)
    
    mejores_metricas = {
        'icm': {'umbral': None, 'valor': -np.inf, 'metricas': None},
        'icm_norm': {'umbral': None, 'valor': -np.inf, 'metricas': None},
        'f1': {'umbral': None, 'valor': -np.inf, 'metricas': None},
    }
    
    for umbral in umbrales:
        # Aplicar umbral
        predicciones = (puntos_borda >= umbral).astype(int)
        
        # Calcular métricas
        metricas = calcular_metricas(predicciones)
        
        # Actualizar mejores resultados
        for metrica in ['icm', 'icm_norm', 'f1']:
            if metricas[metrica] > mejores_metricas[metrica]['valor']:
                mejores_metricas[metrica]['valor'] = metricas[metrica]
                mejores_metricas[metrica]['umbral'] = umbral
                mejores_metricas[metrica]['metricas'] = metricas
    
    return mejores_metricas

# Aplicar Borda
print("=== MÉTODO BORDA CLÁSICO ===\n")

# Calcular puntos Borda
puntos_borda, rankings_modelos = borda_clasico(df, modelos)

# Estadísticas de los puntos Borda
print("Estadísticas de puntos Borda:")
print(f"Mínimo: {puntos_borda.min():.0f}")
print(f"Máximo: {puntos_borda.max():.0f}")
print(f"Media: {puntos_borda.mean():.0f}")
print(f"Mediana: {np.median(puntos_borda):.0f}")
print(f"Desv. estándar: {puntos_borda.std():.0f}")

# Encontrar mejores umbrales
mejores_umbrales = encontrar_mejor_umbral_borda(puntos_borda, ground_truth)

# Mostrar resultados
print("\n=== MEJORES RESULTADOS POR MÉTRICA ===\n")

for metrica_objetivo in ['icm', 'icm_norm', 'f1']:
    resultado = mejores_umbrales[metrica_objetivo]
    print(f"\nOptimizando para {metrica_objetivo.upper()}:")
    print(f"Umbral óptimo: {resultado['umbral']:.0f} puntos Borda")
    print(f"Valor de {metrica_objetivo}: {resultado['valor']:.4f}")
    print("Todas las métricas:")
    for metrica, valor in resultado['metricas'].items():
        print(f"  {metrica}: {valor:.4f}")

# Usar el mejor resultado de ICM
mejor_resultado = mejores_umbrales['icm']
umbral_optimo = mejor_resultado['umbral']
predicciones_borda = (puntos_borda >= umbral_optimo).astype(int)

print(f"\n=== RESULTADO FINAL (optimizando ICM) ===")
print(f"Umbral: {umbral_optimo:.0f} puntos Borda")
for metrica, valor in mejor_resultado['metricas'].items():
    print(f"{metrica}: {valor:.4f}")

# Análisis del comportamiento de Borda
print("\n=== ANÁLISIS DEL MÉTODO BORDA ===")

# Comparar con ground truth
df['puntos_borda'] = puntos_borda
df['pred_borda'] = predicciones_borda

# Puntos Borda promedio por clase
print("\nPuntos Borda promedio por clase:")
print(f"Clase 0: {df[df['ground_truth'] == 0]['puntos_borda'].mean():.0f}")
print(f"Clase 1: {df[df['ground_truth'] == 1]['puntos_borda'].mean():.0f}")

# Análisis de rankings por modelo
print("\nRanking promedio de positivos verdaderos por modelo:")
positivos_verdaderos = df['ground_truth'] == 1
for i, modelo in enumerate(modelos):
    ranking_promedio = rankings_modelos[positivos_verdaderos, i].mean()
    print(f"{modelo}: {ranking_promedio:.0f}")

# 4. Matriz de confusión
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(ground_truth, predicciones_borda)
plt.figure(figsize=(8, 6))
ax = plt.gca() 
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',ax=ax)
ax.set_xlabel('Predicción')
ax.set_ylabel('Valor Real')
ax.set_title('Matriz de Confusión')

plt.tight_layout()
plt.savefig("matriz_confusion_borda_"+tipo+".png")

# Comparación de rankings entre modelos
print("\n=== CORRELACIÓN ENTRE RANKINGS DE MODELOS ===")
correlacion_rankings = np.corrcoef(rankings_modelos.T)
plt.figure(figsize=(10, 8))
sns.heatmap(correlacion_rankings, 
            xticklabels=[m.split('/')[-1] for m in modelos],
            yticklabels=[m.split('/')[-1] for m in modelos],
            annot=True, 
            fmt='.3f', 
            cmap='coolwarm',
            center=0)
plt.title('Correlación entre Rankings de Modelos')
plt.tight_layout()
plt.savefig("matriz_correlacion_borda_"+tipo+".png")

