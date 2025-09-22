import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
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

def calcular_metricas(y_pred):
    ids = df["id_exist"].values
    icm, icm_norm, f1 = do_ICM(y_pred, ids, "OIQ_"+tipo+".json")
    
    return {
        'icm': icm,
        'icm_norm': icm_norm,
        'f1': f1
    }

# Ground truth
ground_truth = df['ground_truth'].values

# Implementación de OIQ Probabilístico
class OIQ_Probabilistico:
    def __init__(self, modelos, tipo):
        self.modelos = modelos
        self.n_modelos = len(modelos)
        
        if tipo =="mio":
            self.umbrales = {'facebook/wav2vec2-large-xlsr-53':0.455, 
                            'google/vivit-b-16x2-kinetics400':0.444,
                            'FacebookAI/xlm-roberta-large_transcripcion':0.737,
                            'FacebookAI/xlm-roberta-large_texto_video':0.485,
                            #'FacebookAI/xlm-roberta-large_original':0.889
            }
        
        if tipo =="original":
            self.umbrales = {'facebook/wav2vec2-large-xlsr-53':0.455, 
                            'google/vivit-b-16x2-kinetics400':0.444,
                            #'FacebookAI/xlm-roberta-large_transcripcion':0.737,
                            #'FacebookAI/xlm-roberta-large_texto_video':0.485,
                            'FacebookAI/xlm-roberta-large_original':0.889
            }
        
        if tipo =="todos":
            self.umbrales = {'facebook/wav2vec2-large-xlsr-53':0.455, 
                            'google/vivit-b-16x2-kinetics400':0.444,
                            'FacebookAI/xlm-roberta-large_transcripcion':0.737,
                            'FacebookAI/xlm-roberta-large_texto_video':0.485,
                            'FacebookAI/xlm-roberta-large_original':0.889
            }
        self.tabla_probabilidades = {}
        self.combinaciones_totales = 2 ** self.n_modelos
            
    def crear_tabla_probabilidades(self, X, y):
        """
        Crea la tabla de probabilidades para cada combinación
        """
        # Aplicar umbrales para obtener predicciones binarias
        predicciones_binarias = {}
        for modelo in self.modelos:
            predicciones_binarias[modelo] = (X[modelo] >= self.umbrales[modelo]).astype(int)
        
        # Inicializar tabla
        self.tabla_probabilidades = {}
        
        # Para cada posible combinación
        for combinacion in product([0, 1], repeat=self.n_modelos):
            # Encontrar muestras que coinciden con esta combinación
            mascara = np.ones(len(X), dtype=bool)
            
            for i, modelo in enumerate(self.modelos):
                mascara &= (predicciones_binarias[modelo] == combinacion[i])
            
            muestras_combinacion = np.sum(mascara)
            
            if muestras_combinacion > 0:
                # Calcular probabilidad de ser clase 1
                positivos = np.sum(y[mascara] == 1)
                probabilidad = positivos / muestras_combinacion
                
                self.tabla_probabilidades[combinacion] = {
                    'probabilidad': probabilidad,
                    'muestras': muestras_combinacion,
                    'positivos': positivos
                }
            else:
                # Si no hay muestras, usar probabilidad por defecto
                self.tabla_probabilidades[combinacion] = {
                    'probabilidad': 0.5,
                    'muestras': 0,
                    'positivos': 0
                }
    
    def predecir_probabilidades(self, X):
        """
        Predice probabilidades usando la tabla OIQ
        """
        # Aplicar umbrales
        predicciones_binarias = np.zeros((len(X), self.n_modelos))
        for i, modelo in enumerate(self.modelos):
            predicciones_binarias[:, i] = (X[modelo] >= self.umbrales[modelo]).astype(int)
        
        # Obtener probabilidades de la tabla
        probabilidades = np.zeros(len(X))
        
        for i in range(len(X)):
            combinacion = tuple(predicciones_binarias[i].astype(int))
            if combinacion in self.tabla_probabilidades:
                probabilidades[i] = self.tabla_probabilidades[combinacion]['probabilidad']
            else:
                # Esto no debería pasar si la tabla está completa
                probabilidades[i] = 0.5
        
        return probabilidades
    
    def entrenar(self, X, y):
        """
        Entrena el modelo OIQ
        """
        print("=== ENTRENANDO OIQ PROBABILÍSTICO ===\n")
        
        # Paso 2: Crear tabla de probabilidades
        print("\nPaso 1: Creando tabla de probabilidades...")
        self.crear_tabla_probabilidades(X, y)
        
        # Estadísticas de la tabla
        print(f"\nTabla creada con {len(self.tabla_probabilidades)} combinaciones")
        combinaciones_con_muestras = sum(1 for v in self.tabla_probabilidades.values() if v['muestras'] > 0)
        print(f"Combinaciones con muestras: {combinaciones_con_muestras}")
        
    def analizar_tabla(self):
        """
        Analiza la tabla de probabilidades
        """
        print("\n=== ANÁLISIS DE LA TABLA OIQ ===\n")
        
        # Ordenar por probabilidad
        combinaciones_ordenadas = sorted(self.tabla_probabilidades.items(), 
                                       key=lambda x: x[1]['probabilidad'], 
                                       reverse=True)
        
        # Mostrar top 10 combinaciones más probables
        print("Top 10 combinaciones con mayor probabilidad de ser clase 1:")
        for i, (comb, info) in enumerate(combinaciones_ordenadas[:10]):
            if info['muestras'] > 0:
                print(f"\nCombinación {comb}:")
                print(f"  Probabilidad: {info['probabilidad']:.3f}")
                print(f"  Muestras: {info['muestras']}")
                print(f"  Positivos: {info['positivos']}")
        
        # Mostrar bottom 10
        print("\n\nBottom 10 combinaciones con menor probabilidad de ser clase 1:")
        for i, (comb, info) in enumerate(combinaciones_ordenadas[-10:]):
            if info['muestras'] > 0:
                print(f"\nCombinación {comb}:")
                print(f"  Probabilidad: {info['probabilidad']:.3f}")
                print(f"  Muestras: {info['muestras']}")
                print(f"  Positivos: {info['positivos']}")
        
        return combinaciones_ordenadas

# Crear y entrenar el modelo OIQ
oiq = OIQ_Probabilistico(modelos,tipo=tipo)
oiq.entrenar(df, ground_truth)

# Predecir probabilidades
probabilidades_oiq = oiq.predecir_probabilidades(df)

# Encontrar mejor umbral de probabilidad
def encontrar_mejor_umbral_probabilidad(probabilidades, y_true):
    """
    Encuentra el mejor umbral sobre las probabilidades OIQ
    """
    mejores_metricas = {}
    umbrales_prob = np.linspace(0, 1, 1000)
    
    for metrica_objetivo in ['icm', 'icm_norm', 'f1']:
        mejor_valor = -np.inf
        mejor_umbral = 0.5
        mejor_todas_metricas = None
        
        for umbral in umbrales_prob:
            predicciones = (probabilidades >= umbral).astype(int)
            
            metricas = calcular_metricas(predicciones)
            
            if metricas[metrica_objetivo] > mejor_valor:
                mejor_valor = metricas[metrica_objetivo]
                mejor_umbral = umbral
                mejor_todas_metricas = metricas
        
        mejores_metricas[metrica_objetivo] = {
            'umbral': mejor_umbral,
            'valor': mejor_valor,
            'metricas': mejor_todas_metricas
        }
    
    return mejores_metricas

# Encontrar mejores umbrales
mejores_umbrales_prob = encontrar_mejor_umbral_probabilidad(probabilidades_oiq, ground_truth)

# Mostrar resultados
print("\n=== RESULTADOS OIQ PROBABILÍSTICO ===\n")

for metrica_objetivo in ['icm', 'icm_norm', 'f1']:
    resultado = mejores_umbrales_prob[metrica_objetivo]
    print(f"\nOptimizando para {metrica_objetivo.upper()}:")
    print(f"  Umbral de probabilidad: {resultado['umbral']:.3f}")
    print(f"  Valor de {metrica_objetivo}: {resultado['valor']:.4f}")
    print("  Todas las métricas:")
    for metrica, valor in resultado['metricas'].items():
        print(f"    {metrica}: {valor:.4f}")

# Usar el mejor resultado de ICM
mejor_resultado = mejores_umbrales_prob['icm']
umbral_prob_optimo = mejor_resultado['umbral']
predicciones_finales = (probabilidades_oiq >= umbral_prob_optimo).astype(int)

print(f"\n=== RESULTADO FINAL (optimizando ICM) ===")
print(f"Umbral de probabilidad: {umbral_prob_optimo:.3f}")
for metrica, valor in mejor_resultado['metricas'].items():
    print(f"{metrica}: {valor:.4f}")

# Análisis de las combinaciones
combinaciones_ordenadas = oiq.analizar_tabla()

# Visualización de combinaciones más frecuentes
print("\n=== COMBINACIONES MÁS FRECUENTES ===")
combinaciones_frecuentes = sorted(oiq.tabla_probabilidades.items(), 
                                 key=lambda x: x[1]['muestras'], 
                                 reverse=True)[:15]

combinaciones = []
muestras = []
probabilidades = []

for comb, info in combinaciones_frecuentes:
    if info['muestras'] > 0:
        combinaciones.append(str(comb))
        muestras.append(info['muestras'])
        probabilidades.append(info['probabilidad'])

plt.figure(figsize=(12, 6))
bars = plt.bar(range(len(combinaciones)), muestras, color='lightblue')

# Color por probabilidad
for i, (bar, prob) in enumerate(zip(bars, probabilidades)):
    color_intensity = prob
    bar.set_color(plt.cm.RdYlGn(color_intensity))

plt.xlabel('Combinación')
plt.ylabel('Número de muestras')
plt.title('Combinaciones más frecuentes (color = probabilidad de clase 1)')
plt.xticks(range(len(combinaciones)), combinaciones, rotation=45, ha='right')
plt.colorbar(plt.cm.ScalarMappable(cmap='RdYlGn'), ax=plt.gca(), 
             label='Probabilidad de clase 1')
plt.tight_layout()
plt.savefig("combinaciones_mas_frecuentes_"+tipo+".png")
