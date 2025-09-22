import pandas as pd
from sklearn.metrics import precision_score, recall_score, confusion_matrix, fbeta_score
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../evaluacion/python')))
from do_ICM import do_ICM

model = "FacebookAI/xlm-roberta-large_original"
threshold = 0.9340

results = pd.read_csv("../../entrenamiento/audio/results/resultados_train.csv")[model].values
ground_truth = pd.read_csv("../../entrenamiento/audio/results/resultados_train.csv")["ground_truth"]
ids = pd.read_csv("../../entrenamiento/audio/results/resultados_train.csv")["id_exist"]

preds = [1 if p >= threshold else 0 for p in results]

precision = precision_score(ground_truth, preds, pos_label=1, zero_division=1)
recall = recall_score(ground_truth, preds, pos_label=1, zero_division=1)
fbeta = fbeta_score(ground_truth, preds, beta=0.5, pos_label=1, zero_division=1)

model_name = model.split("/")[1]
filename = f"../evaluation_files/resultados_{model_name}_huggingface.json"
icm, icm_norm, f1 = do_ICM(preds, ids, filename)

print(f"Precision (Clase YES): {precision:.4f}")
print(f"Recall (Clase YES): {recall:.4f}")
print(f"F-beta Score (Clase YES): {fbeta:.4f}")
print(f"ICM: {icm:.4f}")
print(f"ICM Norm: {icm_norm:.4f}")
print(f"F-Measure: {f1:.4f}")