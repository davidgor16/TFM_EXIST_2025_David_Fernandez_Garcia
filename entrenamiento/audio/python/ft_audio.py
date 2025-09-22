# Fine-tuning de Wav2Vec para clasificación binaria con énfasis en precisión

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, confusion_matrix, fbeta_score
from sklearn.model_selection import train_test_split
from transformers import Wav2Vec2Model, Wav2Vec2Processor, Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor
from torch.utils.data import Dataset, DataLoader
import json
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
from sklearn.model_selection import KFold

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../evaluacion/python')))
from do_ICM import do_ICM

# Configuración general
SEED = 42
BATCH_SIZE = 4
LEARNING_RATE = 1e-4
NUM_EPOCHS = 1
BETA = 0.5
AUDIO_MAX_LENGTH = 16000 * 61  # 136 segundos a 16kHz
MODEL_NAME = "facebook/wav2vec2-large-xlsr-53"  # Puedes cambiarlo por otro modelo pre-entrenado
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Fijar semillas para reproducibilidad
torch.manual_seed(SEED)
np.random.seed(SEED)

# Dataset personalizado
class AudioClassificationDataset(Dataset):
    def __init__(self, audio_paths, labels, ids, processor):
        self.audio_paths = audio_paths
        self.labels = labels
        self.ids = ids
        self.processor = processor
        
    def __len__(self):
        return len(self.audio_paths)
    
    def __getitem__(self, idx):
        
        audio_path = self.audio_paths[idx]
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Convertir a mono si es necesario
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Remuestrear a 16kHz si es necesario
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)
            sample_rate = 16000
        
        # Asegurar que tenemos un tensor de forma [1, time]
        waveform = waveform.squeeze(0)
                
        # Procesar para Wav2Vec
        inputs = self.processor(waveform, sampling_rate=sample_rate, return_tensors="pt", truncation=True, padding="max_length", do_normalize=True, max_length=AUDIO_MAX_LENGTH, return_attention_mask=True)
        input_values = inputs.input_values.squeeze(0)
        attention_mask = inputs.attention_mask.squeeze(0)
        
        # Liberar memoria explícitamente
        del waveform
        torch.cuda.empty_cache() 
        
        return {
            "input_values": input_values,
            "attention_mask": attention_mask,
            "label": torch.tensor(self.labels[idx], dtype=torch.float),
            "id": self.ids[idx]
        }

# Modelo personalizado con cabeza de clasificación
class Wav2VecForBinaryClassification(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        
        self.wav2vec = Wav2Vec2Model.from_pretrained(model_name,local_files_only=False)
        #self.classifier = nn.Sequential(nn.Linear(self.wav2vec.config.hidden_size, 1))
        
        # Replace the classifier with a more regularized version
        hidden_size = self.wav2vec.config.hidden_size // 2  # Reduce complexity
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(self.wav2vec.config.hidden_size, hidden_size),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_size, 1),
            nn.Sigmoid())
        
        self.loss = nn.BCELoss()
        
    def forward(self, input_values, attention_mask=None):
        
        input_values = torch.clamp(input_values, min=-1.0, max=1.0)
        outputs = self.wav2vec(input_values, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
                
        # Promediar todas las representaciones
        pooled_output = torch.mean(hidden_states, dim=1)
        
        return self.classifier(pooled_output)
    
    def do_loss(self, output, target):
        return self.loss(output,target)

# Función para cargar datos
def load_data(json_path):
    # Cargamos el JSON
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    # Lo convertimos a DataFrame por comodidad
    records = []
    for key, value in data.items():
        value["original_key"] = key
        records.append(value)

    df = pd.DataFrame(records)
    
    trainning_type = sys.argv[1]
    
    if trainning_type == "HARD":
        df =df.rename(columns={"hard_value": "label"})
    
    if trainning_type == "SOFT":
        df =df.rename(columns={"soft_value": "label"})
        
    # Ajustamos los nombres de los ficheros
    df["path_audio"] = df["path_audio"].str.replace('audio/', '../hghub/ownDatasets/TFM_EXIST/audio/train/audios/')
    
    # Codificamos las labels
    df['label'] = df['label'].map({'YES': 1, 'NO': 0})
            
    return df['path_audio'].values, df['label'].values, df["original_key"].values

def load_data_test(json_path):
    # Cargamos el JSON
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    # Lo convertimos a DataFrame por comodidad
    records = []
    for key, value in data.items():
        value["id_EXIST"] = key
        records.append(value)

    df = pd.DataFrame(records)
            
    # Ajustamos los nombres de los ficheros
    df["path_audio"] = df["path_audio"].str.replace('audio/', '../hghub/ownDatasets/TFM_EXIST/audio/test/audios/')
            
    return df['path_audio'].values, df["id_EXIST"].values

# Función para encontrar umbral óptimo
def find_optimal_threshold(model, val_loader, device):
    model.eval()
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for batch in val_loader:
            input_values = batch["input_values"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].cpu().numpy()
            
            outputs = model(input_values, attention_mask)
            probs = outputs.squeeze().cpu().numpy()
            
            if probs.ndim == 0:  
                all_probs.append(probs.item())  
            else:
                all_probs.extend(probs)
            
            all_labels.extend(labels)
    
    # Probar diferentes umbrales
    thresholds = np.arange(0, 0.99, 0.01)
    best_fbeta = -100
    best_precision = -100
    best_threshold = 0
    best_recall = -100
    best_preds = None
    
    for threshold in thresholds:
        preds = [1 if p >= threshold else 0 for p in all_probs]

        precision = precision_score(all_labels, preds, pos_label=1, zero_division=1)
        recall = recall_score(all_labels, preds, pos_label=1, zero_division=1)
        fbeta = fbeta_score(all_labels, preds, beta=BETA,pos_label=1, zero_division=1)
        
        if fbeta > best_fbeta:
            best_fbeta = fbeta
            best_precision = precision
            best_threshold = threshold
            best_recall = recall
            best_preds = preds
    
    print(f"Umbral óptimo: {best_threshold:.3f}, F-beta: {best_fbeta:.3f}, Precisión: {best_precision:.3f}, Recall: {best_recall:.3f}")
    sys.stdout.flush()
    return best_threshold, best_fbeta, best_precision, best_recall, best_preds, all_labels, all_probs

# Función de entrenamiento
from tqdm import tqdm
import torch
from sklearn.metrics import precision_score, recall_score, fbeta_score, confusion_matrix
import sys

def train(model, train_loader, val_loader, optimizer, num_epochs, device, scheduler=None, trial=None, accumulation_steps=4):
    
    for epoch in range(num_epochs):
        # Entrenamiento
        model.train()
        train_loss = 0
        batch_count = 0
        
        # Fuera del bucle para gradientes acumulados
        optimizer.zero_grad()
        
        # Barra de progreso para entrenamiento
        train_pbar = tqdm(train_loader, 
                         desc=f"Epoch {epoch+1}/{num_epochs} [Train]", 
                         leave=False,
                         dynamic_ncols=True,
                         disable=False)
        
        for batch_idx, batch in enumerate(train_pbar):
            input_values = batch["input_values"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            
            outputs_train = model(input_values, attention_mask)
            loss = model.do_loss(outputs_train.squeeze(1), labels)
            
            # Normalizar la pérdida según los pasos de acumulación
            loss = loss / accumulation_steps
            loss.backward()

            # Deshacemos la normalización para el seguimiento
            train_loss += loss.item() * accumulation_steps
            batch_count += 1
            
            # Actualizar parámetros solo cada accumulation_steps
            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()
                
                # Actualizar barra con métricas
                train_pbar.set_postfix({
                    'loss': f'{loss.item() * accumulation_steps:.4f}',
                    'avg_loss': f'{(train_loss/batch_count):.4f}',
                    'acc_batch': f'{(batch_idx + 1) // accumulation_steps}'
                })
            else:
                # Actualizar barra para pasos intermedios
                train_pbar.set_postfix({
                    'loss': f'{loss.item() * accumulation_steps:.4f}',
                    'avg_loss': f'{(train_loss/batch_count):.4f}',
                    'acc_step': f'{(batch_idx + 1) % accumulation_steps}/{accumulation_steps}'
                })
                 
        train_loss /= batch_count
        
        threshold_train, fbeta_train , precision_train, recall_train,  _, _, _ = find_optimal_threshold(model, train_loader, DEVICE)
        
        # Validación
        model.eval()
        val_loss = 0
                
        # Barra de progreso para validación
        val_pbar = tqdm(val_loader, 
                       desc=f"Epoch {epoch+1}/{num_epochs} [Val]", 
                       leave=False,
                       dynamic_ncols=True,
                       disable=False)
        
        with torch.no_grad():
            for batch in val_pbar:
                input_values = batch["input_values"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)
                
                outputs_val = model(input_values, attention_mask)
                loss = model.do_loss(outputs_val.squeeze(1), labels)
                val_loss += loss.item()
                               
                # Actualizar barra con métricas
                val_pbar.set_postfix({
                    'val_loss': f'{loss.item():.4f}',
                    'avg_val_loss': f'{(val_loss/(val_pbar.n+1)):.4f}'
                })
        
        val_loss /= len(val_loader)
        
        threshold_val, fbeta_val , precision_val, recall_val, _, _, _ = find_optimal_threshold(model, val_loader, DEVICE)

        # Mostrar resultados al final de la época
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"TRAINNING: F-beta: {fbeta_train:.4f} | Precision: {precision_train:.4f} | Recall: {recall_train:.4f}")
        print(f"VALIDATION: F-beta: {fbeta_val:.4f} | Precision: {precision_val:.4f} | Recall: {recall_val:.4f}")
        sys.stdout.flush()
        
        # Actualizar learning rate si hay scheduler
        if scheduler:
            scheduler.step(val_loss)
        
        # Pruning con Optuna
        if trial is not None:
            trial.report(val_loss, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
            
    return model

# Función principal
def main():
    
    json_path = '../hghub/ownDatasets/TFM_EXIST/audio/train/metadata_EXIST2025_audio.json'
    
    # Cargamos los datos
    audio_paths, ground_truth, ids = load_data(json_path)
    
    '''
    audio_paths = audio_paths[:25]
    ground_truth = ground_truth[:25]
    ids = ids[:25]
    '''
    
    stage = sys.argv[2]
    
    if stage == "AJUSTE_HIPERPARAMETROS":
            
        # AJUSTE DE HIPER-PARÁMETROS CON OPTUNA ---------------------------------------------------------------------------
        import optuna
        from sklearn.model_selection import train_test_split
        import pandas as pd
        import numpy as np
        
        '''
        # Definir el ratio de división para train/val (por ejemplo, 80% entrenamiento, 20% validación)
        train_size = 0.75
        
        # Dividir los datos una sola vez
        X_train, X_val, y_train, y_val, ids_train, ids_val = train_test_split(
            audio_paths, ground_truth, ids, train_size=train_size, random_state=SEED
        )
        
        print(f"Datos de entrenamiento: {len(X_train)}, Datos de validación: {len(X_val)}")
        sys.stdout.flush()
        
        # Cargar el procesador de Wav2Vec una vez
        processor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME, local_files_only=False)
        
        # Lista para almacenar todos los resultados
        all_results = []
        
        # Definir la función objetivo para Optuna
        def objective(trial):
            BATCH_SIZE = 4
            
            # Sugerir valores para los hiperparámetros
            lr = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
            bs = trial.suggest_categorical('batch_size', [8, 16, 32])
            #epochs = trial.suggest_categorical('epochs', [1,2,3,4,5])
            epochs = trial.suggest_int('epochs', 1, 5)
            
            print(f"\n{'='*70}")
            print(f"PRUEBA #{trial.number}: LR={lr}, BS={bs}, EPOCHS={epochs}")
            print(f"{'='*70}")
            sys.stdout.flush()
            
            # Crear datasets para entrenamiento y validación (una sola vez por prueba)
            train_dataset = AudioClassificationDataset(X_train, y_train, ids_train, processor)
            val_dataset = AudioClassificationDataset(X_val, y_val, ids_val, processor)
            
            # Crear dataloaders con el batch size actual
            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
            
            # Inicializar modelo
            model = Wav2VecForBinaryClassification(MODEL_NAME).to(DEVICE)
            
            # Congelamos el Extractor de Caracteristicas
            model.wav2vec.freeze_feature_encoder()
            
            # Optimizador con tasas de aprendizaje diferentes
            optimizer = optim.AdamW([
                {'params': model.wav2vec.parameters(), 'lr': lr / 10},
                {'params': model.classifier.parameters(), 'lr': lr}
            ])
            
            # Scheduler
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
            
            # Entrenar modelo con el número de epochs actual
            print(f"Entrenando: LR={lr}, BS={bs}, EPOCHS={epochs}")
            sys.stdout.flush()
            
            # Calculamos los accumulation steps
            accumulation_steps = bs/BATCH_SIZE
            
            # Entrenamiento con early stopping a través de Optuna pruning
            model = train(model, train_loader, val_loader, optimizer, epochs, DEVICE, scheduler, trial=trial, accumulation_steps=accumulation_steps)
            
            # Encontrar umbral óptimo
            threshold, fbeta, precision, recall, _, _, _ = find_optimal_threshold(model, val_loader, DEVICE)
            
            if fbeta < 0.01:  
                raise optuna.exceptions.TrialPruned()
            
            # Guardar todos los resultados para análisis posterior
            trial_result = {
                'trial': trial.number,
                'learning_rate': lr,
                'batch_size': bs,
                'epochs': epochs,
                'precision': precision,
                'recall': recall,
                'f-beta': fbeta,
                'threshold': threshold
            }
            all_results.append(trial_result)
            
            print(f"Resultados: Precision={precision:.4f}, Recall={recall:.4f}, F-beta={fbeta:.4f}")
            sys.stdout.flush()
            
            # Informar valor a Optuna para pruning
            trial.report(fbeta, 0)
            
            # Verificar si debemos terminar esta prueba prematuramente
            if trial.should_prune():
                print("Prueba interrumpida por pruning")
                raise optuna.exceptions.TrialPruned()
            
            return fbeta
        
        # Crear un estudio de Optuna
        study = optuna.create_study(
            direction='maximize',  # Queremos maximizar el F-beta score
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=2)  # Pruning para detener pruebas no prometedoras
        )
        
        # Número de pruebas a realizar (reemplaza el valor según tus necesidades)
        n_trials = 10
        
        print(f"Iniciando optimización de hiperparámetros con Optuna ({n_trials} pruebas)")
        print(f"Espacio de búsqueda: LR=[1e-5, 1e-3], BS=[16, 32, 64], EPOCHS=[1, 2, 3, 4, 5]")
        sys.stdout.flush()
        
        # Ejecutar la optimización
        study.optimize(objective, n_trials=n_trials)
        
        # Obtener el mejor conjunto de hiperparámetros
        best_params = study.best_params
        best_value = study.best_value
        
        # Mostrar el mejor resultado
        print("\n" + "="*70)
        print("MEJORES HIPERPARÁMETROS ENCONTRADOS")
        print("="*70)
        print(f"Mejor combinación:")
        print(f"Learning Rate: {best_params['learning_rate']}")
        print(f"Batch Size: {best_params['batch_size']}")
        print(f"Epochs: {best_params['epochs']}")
        print(f"F-beta Score: {best_value:.4f}")
        
        # Crear un DataFrame con todos los resultados
        results_df = pd.DataFrame(all_results)
        
        # Guardar los resultados en un archivo CSV
        results_df.to_csv('optuna_results.csv', index=False)
        
        # Mostrar las 5 mejores pruebas
        print("\nTop 5 pruebas:")
        top_trials = sorted(study.trials, key=lambda t: t.value if t.value is not None else -float('inf'), reverse=True)[:5]
        for t in top_trials:
            if t.value is not None:
                print(f"Trial #{t.number}: LR={t.params['learning_rate']:.1e}, BS={t.params['batch_size']}, EP={t.params['epochs']}, F-beta={t.value:.4f}")
            
        sys.stdout.flush()
        '''
        #-----------------------------------------------------------------------------------------------------------------------------------------------------
        
        # EVALUACIÓN REAL DEL MODELO ---------------------------------------------------------------------------------------------------------------------------------------------
        
        from sklearn.model_selection import KFold
        print("\nEmpezando el entrenamiento final del modelo...")
        best_params = {"epochs":2, "learning_rate":0.000264668939860837, 'batch_size': 32}
        
        # Abrimos el csv de resultados
        csv_name = "TFM/entrenamiento/audio/results/resultados_train.csv"
        results_csv = pd.read_csv(csv_name)
        
        # Añadimos la columna del nuevo modelo
        results_csv[MODEL_NAME] = -100.0
                
        # Número de folds para cross-validation
        n_splits = 5
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
        
        # Métricas para almacenar resultados de todos los folds
        all_fold_precisions = []
        all_fold_recalls = []
        all_fold_fbetas = []
        all_fold_thresholds = []
        all_fold_f1 = []
        all_fold_icm = []
        all_fold_icm_norm = []
        all_confusion_matrix = []
        
        # Cargar el procesador de Wav2Vec una vez (no cambia entre folds)
        processor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME, local_files_only=False)
        
        # Iterar sobre cada fold
        for fold, (train_idx, val_idx) in enumerate(kf.split(audio_paths)):
           
            print(f"\n{'='*50}")
            print(f"FOLD {fold+1}/{n_splits}")
            print(f"{'='*50}")
            sys.stdout.flush()
             
            # Obtener los índices para este fold
            X_train, X_val = audio_paths[train_idx], audio_paths[val_idx]
            y_train, y_val = ground_truth[train_idx], ground_truth[val_idx]
            ids_train, ids_val = ids[train_idx], ids[val_idx]
            
            print(f"Datos de entrenamiento: {len(X_train)}, Datos de validación: {len(X_val)}")
            sys.stdout.flush()
            
            # Crear datasets para este fold
            train_dataset = AudioClassificationDataset(X_train, y_train, ids_train, processor)
            val_dataset = AudioClassificationDataset(X_val, y_val, ids_val, processor)
            
            # Crear dataloaders
            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
            
            # Inicializar modelo para este fold
            model = Wav2VecForBinaryClassification(MODEL_NAME).to(DEVICE)
            
            # Congelamos el Extractor de Caracteristicas
            model.wav2vec.freeze_feature_encoder()
            
            # Optimizador con tasas de aprendizaje diferentes
            optimizer = optim.AdamW([
                {'params': model.wav2vec.parameters(), 'lr': best_params['learning_rate'] / 10},
                {'params': model.classifier.parameters(), 'lr': best_params['learning_rate']}
            ])
            
            # Scheduler para este fold
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
            
            # Entrenar modelo para este fold
            print(f"Iniciando entrenamiento del fold {fold+1}...")
            sys.stdout.flush()
            
            # Calculamos los accumulation steps
            accumulation_steps = best_params["batch_size"]/BATCH_SIZE
            
            model = train(model, train_loader, val_loader, optimizer, best_params['epochs'], DEVICE, scheduler, accumulation_steps=accumulation_steps)
           
            # Encontrar umbral óptimo para maximizar precisión
            threshold = 0.5
            threshold, fbeta , precision, recall, final_preds, all_labels, probs = find_optimal_threshold(model, val_loader, DEVICE)
            all_fold_thresholds.append(threshold)
            
            model_name = MODEL_NAME.split("/")[1]
            filename = "TFM/evaluacion/evaluation_files/resultados"+model_name+"_"+str(best_params['batch_size'])+"_"+str(best_params['epochs'])+"_HP.json"
            icm, icm_norm, f1 = do_ICM(final_preds, ids_val, filename)
            cm = confusion_matrix(all_labels, final_preds)
            
            # Guardar métricas de este fold
            all_fold_precisions.append(precision)
            all_fold_recalls.append(recall)
            all_fold_fbetas.append(fbeta)
            all_fold_f1.append(f1)
            all_fold_icm.append(icm)
            all_fold_icm_norm.append(icm_norm)
            all_confusion_matrix.append(cm)
            
            print(f"\nResultados del fold {fold+1}:")
            print(f"Umbral : {threshold:.3f}")
            print(f"Precision (Clase YES): {precision:.4f}")
            print(f"Recall (Clase YES): {recall:.4f}")
            print(f"F-beta Score (Clase YES): {fbeta:.4f}")
            print(f"ICM: {icm:.4f}")
            print(f"ICM Norm: {icm_norm:.4f}")
            print(f"F-Measure: {f1:.4f}")
            print("Matriz de confusión:")
            print(cm)
            sys.stdout.flush()
            
            # Guardamos las predicciones finales en el CSV de resultados
            # Primero convertimos los ids a ints
            ids_val_int = list(map(int,ids_val))
            results_csv.loc[results_csv['id_exist'].isin(ids_val_int), MODEL_NAME] = probs
        
        # Mostrar resultados promedio de todos los folds
        avg_precision = np.mean(all_fold_precisions)
        avg_recall = np.mean(all_fold_recalls)
        avg_fbeta = np.mean(all_fold_fbetas)
        avg_threshold = np.mean(all_fold_thresholds)
        avg_f1 = np.mean(all_fold_f1)
        avg_icm = np.mean(all_fold_icm)
        avg_icm_norm = np.mean(all_fold_icm_norm)
        sum_cm = np.sum(all_confusion_matrix, axis=0)
        print(sum_cm)
        
        model_name = MODEL_NAME.split("/")[1]
        filename = "TFM/entrenamiento/audio/confusion-matrix/matriz_confusion_"+model_name+"_"+str(best_params['batch_size'])+"_"+str(best_params['epochs'])+"_HP"
        save_confusion_matrix(sum_cm,filename=filename)
        
        # Guardamos el csv
        results_csv.to_csv(csv_name, index=False)
        
        print("\n" + "="*50)
        print("RESULTADOS DE CROSS-VALIDATION")
        print("="*50)
        print(f"Precisión media (Clase YES): {avg_precision:.4f} (±{np.std(all_fold_precisions):.4f})")
        print(f"Recall medio (Clase YES): {avg_recall:.4f} (±{np.std(all_fold_recalls):.4f})")
        print(f"F-beta Score medio (Clase YES): {avg_fbeta:.4f} (±{np.std(all_fold_fbetas):.4f})")
        print(f"F1 Score medio: {avg_f1:.4f} (±{np.std(all_fold_f1):.4f})")
        print(f"ICM medio: {avg_icm:.4f} (±{np.std(all_fold_icm):.4f})")
        print(f"ICM-Norm medio: {avg_icm_norm:.4f} (±{np.std(all_fold_icm_norm):.4f})")
        print(f"Umbral medio: {avg_threshold:.4f} (±{np.std(all_fold_thresholds):.4f})")
        sys.stdout.flush()
        
                #-----------------------------------------------------------------------------------------------------------------------------------------------------
        
        # TEST DEL MODELO ---------------------------------------------------------------------------------------------------------------------------------------------
        print("\n" + "="*70)
        print("ENTRENAMIENTO DEL MODELO FINAL Y EVALUACIÓN DE TEST")
        print("="*70)
        sys.stdout.flush()

        # Cargar el procesador una vez para todo el proceso
        processor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME, local_files_only=False)

        # Crear dataset con todos los datos de entrenamiento
        full_train_dataset = AudioClassificationDataset(audio_paths, ground_truth, ids, processor)
        full_train_loader = DataLoader(full_train_dataset, batch_size=BATCH_SIZE, shuffle=True)

        # Inicializar modelo final
        final_model = Wav2VecForBinaryClassification(MODEL_NAME).to(DEVICE)

        # Congelamos el Extractor de Características
        final_model.wav2vec.freeze_feature_encoder()

        # Optimizador con tasas de aprendizaje diferentes
        optimizer = optim.AdamW([
            {'params': final_model.wav2vec.parameters(), 'lr': best_params['learning_rate'] / 10},
            {'params': final_model.classifier.parameters(), 'lr': best_params['learning_rate']}
        ])

        # Scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

        # Calculamos los accumulation_steps
        accumulation_steps = best_params["batch_size"]/BATCH_SIZE

        print(f"Iniciando entrenamiento del modelo final con todos los datos...")
        print(f"Hiperparámetros: LR={best_params['learning_rate']}, BS={best_params['batch_size']}, Epochs={best_params['epochs']}")
        sys.stdout.flush()

        # Entrenar modelo final con todos los datos
        final_model = train(final_model, full_train_loader, full_train_loader, optimizer, best_params['epochs'], DEVICE, 
                            scheduler, accumulation_steps=accumulation_steps)
        
        # Inferimos sobre TEST
        json_path = '../hghub/ownDatasets/TFM_EXIST/audio/test/metadata_EXIST2025_audio.json'
    
        # Cargamos los datos
        audio_paths_test, ids_test = load_data_test(json_path)
        
        ground_truth = np.zeros(len(audio_paths_test))
        print(len(ground_truth))
        # Crear dataset con todos los datos de entrenamiento
        test_dataset = AudioClassificationDataset(audio_paths_test, ground_truth, ids_test, processor)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        final_model.eval()
        all_probs = []
        with torch.no_grad():
            for batch in test_loader:
                input_values = batch["input_values"].to("cuda")
                attention_mask = batch["attention_mask"].to("cuda")
                outputs = final_model(input_values, attention_mask)
                
                probs = outputs.squeeze().cpu().numpy()
            
                if probs.ndim == 0:  
                    all_probs.append(probs.item())  
                else:
                    all_probs.extend(probs)
        
        # Abrimos el csv de resultados
        csv_name = "TFM/entrenamiento/audio/results/resultados_test.csv"
        results_csv = pd.read_csv(csv_name)
        
        # Añadimos la columna del nuevo modelo
        results_csv[MODEL_NAME] = -100.0
        
        ids_test_int = list(map(int,ids_test))
        results_csv.loc[results_csv['id_exist'].isin(ids_test_int), MODEL_NAME] = all_probs
                
        results_csv.to_csv(csv_name, index=False)
        
        print("FIN DEL EXPERIMENTO")

    
def save_confusion_matrix(cm,classes=["NO","YES"],title='Matriz de Confusión', filename='confusion_matrix.png'): 
       
    # Crear figura
    plt.figure(figsize=(8, 6))
    sns.set(font_scale=1.2)  # Tamaño de fuente
    
    # Crear heatmap
    ax = sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', 
                    cbar=False, linewidths=0.5, linecolor='gray')
    
    # Personalizar ejes
    ax.set(xlabel='Predicciones', ylabel='Verdaderos',
          xticklabels=classes, yticklabels=classes)
    plt.title(title, pad=20, fontsize=14)
    
    total = cm.sum()
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j+0.5, i+0.5, f"{cm[i,j]}\n({cm[i,j]/total*100:.1f}%)", 
                   ha='center', va='center', color='black')
    
    # Guardar figura
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✅ Matriz de confusión guardada como {filename}")
    plt.close()
        

# Código para inferencia
def predict_audio(audio_path, model_path):
    # Cargar modelo guardado
    checkpoint = torch.load(model_path)
    model_name = checkpoint['config']['model_name']
    precision_weight = checkpoint['config']['precision_weight']
    threshold = checkpoint['threshold']
    
    # Inicializar modelo y cargar pesos
    model = Wav2VecForBinaryClassification(model_name, precision_weight)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    model.eval()
    
    # Cargar procesador
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    
    # Procesar audio
    waveform, sample_rate = torchaudio.load(audio_path)
    
    # Convertir a mono si es necesario
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    # Remuestrear a 16kHz si es necesario
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
        waveform = resampler(waveform)
        sample_rate = 16000
        
    inputs = processor(waveform, sampling_rate=sample_rate, return_tensors="pt", padding="max_length", max_length=AUDIO_MAX_LENGTH)
    input_values = inputs.input_values.to(DEVICE)
    attention_mask = inputs.attention_mask.to(DEVICE)
    
    # Hacer predicción
    with torch.no_grad():
        outputs = model(input_values, attention_mask)
        prob = outputs.squeeze().cpu().numpy()
        prediction = 1 if prob >= threshold else 0
    
    return {"prediction": "SI" if prediction == 1 else "NO", "probability": float(prob), "threshold": threshold}

if __name__ == "__main__":
    main()