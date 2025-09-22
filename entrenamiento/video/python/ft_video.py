import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, confusion_matrix, fbeta_score
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import json
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
from sklearn.model_selection import KFold
import cv2
from transformers import VivitModel, VivitImageProcessor, VivitConfig
from transformers import ViTImageProcessor, ViTModel
import torchvision.transforms as transforms
from PIL import Image

# Importar la función de evaluación
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../evaluacion/python')))
from do_ICM import do_ICM

# Configuración general
SEED = 42
BATCH_SIZE = 1
LEARNING_RATE = 1e-4
NUM_EPOCHS = 1
BETA = 0.5
MODEL_NAME = "google/vivit-b-16x2"
VIDEO_MAX_FRAMES = 32  # Número máximo de frames a extraer
VIDEO_FRAME_RATE = 5   # Frames por segundo a extraer
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Fijar semillas para reproducibilidad
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


class VideoClassificationDataset(Dataset):
    def __init__(self, video_paths, labels, ids, processor, num_frames=32, frame_size=(224, 224)):
        self.video_paths = video_paths
        self.labels = labels
        self.ids = ids
        self.processor = processor
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.image_mean = self.processor.image_mean
        self.image_std = self.processor.image_std
        
        self.size = self.processor.size['shortest_edge']
        #size = self.processor.size['height']

        self.transform = transforms.Compose([
            transforms.Resize((self.size, self.size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.image_mean, std=self.image_std)
        ])
    
    def __len__(self):
        return len(self.video_paths)
    
    def extract_frames(self, video_path):
        import subprocess
        import tempfile
        import os
        import shutil
        import numpy as np
        from PIL import Image
        
        # Crear directorio temporal
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Extraer exactamente el número de frames que necesitamos
            output_pattern = os.path.join(temp_dir, "frame_%04d.jpg")
            
            # Obtener duración del video
            duration_cmd = f"ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 \"{video_path}\""
            try:
                duration = float(subprocess.check_output(duration_cmd, shell=True, stderr=subprocess.DEVNULL).decode().strip())
            except:
                duration = 10.0  # Valor predeterminado si no se puede detectar
            
            # Calcular tasa para obtener los frames necesarios
            fps = max(1, self.num_frames / duration)
            
            # Extraer frames
            extract_cmd = f"ffmpeg -i \"{video_path}\" -vsync 0 -vf \"fps={fps}\" -q:v 1 {output_pattern} -hide_banner -loglevel error"
            subprocess.call(extract_cmd, shell=True, stderr=subprocess.DEVNULL)
            
            # Cargar los frames
            frames = []
            frame_files = sorted([os.path.join(temp_dir, f) for f in os.listdir(temp_dir) if f.endswith('.jpg')])
            
            if len(frame_files) == 0:
                # Falback: intentar con un método más agresivo
                extract_cmd = f"ffmpeg -i \"{video_path}\" -vsync 0 -vf \"thumbnail=100\" -frames:v {self.num_frames} {output_pattern} -hide_banner -loglevel error"
                subprocess.call(extract_cmd, shell=True, stderr=subprocess.DEVNULL)
                frame_files = sorted([os.path.join(temp_dir, f) for f in os.listdir(temp_dir) if f.endswith('.jpg')])
            
            # Procesar los frames disponibles
            if len(frame_files) > 0:
                # Seleccionar frames equidistantes si es necesario
                indices = np.round(np.linspace(0, len(frame_files) - 1, min(self.num_frames, len(frame_files)))).astype(int)
                for idx in indices:
                    try:
                        img = Image.open(frame_files[idx])
                        frames.append(img)
                    except:
                        frames.append(Image.new('RGB', self.frame_size, (0, 0, 0)))
            
            # Completar con frames repetidos si es necesario
            while len(frames) < self.num_frames:
                if len(frames) > 0:
                    frames.append(frames[len(frames) % len(frames)])
                else:
                    frames.append(Image.new('RGB', self.frame_size, (0, 0, 0)))
            
            return frames[:self.num_frames]  # Asegurar el número exacto de frames
        
        finally:
            # Limpiar archivos temporales
            try:
                shutil.rmtree(temp_dir)
            except:
                pass
            
    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        frames = self.extract_frames(video_path)
        
        # Procesar cada frame con ViT processor
        processed_frames = []
        for frame in frames:
            inputs = self.processor(images=frame, return_tensors="pt")
            processed_frames.append(inputs.pixel_values.squeeze(0))
        
        # Apilar los frames [num_frames, channels, height, width]
        stacked_frames = torch.stack(processed_frames).squeeze(1)
        
        # Liberar memoria explícitamente
        del frames, processed_frames
        torch.cuda.empty_cache() 
        
        return {
            "input_values": stacked_frames,
            "label": torch.tensor(self.labels[idx], dtype=torch.float),
            "id": self.ids[idx]
        }

# Modelo para clasificación de videos
class VideoBinaryClassification(nn.Module):
    def __init__(self, model_name, dropout_vit, dropout_cls, layers):
        super().__init__()
        
        config = VivitConfig.from_pretrained(MODEL_NAME)

        # Update dropout rates (example values)
        config.hidden_dropout_prob = dropout_vit  # Dropout after MLP layers
        config.attention_probs_dropout_prob = dropout_vit  # Dropout for attention weights
        
        self.vivit = VivitModel.from_pretrained(model_name,local_files_only=False, config=config)
        self.classifier = nn.Sequential(nn.Dropout(dropout_cls),nn.Linear(self.vivit.config.hidden_size, 1))
        self.loss = nn.BCEWithLogitsLoss()
        
        for param in list(self.vivit.parameters())[:layers]:  # Mantén descongeladas solo las últimas capas
            param.requires_grad = False
        
        
    def forward(self, pixel_values):
        # ViViT espera directamente tensores con forma [batch_size, num_frames, channels, height, width]
        # No es necesario reformatear como en ViT
        
        # Procesar el video completo con ViViT
        outputs = self.vivit(pixel_values)
        
        # Obtener el token de clasificación (CLS)
        video_features = outputs.pooler_output
        
        # ViViT ya integra la información temporal, por lo que no necesitamos hacer pooling temporal
        
        # Clasificación final
        logits = self.classifier(video_features)
        
        return logits
        
    def do_loss(self, output, target):
        return self.loss(output, target)
    
    
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
    df["path_video"] = df["path_video"].str.replace('videos/', '../hghub/ownDatasets/TFM_EXIST/video/train/videos/')
    
    # Codificamos las labels
    df['label'] = df['label'].map({'YES': 1, 'NO': 0})
            
    return df['path_video'].values, df['label'].values, df["original_key"].values

def load_data_test(json_path):
    # Cargamos el JSON
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    # Lo convertimos a DataFrame por comodidad
    records = []
    for key, value in data.items():
        value["original_key"] = key
        records.append(value)

    df = pd.DataFrame(records)
            
    # Ajustamos los nombres de los ficheros
    df["path_video"] = df["path_video"].str.replace('videos/', '../hghub/ownDatasets/TFM_EXIST/video/test/videos/')
            
    return df['path_video'].values, df["original_key"].values

# Función para encontrar umbral óptimo
def find_optimal_threshold(model, val_loader, device):
    model.eval()
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for batch in val_loader:
            input_values = batch["input_values"].to(device)
            labels = batch["label"].cpu().numpy()
            
            outputs = model(input_values)
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
                         disable=True)
        
        for batch_idx, batch in enumerate(train_pbar):
            input_values = batch["input_values"].to(device)
            labels = batch["label"].to(device)
            
            outputs_train = model(input_values)
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
                       disable=True)
        
        with torch.no_grad():
            for batch in val_pbar:
                input_values = batch["input_values"].to(device)
                labels = batch["label"].to(device)
                
                outputs_val = model(input_values)
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
    
    json_path = '../hghub/ownDatasets/TFM_EXIST/video/train/metadata_EXIST2025_video.json'
    
    # Cargamos los datos
    video_paths, ground_truth, ids = load_data(json_path)
    
    '''
    video_paths = video_paths[:25]
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
        
        # Definir el ratio de división para train/val (por ejemplo, 80% entrenamiento, 20% validación)
        train_size = 0.75
        
        # Dividir los datos una sola vez
        X_train, X_val, y_train, y_val, ids_train, ids_val = train_test_split(
            video_paths, ground_truth, ids, train_size=train_size, random_state=SEED
        )
        
        print(f"Datos de entrenamiento: {len(X_train)}, Datos de validación: {len(X_val)}")
        sys.stdout.flush()
        
        # Cargar el procesador de VIT una vez
        processor = VivitImageProcessor.from_pretrained(MODEL_NAME, local_files_only=False)

        # Lista para almacenar todos los resultados
        all_results = []
        
        # Definir la función objetivo para Optuna
        def objective(trial):
            BATCH_SIZE = 1
            
            '''
            # Sugerir valores para los hiperparámetros
            lr = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
            bs = trial.suggest_categorical('batch_size', [8, 16, 32])
            epochs = trial.suggest_categorical('epochs', [2])
            #dropout_vit = trial.suggest_float('dropout_vit', 0.1, 0.5)
            #dropout_cls = trial.suggest_float('dropout_cls', 0.1, 0.5)
            #epochs = trial.suggest_int('epochs', 1, 5)
            '''
            
            # Sugerir valores para los hiperparámetros
            lr = trial.suggest_float('learning_rate',6.677867092346639e-05,6.677867092346639e-05)
            bs = trial.suggest_categorical('batch_size', [8])
            epochs = trial.suggest_categorical('epochs', [2])
            dropout_vit = trial.suggest_float('dropout_vit', 0,0)
            dropout_cls = trial.suggest_float('dropout_cls', 0,0)
            layers = trial.suggest_categorical("layers", [50])
            
            print(f"\n{'='*70}")
            print(f"PRUEBA #{trial.number}: LR={lr}, BS={bs}, EPOCHS={epochs}, DROPOUT_VIT={dropout_vit}, DROPOUT_CLS={dropout_cls}, LAYERS_FREEZE={layers}")
            print(f"{'='*70}")
            sys.stdout.flush()
            
            # Crear datasets para entrenamiento y validación (una sola vez por prueba)
            train_dataset = VideoClassificationDataset(X_train, y_train, ids_train, processor)
            val_dataset = VideoClassificationDataset(X_val, y_val, ids_val, processor)
            
            # Crear dataloaders con el batch size actual
            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
            
            # Inicializar modelo
            model = VideoBinaryClassification(MODEL_NAME, dropout_vit=dropout_vit, dropout_cls=dropout_cls,layers=layers).to(DEVICE)
                        
            # Optimizador con tasas de aprendizaje diferentes
            optimizer = optim.AdamW([
                {'params': model.vivit.parameters(), 'lr': lr / 10},
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
        
        #-----------------------------------------------------------------------------------------------------------------------------------------------------
        
        # EVALUACIÓN REAL DEL MODELO ---------------------------------------------------------------------------------------------------------------------------------------------
        
        
        
        from sklearn.model_selection import KFold
        print("\nEmpezando el entrenamiento final del modelo...")
        
        # Abrimos el csv de resultados
        csv_name = "TFM/entrenamiento/audio/results/resultados_train.csv"
        results_csv = pd.read_csv(csv_name)
        
        # Añadimos la columna del nuevo modelo
        results_csv[MODEL_NAME] = -100
                
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
        
        processor = VivitImageProcessor.from_pretrained(MODEL_NAME, local_files_only=False)
        
        # Iterar sobre cada fold
        for fold, (train_idx, val_idx) in enumerate(kf.split(video_paths)):
           
            print(f"\n{'='*50}")
            print(f"FOLD {fold+1}/{n_splits}")
            print(f"{'='*50}")
            sys.stdout.flush()
             
            # Obtener los índices para este fold
            X_train, X_val = video_paths[train_idx], video_paths[val_idx]
            y_train, y_val = ground_truth[train_idx], ground_truth[val_idx]
            ids_train, ids_val = ids[train_idx], ids[val_idx]
            
            print(f"Datos de entrenamiento: {len(X_train)}, Datos de validación: {len(X_val)}")
            sys.stdout.flush()
            
            # Crear datasets para este fold
            train_dataset = VideoClassificationDataset(X_train, y_train, ids_train, processor)
            val_dataset = VideoClassificationDataset(X_val, y_val, ids_val, processor)
            
            # Crear dataloaders
            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
            
            # Inicializar modelo para este fold
            model = VideoBinaryClassification(MODEL_NAME).to(DEVICE)
                        
            # Optimizador con tasas de aprendizaje diferentes
            optimizer = optim.AdamW([
                {'params': model.vivit.parameters(), 'lr': best_params['learning_rate'] / 10},
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
        filename = "TFM/entrenamiento/video/confusion-matrix/matriz_confusion_"+model_name+"_"+str(best_params['batch_size'])+"_"+str(best_params['epochs'])+"_HP"
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
        processor = VivitImageProcessor.from_pretrained(MODEL_NAME, local_files_only=False)

        # Crear dataset con todos los datos de entrenamiento
        full_train_dataset = VideoClassificationDataset(video_paths, ground_truth, ids, processor)
        full_train_loader = DataLoader(full_train_dataset, batch_size=BATCH_SIZE, shuffle=True)

        # Inicializar modelo final
        final_model = VideoBinaryClassification(MODEL_NAME).to(DEVICE)

        # Optimizador con tasas de aprendizaje diferentes
        optimizer = optim.AdamW([
            {'params': final_model.vivit.parameters(), 'lr': best_params['learning_rate'] / 10},
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
        json_path = '../hghub/ownDatasets/TFM_EXIST/video/test/metadata_EXIST2025_video.json'
    
        # Cargamos los datos
        audio_paths_test, ids_test = load_data_test(json_path)
        
        ground_truth = np.zeros(len(audio_paths_test))
        print(len(ground_truth))
        # Crear dataset con todos los datos de entrenamiento
        test_dataset = VideoBinaryClassification(audio_paths_test, ground_truth, ids_test, processor)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        final_model.eval()
        all_probs = []
        with torch.no_grad():
            for batch in test_loader:
                input_values = batch["input_values"].to("cuda")
                outputs = final_model(input_values)
                
                probs = outputs.squeeze().cpu().numpy()
            
                if probs.ndim == 0:  
                    all_probs.append(probs.item())  
                else:
                    all_probs.extend(probs)
        
        # Abrimos el csv de resultados
        csv_name = "TFM/entrenamiento/audio/results/resultados_test.csv"
        results_csv = pd.read_csv(csv_name)
        
        # Añadimos la columna del nuevo modelo
        results_csv[MODEL_NAME] = -100
        
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
        
if __name__ == "__main__":
    main()