import cv2
import easyocr
import pytesseract
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import re
import json
from spellchecker import SpellChecker
import pandas as pd

def extract_text_from_video(video_path, output_txt=None, frame_interval=30, lang='en'):

    if output_txt is None:
        filename = os.path.splitext(os.path.basename(video_path))[0]
        output_txt = os.path.join(os.path.dirname(video_path), f"{filename}_text.txt")
    
    # Abrir el video
    video = cv2.VideoCapture(video_path)
    
    if not video.isOpened():
        print(f"Error al abrir el video: {video_path}")
        return []
    
    # Obtener información del video
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    
    print(f"Procesando video: {video_path}")
    print(f"Duración: {duration:.2f} segundos, {frame_count} frames, {fps} FPS")
    print(f"Idioma: {lang}")
    
    all_text = []
    frame_num = 0
    processed_frames = 0
    
    while True:
        ret, frame = video.read()
        if not ret:
            break
        
        if frame_num % frame_interval == 0:
            processed_frames += 1
            
            # Detectamos regiones de texto y preprocesamos
            processed_frame = preprocess_frame_for_ocr(frame)
               
            # Realizar OCR
            text = extract_text_with_easyocr(np.array(processed_frame), languages=[lang])
            text = clean_text(text, lang)
            
            if text.strip():
                timestamp = frame_num / fps
                formatted_timestamp = format_timestamp(timestamp)
                text_entry = f"[{formatted_timestamp}] {text}"
                all_text.append(text_entry)
        
        frame_num += 1
    
    video.release()
    
    # Guardar todo el texto en un archivo
    with open(output_txt, 'w', encoding='utf-8') as f:
        f.write('\n\n'.join(all_text))
    
    print(f"Texto extraído guardado en: {output_txt}")
    return all_text

def extract_text_with_easyocr(frame, languages=['en']):
    reader = easyocr.Reader(languages, gpu=True, 
                      detect_network='craft',
                      recognizer='standard')
    results = reader.readtext(frame)
    return ' '.join([text for _, text, conf in results if conf > 0])

def preprocess_frame_for_ocr(frame):
    
    # Convertir a HSV para mejor segmentación de colores
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Crear una máscara para texto blanco y amarillo (comunes en subtítulos)
    # Ajusta estos rangos según sea necesario
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 30, 255])
    mask_white = cv2.inRange(hsv, lower_white, upper_white)
    
    lower_yellow = np.array([20, 100, 200])
    upper_yellow = np.array([30, 255, 255])
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    # Combinar máscaras
    mask = cv2.bitwise_or(mask_white, mask_yellow)
    
    # Aplicar dilatación para conectar componentes de texto
    kernel = np.ones((2, 2), np.uint8)
    dilated_mask = cv2.dilate(mask, kernel, iterations=1)
    
    # Aplicar la máscara a la imagen original en escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    masked_img = cv2.bitwise_and(gray, gray, mask=dilated_mask)
    
    # Mejorar contraste
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(masked_img)
    
    # Umbralización
    _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return thresh

def detect_text_regions(frame):
    # Primero preprocesar el frame con la otra función
    preprocessed_frame = preprocess_frame_for_ocr(frame)
    
    # Usar el frame preprocesado en lugar del original
    gray = preprocessed_frame  # Ya está en escala de grises y procesado
    
    # Detector MSER
    mser = cv2.MSER_create()
    regions, _ = mser.detectRegions(gray)
    
    # Crear máscara para las regiones de texto
    mask = np.zeros(gray.shape, dtype=np.uint8)
    for region in regions:
        x, y, w, h = cv2.boundingRect(region.reshape(-1, 1, 2))
        # Filtrar por tamaño para eliminar ruido
        if w*h > 100 and w/h < 10 and h/w < 10:
            cv2.rectangle(mask, (x, y), (x+w, y+h), (255), -1)
    
    # Aplicar la máscara
    text_only = cv2.bitwise_and(gray, gray, mask=mask)
    
    return text_only

def clean_text(text, lang):
    
    # Pasamos a minusculas
    text = text.lower()
    
    # Eliminar caracteres no deseados
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    
    # Comprobamos si la palabra pertenece al ingles/español
    text = filter_no_valid_words(text, lang)
    
    # Eliminar líneas vacías
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    
    return '\n'.join(lines)

def format_timestamp(seconds):

    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f"{minutes:02d}:{seconds:02d}"

def extract_text_from_videos_in_directory(directory, output_dir=None, metadata=None, frame_interval=30):
    
        
    if output_dir is None:
        output_dir = directory

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    results = {}
    
    # Procesar cada video en el directorio
    for filename in os.listdir(directory):
           
        # Obtenemos el idioma del audio (para despues ajustar el spellChecker)
        lang = metadata[metadata["video"] == filename]["lang"].values[0]
        
        if filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv')):
            video_path = os.path.join(directory, filename)
            output_txt = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_easyOCR.txt")
            
            print(f"\nProcesando video: {filename}")
            texts = extract_text_from_video(video_path, output_txt, frame_interval, lang)
            
            results[video_path] = texts
    
    return results

def filter_no_valid_words(texto, lang):
    global spell_en
    global spell_es
    
    if lang == "en":
        comprobador = spell_en
    
    if lang == "es":
        comprobador = spell_es
    
    palabras = texto.split()  
    palabras_filtradas = []
        
    for palabra in palabras:
        if len(comprobador.known([palabra])) > 0 :
            palabras_filtradas.append(palabra)
        
    return " ".join(palabras_filtradas)

# Cargamos el JSON
with open('EXIST2025_training.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
    
# Lo convertimos a DataFrame por comodidad
records = []
for key, value in data.items():
    value["original_key"] = key
    records.append(value)

df = pd.DataFrame(records)
    
# Definimos los correctores ortográficos
global spell_en
global spell_es 

spell_en = SpellChecker()
spell_es = SpellChecker(language='es')
  
# O para procesar todos los videos en un directorio
directory_input = "videos_prueba"
directory_output = "textos_extraidos_prueba"
extract_text_from_videos_in_directory(directory_input, output_dir=directory_output, metadata=df)