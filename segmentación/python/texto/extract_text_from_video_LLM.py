import cv2
import torch
from transformers import AutoModelForCausalLM
from deepseek_vl.models import VLChatProcessor, MultiModalityCausalLM
from deepseek_vl.utils.io import load_pil_images
from PIL import Image
import re
import json
import os
import time
import pandas as pd
from datetime import timedelta
from spellchecker import SpellChecker
import shutil
import subprocess
import tempfile


def extract_frames(video_path, output_dir, frame_rate=1):
    
    # Crear directorio si no existe
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    frame_paths = []
    
    try:
        # Primer intento: OpenCV directo
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            # Obtener FPS del video y calcular intervalo
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0 or fps > 1000:  # Detectar valores inválidos
                print(f"Advertencia: FPS inválido ({fps}), usando valor predeterminado de 30")
                fps = 30
                
            frame_interval = max(1, int(fps / frame_rate))
            
            # Extraer frames
            frame_count = 0
            frame_saved = 0
            
            while True:
                try:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    if frame_count % frame_interval == 0:
                        # Guardar frame como imagen
                        frame_path = os.path.join(output_dir, f"frame_{frame_saved:04d}.jpg")
                        cv2.imwrite(frame_path, frame)
                        frame_paths.append(frame_path)
                        frame_saved += 1
                    
                    frame_count += 1
                except Exception as e:
                    print(f"Error al procesar frame {frame_count}: {str(e)}")
                    frame_count += 1
                    continue
            
            cap.release()
            
            if len(frame_paths) > 0:
                print(f"Se extrajeron {len(frame_paths)} frames del video.")
                return frame_paths
    
    except Exception as e:
        print(f"Falló la extracción con OpenCV: {str(e)}")
    
    # Si OpenCV falla o no extrae frames, intentar con FFmpeg
    print("Intentando extracción alternativa con FFmpeg...")
    
    try:
        # Comprobar que FFmpeg está instalado
        try:
            subprocess.run(["ffmpeg", "-version"], capture_output=True, check=False)
        except FileNotFoundError:
            print("FFmpeg no está instalado. Por favor, instálalo para mejorar la recuperación de videos dañados.")
            return frame_paths
        
        # Crear un directorio temporal para los frames de FFmpeg
        with tempfile.TemporaryDirectory() as temp_dir:
            # Comando FFmpeg para extraer frames con opciones robustas
            ffmpeg_cmd = [
                "ffmpeg",
                "-err_detect", "ignore_err",
                "-i", video_path,
                "-vsync", "0",  # No sincronizar video
                "-vf", f"fps={frame_rate}",
                "-q:v", "2",    # Alta calidad
                f"{temp_dir}/frame_%04d.jpg"
            ]
            
            # Ejecutar FFmpeg
            result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"FFmpeg reportó problemas: {result.stderr[:500]}...")
            
            # Copiar los frames al directorio de salida
            import glob
            temp_frames = sorted(glob.glob(f"{temp_dir}/frame_*.jpg"))
            
            for i, temp_frame in enumerate(temp_frames):
                frame_path = os.path.join(output_dir, f"frame_{i:04d}.jpg")
                shutil.copy2(temp_frame, frame_path)
                frame_paths.append(frame_path)
    
    except Exception as e:
        print(f"Error durante la extracción con FFmpeg: {str(e)}")
    
    print(f"Se extrajeron {len(frame_paths)} frames del video.")
    return frame_paths
def extract_text_from_image(image_path, vl_chat_processor, tokenizer, vl_gpt):
    # Construir el mensaje en el formato esperado por DeepSeek-VL
    conversation = [
          {
            "role": "User",
            "content": "<image_placeholder> Extract ONLY the main visible text in this image and give it back. Ignore TikTok interface elements such as: hashtags, user IDs, counters, buttons, mentions, tags, or any text overlaid by the platform. Focus exclusively on text that is part of the original video content. The text may be in English or Spanish. Preserve the original format (uppercase/lowercase) and organize the text in natural reading order. Do not add interpretations, context, or explanations. Return ONLY the extracted text."
                        "In order to give back the text extracted use this form: The text extracted in the image is: <here put the text you have extracted>.",            
            "images": [image_path]
        },
        {
            "role": "Assistant",
            "content": ""
        }
            ]

    # load images and prepare for inputs
    pil_images = load_pil_images(conversation)
    prepare_inputs = vl_chat_processor(
            conversations=conversation,
            images=pil_images,
            force_batchify=True
        ).to(vl_gpt.device)

    # run image encoder to get the image embeddings
    inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

    # run the model to get the response
    outputs = vl_gpt.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=100,
            do_sample=False,
            use_cache=True
        )

    generated_text = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
    
    # Eliminamos la cabecera impuesta
    generated_text = generated_text.replace("The text extracted in the image is", "")
    
    generated_clean_text = clean_text(generated_text)
    
    return generated_clean_text
def clean_text(text):
 
    # Pasamos a minusculas
    text = text.lower()
    
    # Eliminamos las palabras que empiecen por @
    text = re.sub(r'@\w+\s*', '', text).strip()
    
    # Eliminamos la cabecera inicial
    text = text.replace("the text extracted in the image is", "")
    
    # Eliminamos la palabra tiktok
    text = text.replace("tiktok", "")
    
    # Eliminamos <here put the text you have extracted> para cuando el modelo se equivoca
    text = text.replace("<here put the text you have extracted>", "")
    
    # Eliminar caracteres no deseados
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    
    # Pasamos los spell checkers para eliminar nombres de usuario u otras cadenas sin información
    text = filter_no_valid_words(text)
        
    # Eliminar líneas vacías
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    
    if not lines:
        lines = ["<THERE IS NO TEXT IN THIS FRAME>"]
    
    return '\n'.join(lines)


def process_video(video_path, output_dir, frame_rate=1):
    
    print("Iniciando procesamiento de video...")
    start_time = time.time()
    
    model_path = "deepseek-ai/deepseek-vl-1.3b-chat"
    vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
    tokenizer = vl_chat_processor.tokenizer
    vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()
        
    # Extraer frames
    print(f"Extrayendo frames del video a {frame_rate} frame(s) por segundo...")
    frame_paths = extract_frames(video_path, output_dir, frame_rate)
    
    # Procesar cada frame para extraer texto
    print("Extrayendo texto de los frames...")
    results = []
    
    for i, frame_path in enumerate(frame_paths):
        print(f"Procesando frame {i+1}/{len(frame_paths)}: {frame_path}")
        text = extract_text_from_image(frame_path, vl_chat_processor, tokenizer, vl_gpt)
        timestamp = i / frame_rate  
        results.append({
            "timestamp": timestamp,
            "text": text
        })
        
    return results

def filter_no_valid_words(texto):
    global spell_en
    global spell_es
        
    palabras = texto.split()  
    palabras_filtradas = []
        
    for palabra in palabras:
        if len(spell_en.known([palabra])) > 0 or len(spell_es.known([palabra])) > 0 :
            palabras_filtradas.append(palabra)
        
    return " ".join(palabras_filtradas)

def extract_text_from_videos_in_directory(directory, output_frame_dir, df, frame_rate= 1):
     
    # Creamos el directorio donde vamos a guardar los frames
    os.mkdir(output_frame_dir)
    
    # Procesar cada video en el directorio
    for filename in os.listdir(directory):
    
        if filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv')):
            video_path = os.path.join(directory, filename)
            
            print(f"\nProcesando video: {filename}")
            texts = process_video(video_path, output_frame_dir, frame_rate)
            
            resultado = []
            for item in texts:
                mm_ss = str(timedelta(seconds=int(item['timestamp'])))[2:7]  # formato MM:SS
                texto = item['text']
                resultado.append(f"[{mm_ss}] {texto}")

            # Concatenar todo con saltos de línea
            salida = '\n'.join(resultado)
            
            df.loc[df["video"] == filename, "extracted_text"] = salida
    
    shutil.rmtree(output_frame_dir)  

if __name__ == "__main__":
    
    # Definimos los correctores ortográficos
    global spell_en
    global spell_es 

    spell_en = SpellChecker()
    spell_es = SpellChecker(language='es')
    
    # Cargamos el JSON
    with open('TFM/segmentación/python/texto/EXIST2025_test_clean.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    # Lo convertimos a DataFrame por comodidad
    records = []
    for key, value in data.items():
        value["original_key"] = key
        records.append(value)

    df = pd.DataFrame(records)
    
    # Eliminamos las columnas que no queremos
    campos_a_eliminar = ["text","path_video","url","annotators", "number_annotators", "gender_annotators","split"]
    df = df.drop(campos_a_eliminar, axis=1)
      
    df["extracted_text"] = ""
    
    video_dir = "TFM/segmentación/python/texto/test/videos/"
    output_frame_dir = "TFM/segmentación/python/texto/frames"
    extract_text_from_videos_in_directory(video_dir, output_frame_dir, df, 1)
    
    df = df[df["extracted_text"] != ""]
     
    dict_json = df.set_index("id_EXIST").to_dict(orient="index")

    # Guardar en un archivo JSON
    with open("TFM/segmentación/python/texto/metadata_EXIST2025_texto_extraido.json", "w", encoding="utf-8") as f:
        json.dump(dict_json, f, ensure_ascii=False, indent=2)