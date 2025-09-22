import json
import os
import pandas as pd
import whisperx
import torch
import numpy as np
import soundfile as sf
import librosa

# Función para clasificar el género (simplificada)
def classify_gender(speakers, audio_file):
    
    # Cargar el audio completo
    audio, sr = librosa.load(audio_file)
    
    # Convertir a mono si es estéreo
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)
    
    for speaker in speakers:
        
        audio_segments = speakers[speaker]
        
        segments = []
        for start_time,end_time in audio_segments:
            # Convertir tiempo a muestras
            start_sample = int(start_time * sr)
            end_sample = int(end_time * sr)
        
            # Extraer segmento de audio para el hablante
            segment = audio[start_sample:end_sample]
            segment = np.array(segment)
            segments.append(segment)
            
        # Concatenamos los segmentos
        concatenated = np.concatenate(segments)
        
        # Extract pitch features
        pitches, magnitudes = librosa.piptrack(y=concatenated, sr=sr)
        
        # Find the pitch with highest magnitude in each frame
        pitch_values = []
        for i in range(pitches.shape[1]):
            index = magnitudes[:, i].argmax()
            pitch = pitches[index, i]
            if pitch > 0:  # Filter out zero pitch frames
                pitch_values.append(pitch)
        
        if not pitch_values:
            return "Unknown"
        
        # Calculate median pitch (more robust than mean)
        median_pitch = np.median(pitch_values)
        
        if median_pitch > 170:  
            print("Female")
        elif median_pitch > 70:  
            print("Male")
        else:
            print("Unknown")
    
    return None
    

def transcribe_audio(audio_file, model, model_a, metadata, lang):
    
    result = model.transcribe(audio_file, language=lang)

    # Alinear el audio con timestamps más precisos
    result = whisperx.align(result["segments"], model_a, metadata, audio_file,device="cuda")
         
    if result["segments"] == []:
        
        if lang == "es":
            result = {"segments": [{"text": "No se ha encontrado habla en este audio", "start": -6000}]}
        
        if lang == "en":
            result = {"segments": [{"text": "No speech found in this audio", "start": -6000}]}
        
     
    formatted_transcripts = []

    # Process each segment
    for segment in result["segments"]:
        text = segment["text"]
        start_seconds = int(segment["start"])
        
        # Format the timestamp as [MM:SS]
        minutes = start_seconds // 60
        seconds = start_seconds % 60
        timestamp = f"[{minutes:02d}:{seconds:02d}]"
        
        # Add this formatted segment to our list
        formatted_transcripts.append(f"{timestamp} {text}")

    # Join all formatted segments into a single string if needed
    full_transcript = "\n".join(formatted_transcripts)
        
    return full_transcript

def extract_text_from_videos_in_directory(directory, df):
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Procesar cada video en el directorio
    for filename in os.listdir(directory):
        
        video_name = filename.replace(".mp3",".mp4")
        lang = df[df["video"] == video_name]["lang"].values[0]
        
        model = whisperx.load_model("large-v3", device, language=lang)
        model_a, metadata = whisperx.load_align_model(language_code=lang, device=device,  model_name="WAV2VEC2_ASR_LARGE_LV60K_960H")
      
        if filename.lower().endswith(('.mp3')):
            audio_path = os.path.join(directory, filename)
            
            print(f"\nProcesando audio: {filename}")
            transcription = transcribe_audio(audio_path, model, model_a, metadata, lang)
            df.loc[df["video"] == video_name, "transcription"] = transcription
            
            
if __name__ == "__main__":
    
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
    
    df["transcription"] = ""
    
    audio = "../hghub/ownDatasets/TFM_EXIST/audio/test/audios/"
    extract_text_from_videos_in_directory(audio, df)
    
    df = df[df["transcription"] != ""]
     
    dict_json = df.set_index("id_EXIST").to_dict(orient="index")

    # Guardar en un archivo JSON
    with open("TFM/segmentación/python/texto/metadata_EXIST2025_transcripcion_WHISPER_X.json", "w", encoding="utf-8") as f:
        json.dump(dict_json, f, ensure_ascii=False, indent=2)