import os
import ffmpeg

def extract_audio_ffmpeg(video_path, output_path=None, format="mp3", bitrate="192k"):

    if output_path is None:
        # Crear nombre para archivo de salida basado en el nombre del video
        filename = os.path.splitext(os.path.basename(video_path))[0]
        output_path = os.path.join(os.path.dirname(video_path), f"{filename}.{format}")
    
    try:
        # Configurar el proceso de ffmpeg
        stream = ffmpeg.input(video_path)
        stream = ffmpeg.output(stream, output_path, acodec="libmp3lame" if format == "mp3" else "copy", ab=bitrate)
        
        # Ejecutar ffmpeg sin mostrar output
        ffmpeg.run(stream, quiet=True, overwrite_output=True)
        
        return output_path
    except ffmpeg.Error as e:
        print(f"Error al extraer audio con ffmpeg: {e.stderr.decode() if e.stderr else str(e)}")
        return None

video_dir = "../../../../../hghub/ownDatasets/TFM_EXIST/2025EXIST/dataset/videos/test/videos/"
output_dir = "../../../../../hghub/ownDatasets/TFM_EXIST/audio/test/prueba/"

extracted_files = []
    
# Procesar cada archivo de video en el directorio
for filename in os.listdir(video_dir):
    if filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv')):
        video_path = os.path.join(video_dir, filename)
        audio_filename = os.path.splitext(filename)[0] + ".mp3"
        audio_path = os.path.join(output_dir, audio_filename)
            
        print(f"Extrayendo audio de {filename}...")
        result_path = extract_audio_ffmpeg(video_path, audio_path)
        
         
        if result_path:
            extracted_files.append(result_path)
            print(f"Audio extraído y guardado en {result_path}")

