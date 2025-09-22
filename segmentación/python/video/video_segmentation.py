import os
import ffmpeg

def remove_audio_ffmpeg(video_path, output_path=None):

    if output_path is None:
        # Crear nombre para archivo de salida basado en el nombre del video
        filename, extension = os.path.splitext(os.path.basename(video_path))
        output_path = os.path.join(os.path.dirname(video_path), f"{filename}_sin_audio{extension}")
    
    try:
        # Configurar el proceso de ffmpeg
        stream = ffmpeg.input(video_path)
        # Copiar solo el video, sin el audio
        stream = ffmpeg.output(stream, output_path, an=None, vcodec='copy')
        
        # Ejecutar ffmpeg sin mostrar output
        ffmpeg.run(stream, quiet=True, overwrite_output=True)
        
        return output_path
    except ffmpeg.Error as e:
        print(f"Error al eliminar audio con ffmpeg: {e.stderr.decode() if e.stderr else str(e)}")
        return None



video_dir = "../../../../../hghub/ownDatasets/TFM_EXIST/2025EXIST/dataset/videos/test/videos/"
output_dir = "../../../../../hghub/ownDatasets/TFM_EXIST/video/test/videos/"

extracted_files = []

# Procesar cada archivo de video en el directorio
for filename in os.listdir(video_dir):
    if filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv')):
        video_path = os.path.join(video_dir, filename)
        mute_video_filename = os.path.splitext(filename)[0] + ".mp4"
        mute_video_path = os.path.join(output_dir, mute_video_filename)
            
        print(f"Extrayendo audio de {filename}...")
        result_path = remove_audio_ffmpeg(video_path, mute_video_path)

        if result_path:
            extracted_files.append(result_path)
            print(f"Audio extraído y guardado en {result_path}")