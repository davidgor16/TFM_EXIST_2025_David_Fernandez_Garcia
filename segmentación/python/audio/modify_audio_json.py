import json

# Cargamos el archivo JSON
with open('EXIST2025_test_clean.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

campos_a_eliminar = ["text","video","path_video","url","annotators", "number_annotators", "gender_annotators","split"]

for key, value in data.items():
    
    # Modificamos y añadimos los campos
    value["audio"] = value["video"].replace(".mp4", ".mp3")
    value["path_audio"] = "audio/" + value["audio"]
    
    # Eliminamos los campos que no queremos
    for campo in campos_a_eliminar:
        if campo in value:
            del value[campo]
    
# Guardamos el JSON modificado
with open('metadata_EXIST2025_audio.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print("JSON modificado guardado con éxito")