import json

# Cargamos el archivo JSON
with open('EXIST2025_test_clean.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

campos_a_eliminar = ["text","url","annotators", "number_annotators", "gender_annotators","split"]

for key, value in data.items():
        
    # Eliminamos los campos que no queremos
    for campo in campos_a_eliminar:
        if campo in value:
            del value[campo]
    
# Guardamos el JSON modificado
with open('metadata_EXIST2025_video.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print("JSON modificado guardado con éxito")