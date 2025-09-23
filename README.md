# TFM — Segmentación multimodal y fusión de clasificadores para la detección de sexismo en TikTok

> **Trabajo Fin de Máster**: sistema segmental y de **fusión tardía** para la detección automática de sexismo en vídeos de TikTok, dentro del marco **EXIST 2025**.  
> Modalidades: **texto**, **audio** y **vídeo**. Regímenes de fusión: **HARD** (votación discreta) y **SOFT** (agregación probabilística).

Este repositorio contiene el pipeline extremo a extremo: **segmentación**, **entrenamiento**, **fusión**, **evaluación** y **resultados**.

---

## Visión general

Partimos de la hipótesis de que el sexismo puede manifestarse en distintos canales:

- **Texto**: texto original del dataset, **transcripción ASR** y **texto en pantalla**.  
- **Audio**: habla y rasgos paralingüísticos.  
- **Vídeo**: contenido visual y dinámica.

El enfoque es **segmental**: cada vídeo se descompone en canales; los expertos **unimodales** se entrenan de forma independiente, y una **fusión tardía** integra sus decisiones y/o distribuciones.

---

## Estructura del repositorio

TFM_EXIST_2025_David_Fernandez_Garcia/
|-- segmentación/
|   `-- python/            # Scripts para realizar la segmentación por canal
|-- entrenamiento/         # Scripts para el entrenamiento de modelos por modalidad (texto/audio/vídeo)
|-- combinación/
|   `-- python/            # Mecanismos de fusión HARD/ SOFT y los resultados asociados a cada fusión
|-- evaluacion/            # Ficheros necesarios para realizar la evaluación
`-- results/               # Resultados de cada modelo individual al someterse al conjunto de train mediante evaluación cruzada



> **Nota**: ajusta rutas y nombres a tus scripts reales.


---



