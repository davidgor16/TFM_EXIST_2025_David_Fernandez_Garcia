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

├── segmentación/

│   └── python/            # Scripts para realizar la segmentación por canal

├── entrenamiento/         # Entrenamiento por modalidad (texto/audio/vídeo), logs y checkpoints

├── combinación/

│   └── python/            # Mecanismos de fusión HARD/ SOFT y resultados por fusión

├── evaluacion/            # Ficheros necesarios para evaluación

└── results/               # Resultados por modelo tras CV en train



> **Nota**: ajusta rutas y nombres a tus scripts reales.


---



