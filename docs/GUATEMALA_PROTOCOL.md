# Conjunto externo de Guatemala

Estado al 23 de octubre de 2026: **pendiente de recolección y etiquetado**.
No se incluye ni se simula un dataset Guatemala en dataset_v1. Esto no bloquea
el entrenamiento y evaluación con RDD2022.

## Recolección y etiquetas

Guardar originales fuera de Git bajo `data/external/Guatemala/images/` y un
manifest separado con `sample_id`, `image_path`, `has_D40`, `has_any_damage`,
`source`, `capture_session`, `road_segment`, `camera`, `lighting`, `weather`,
`labeler`, `reviewer`, `review_status` y `notes`. Usar identificadores de sesión
y tramo; evitar direcciones exactas y datos personales. Mantener las imágenes
con etiquetas inciertas fuera del conjunto final hasta revisión.

Recoger baches, otros daños y vías sin daño visible; variar cámara, iluminación,
clima y superficie. Registrar el criterio de muestreo y publicar los conteos
reales; no afirmar que una selección manual estima la prevalencia vial nacional.
La etiqueta binaria debe seguir el criterio D40 usado por el proyecto; revisar
casos ambiguos entre dos integrantes. Si se añaden cajas para XAI, documentar
la convención de coordenadas. No inventar bounding boxes a partir de Grad-CAM.

Eliminar duplicados exactos y revisar similitud visual; conservar una única
unidad por secuencia/tramo o documentar la dependencia entre fotos. Comprobar
que no sean copias de RDD2022. Revisar EXIF/GPS, rostros y placas antes de
compartir; conservar permiso/procedencia de cada imagen.

## Evaluación

Guatemala se usa solo después de fijar modelo, preprocessing y threshold con
train/val de dataset_v1. No usar Guatemala para early stopping, ajuste de
hiperparámetros ni umbral. Reportar precision, recall, F1, PR-AUC y matriz de
confusión, conteos por clase y errores por condición de captura. Cualquier
adaptación a Guatemala será otro experimento con versión y particiones propias.

La persona responsable de la recolección y la fecha de entrega deben acordarse
con el equipo. Documentar este protocolo no equivale a disponer de las fotos.
