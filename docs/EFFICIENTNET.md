# EfficientNet-B0: segundo clasificador supervisado de baches

## Hipótesis y estado

**Entrenado y evaluado con RDD2022 local verificado el 31 de octubre de 2026.**
El objetivo es comparar EfficientNet-B0 preentrenada con ResNet18 preentrenada para presencia de D40 a nivel de imagen, manteniendo los mismos datos y protocolo. No se promete que gane. El VAE se conserva como hallazgo negativo: su error de reconstrucción no discriminó D40 en las variantes evaluadas.

EfficientNet-B0 combina convoluciones separables en profundidad, bloques MBConv y atención de canales squeeze-and-excitation. B0 es el modelo base de una familia diseñada para escalar profundidad, ancho y resolución conjuntamente. Se reutilizan pesos **ImageNet IMAGENET1K_V1**, se cambia la última capa por un logit D40 y se ajustan todos los parámetros. Se conserva el dropout y stochastic depth propios de B0. El score es sigmoid(logit); el mapa Grad-CAM no es una caja predicha.

Pregunta para el catedrático: **¿qué compromiso entre discriminación D40, generalización externa y costo de inferencia ofrecen dos arquitecturas supervisadas bajo un protocolo compartido?** No comparar únicamente accuracy ni elegir el ganador a partir del test externo.

## Resultados reales de la primera corrida

Las 38,385 imágenes y sus XML pasaron la verificación SHA256 de `dataset_v1` en `data/raw/RDD2022/`, sin regenerar splits. Se entrenó con las 19,476 imágenes de train en Apple M4 Pro/MPS, sin `--limit` ni cambios de hiperparámetros. Parada temprana en la época 6; checkpoint de la época 2 por máxima AP de val. Umbral fijo de validación: **0.7523648738861084**.

| Partición | Precision | Recall | F1 | AP |
|---|---:|---:|---:|---:|
| Validación | 0.6895 | 0.7401 | 0.7139 | 0.7678 |
| Test interno | 0.6179 | 0.6733 | 0.6444 | 0.7306 |
| Noruega | 0.1020 | 0.0586 | 0.0744 | 0.0586 |
| China_Drone | 0.1508 | 0.5938 | 0.2405 | 0.2940 |

**No resuelve la generalización:** en Noruega detecta 15 de 256 baches y omite 241, aunque reduce las falsas alarmas frente al VAE. Mejora F1/AP en test interno y dron frente a las variantes VAE disponibles; no demuestra superioridad frente a ResNet, cuyos resultados aún faltan. No se cambió el umbral tras observar esta falla.

Se generaron e inspeccionaron 36 mapas reales. La [revisión visual](../reports/xai_evaluation/efficientnet_b0_pretrained/gradcam/visual_review.md) incluye TP, TN, FP y FN de los tres tests, con activación fuera de las cajas en varios casos. El [checkpoint seleccionado](../reports/experiments/efficientnet/efficientnet_b0_pretrained/run_provenance.json) queda identificado por SHA256; los pesos permanecen fuera de Git.

Latencia B0 en el mismo Apple M4 Pro: mediana **96.467 ms CPU / 8.748 ms MPS**, batch 1, cuatro hilos CPU, 200 mediciones y 20 calentamientos. Incluye forward float32 + score, no I/O, preprocesamiento ni transferencia. Los JSON de `benchmark_cpu` y `benchmark_mps` conservan muestras e IDs; no comparar estos valores con las latencias históricas VAE medidas con otro protocolo/hardware.

Fuentes: [métricas completas](../reports/experiments/efficientnet/efficientnet_b0_pretrained/metrics.json), [historial](../reports/experiments/efficientnet/efficientnet_b0_pretrained/history.csv), [consolidación](../reports/xai_evaluation/metrics.csv) y [cambio entre dominios](../reports/xai_evaluation/domain_shift.csv). Las 59 pruebas pasaron; se auditaron los 18,909 scores exportados y se verificó la recarga en CPU/MPS. El informe actualizado está en el worktree de main; estos artefactos se mantienen en la rama experimental.

## Comparación controlada

| Decisión | EfficientNet y ResNet de referencia |
|---|---|
| Configuraciones | `configs/efficientnet/exp1_pretrained.yaml` y `configs/resnet/exp2_pretrained.yaml` |
| Datos | `dataset_v1` sin cambios; positivos D40, negativos incluyen otros daños |
| Preprocesamiento | Imagen RGB completa → resize bilineal 224×224 → normalización ImageNet |
| Recortes y augmentation | Desactivados en este experimento |
| Entrenamiento | Fine-tuning completo, batch 32, máximo 15 épocas, seed 42 |
| Optimización | AdamW, lr 0.0001, weight decay 0.0001, CosineAnnealingLR |
| Pérdida | BCEWithLogitsLoss; pos_weight calculado solamente en train |
| Checkpoint | Mayor AP D40 en val; paciencia 4 |
| Umbral | Mayor F1 D40 en val, congelado para todos los tests |
| Evaluación | Test interno, Norway OOD y China_Drone; Guatemala cuando haya fotos revisadas |

Se utiliza deliberadamente el resize completo del proyecto, **no** el recorte central de la receta ImageNet de torchvision: evita perder baches periféricos y mantiene el preprocesamiento comparable. Esto se registra en `config.yaml`; no equivale a reproducir el benchmark ImageNet del artículo.

El presupuesto máximo y la regla de parada son iguales; las épocas efectivas pueden diferir. La comparación incluye arquitectura e inicialización preentrenada, no prueba causalmente que una única pieza de la arquitectura sea responsable. Una sola seed tampoco demuestra significancia estadística. Medir latencia: menos parámetros no garantiza menor tiempo en todo hardware.

Los experimentos antiguos con augmentation conservan sus configuraciones. No atribuirles una ablación controlada sin igualar épocas/paciencia, ni introducir recortes que eliminen D40 sin revisar las etiquetas. Esta incorporación no modifica esos experimentos ni sus resultados.

## Ejecutar

Desde la raíz del repositorio, con las dependencias instaladas y las imágenes ubicadas según `docs/HANDOFF.md`:

```bash
python3 src/data/make_splits.py --check
python3 src/data/share_dataset.py --verify
python3 -m src.models.train_resnet --config configs/efficientnet/exp1_pretrained.yaml --device cuda
```

**El nombre `train_resnet` se conserva por compatibilidad**, pero el entrenador es compartido y selecciona la arquitectura mediante `family` en el YAML. No existe un segundo entrenamiento divergente para EfficientNet.

En Apple Silicon, usar `--device mps`; sin acelerador, `--device cpu`. AMP solo se activa en CUDA. Si las imágenes están en otra carpeta, añadir `--dataset-root /ruta/a/RDD2022`. La primera ejecución preentrenada descarga los pesos oficiales de torchvision; requiere Internet. Si faltan imágenes, se detiene antes de descargar pesos o crear resultados.

Primero puede ejecutarse una prueba reducida con imágenes reales:

```bash
python3 -m src.models.train_resnet --config configs/efficientnet/exp1_pretrained.yaml --limit 500 --epochs 1 --device cuda
```

Se guarda con sufijo `_limit500` y no entra en la comparación final. Si hay que reducir batch por memoria, registrar una nueva configuración y ejecutar también el control ResNet con ese batch; no cambiar silenciosamente solo un modelo. Nunca usar métricas de una prueba reducida como resultados finales.

Salidas de la corrida completa:

```text
reports/experiments/efficientnet/efficientnet_b0_pretrained/
    config.yaml       # arquitectura, pesos, preprocesamiento, versiones y selección
    history.csv       # pérdidas y AP de validación por época
    metrics.json      # umbral fijo, matrices y métricas por dominio
    predictions.csv   # mismo contrato de siete columnas que ResNet/VAE
checkpoints/efficientnet/efficientnet_b0_pretrained/best.pt
```

Los pesos no van a Git. No mezclar checkpoints con configuraciones de otra arquitectura. Los scores del nuevo entrenador se exportan con precisión suficiente para conservar su valor float32 y evitar diferencias de clasificación por redondeo en el umbral.

## Consolidación, Grad-CAM y latencia

```bash
python3 -m src.evaluation.cross_domain
python3 -m src.xai.gradcam --experiment reports/experiments/efficientnet/efficientnet_b0_pretrained --checkpoint checkpoints/efficientnet/efficientnet_b0_pretrained/best.pt --output reports/xai_evaluation/efficientnet_b0_pretrained/gradcam
python3 -m src.evaluation.inference --experiment reports/experiments/efficientnet/efficientnet_b0_pretrained --checkpoint checkpoints/efficientnet/efficientnet_b0_pretrained/best.pt --device cpu --count 200 --warmup 20 --threads 4 --output reports/xai_evaluation/efficientnet_b0_pretrained/benchmark_cpu
```

La consolidación también funciona desde `notebooks/04_xai_evaluation.ipynb`; descubre los resultados de EfficientNet cuando existen, sin fabricar una fila si no existen. `status.json` conserva `selected_resnet` y añade `selected_classifier` (familia/experimento), elegido solo por AP de val entre clasificadores disponibles.

Grad-CAM usa `features[-1]` en EfficientNet, equivalente al último bloque convolucional antes del pooling, y explica el **logit D40** incluso en falsos negativos. En ResNet sigue usando `layer4`. Revisar TP/FP/FN/TN en los dominios disponibles; conservar el muestreo reproducible y documentar categorías vacías. No interpretar mapas obtenidos de pesos sin entrenar como evidencia sobre baches.

Repetir el benchmark para ResNet en la misma máquina, con los mismos IDs, warm-up, hilos y cantidad de mediciones. El comando mide forward más score, no decodificación de archivos ni preprocesamiento. Los tiempos guardados al entrenar son forward solamente y no deben mezclarse con este benchmark.

Para Guatemala usar el mismo comando de inferencia con `--guatemala-manifest` y `--dataset-root` externos, siguiendo `docs/XAI_EVALUACION.md`. El modelo, los pesos, el resize y el umbral permanecen fijados antes de evaluar Guatemala.

## Evidencia necesaria para discutir resultados

1. Curvas de train/val y época seleccionada: ¿aprendió o sobreajustó?
2. AP frente a prevalencia y al clasificador trivial; F1, precision y recall con umbral de val.
3. Matrices en los mismos dominios y caída ID → OOD; distinguir cambio visual de cambio de prevalencia.
4. Casos de baches pequeños, otros daños y contexto; Grad-CAM como diagnóstico, no prueba causal.
5. Latencia comparable y tamaño de modelo; reconocer si gana precisión a costa de tiempo.
6. Mantener el resultado VAE, aun si un clasificador nuevo funciona mejor.

No concluir «EfficientNet es mejor» ni generalización a Guatemala antes de completar estas mediciones. Si no mejora a ResNet, ese resultado también responde la pregunta experimental.

## Verificación y fuentes

Las pruebas automatizadas ejercitan entrenamiento con imágenes sintéticas, recarga del checkpoint, predicciones exportadas, normalización, Grad-CAM y aislamiento de test: cambiar las etiquetas externas no debe cambiar pesos seleccionados ni umbral. Son pruebas de software, no métricas D40 reales. Ejecutar:

```bash
python3 -m unittest discover -s tests -v
```

- Tan, M. y Le, Q. (2019). [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://proceedings.mlr.press/v97/tan19a.html).
- [Torchvision: EfficientNet-B0, pesos y preprocesamiento oficiales](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.efficientnet_b0.html).
- Historial real de apoyo IA: [conversación del proyecto](https://ampcode.com/threads/T-01a1362c-fd57-7340-bc49-5db870c0902c). Petición: «impleméntalo para que después podamos debatir y enseñar esos resultados al catedrático». Se reutilizó el protocolo supervisado y se separó implementación de evidencia experimental; el historial completo debe conservarse para la bitácora académica.
