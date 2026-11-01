# XAI y evaluación cross-domain · Felipe Aguilar

**Actualización: 31 de octubre de 2026. Estado: EfficientNet entrenada, comparación con VAE, 36 Grad-CAM reales y latencia B0 CPU/MPS; ResNet y Guatemala pendientes.**

Este documento acompaña `notebooks/04_xai_evaluation.ipynb` y sirve como contribución al informe y guía de defensa. Se consultaron `Proyecto Final.pdf`, la copia local de `Proyecto Final - Baches.docx`, el protocolo de Guatemala y las ramas del repositorio. Las 38,385 imágenes y sus XML ya están en `data/raw/RDD2022/` y pasaron la verificación SHA256 de `dataset_v1`; no se necesita acceso a OneDrive para esta copia ni se modificó el documento compartido.

EfficientNet-B0 se incorpora como segundo clasificador supervisado, sin eliminar el resultado VAE. Entrenada durante seis épocas, seleccionó la época 2 por AP de validación; con umbral fijo 0.7523648738861084 obtuvo F1 0.6444 interno, 0.0744 Noruega y 0.2405 dron. En Noruega omitió 241 de 256 baches. El protocolo, las métricas y los comandos están en [EFFICIENTNET.md](EFFICIENTNET.md).

## 1. Qué está disponible y qué falta

| Entregable | Estado comprobable |
|---|---|
| Comparación consolidada | Ejecutada sobre `efficientnet_b0_pretrained`, `vae_base` y `vae_patch512`; faltan resultados ResNet. |
| Matrices de confusión y curvas PR | Generadas para test interno, Noruega y China_Drone. |
| Domain shift | ΔF1/Δrecall calculados sin cambiar el umbral. Guatemala pendiente. |
| Análisis de errores | Conteos por país/daño y revisión visual de los 36 casos B0 seleccionados con seed 42. |
| Grad-CAM | 36 mapas del checkpoint B0 entrenado; tres TP/TN/FP/FN por test. ResNet pendiente. |
| Latencia | B0: mediana CPU 96.467 ms y MPS 8.748 ms; batch 1, cuatro hilos CPU, 200 mediciones y 20 calentamientos. Otros modelos pendientes bajo el mismo protocolo. |
| Presentación/video | Guion y preguntas de defensa en este documento; no equivale a haber grabado o entregado el video. |

La base del trabajo es `origin/feature/vae-variants` (incluye los componentes compartidos y ResNet). El módulo de evaluación se conserva en `feature/xai-evaluation`; `feature/efficientnet` parte de ese commit e integra entrenamiento y resultados B0. No se alteraron los manifests congelados ni los resultados VAE originales. El checkpoint B0 está en `checkpoints/efficientnet/efficientnet_b0_pretrained/best.pt`, fuera de Git. La procedencia se registra en `run_provenance.json` junto a los resultados.

## 2. Protocolo experimental

- Unidad: **imagen con/sin D40**. Es clasificación binaria, no detección de objetos: no reportar mAP de detección ni llamar bounding boxes predichas a un mapa Grad-CAM.
- `dataset_v1`: train 19,476; val 4,173; test_id 4,174; test_ood/Noruega 8,161; test_perspective/China_Drone 2,401. Guatemala es un conjunto externo separado, no parte de estos números.
- SHA256 de splits: `c7b4690943e597a382ab0f7a8ad1298317f267901476b72111a1b1a8041a8614`. El consolidador verifica también los hashes de imágenes/etiquetas y anotaciones frente al lock.
- Se exige una predicción por cada imagen oficial evaluada, sin duplicados ni omisiones, mismas etiquetas/splits/versión, score finito y un único umbral por experimento. Las corridas limitadas de prueba no entran en la comparación final.
- ResNet y EfficientNet: checkpoint por AP de validación, umbral por máximo F1 de validación; salida 0 siempre es D40, incluso si existen tareas auxiliares. `selected_classifier` identifica el clasificador con mayor AP de validación, no de test; se conserva `selected_resnet` para comparar las variantes ResNet.
- VAE: checkpoint por mínima pérdida de validación sin daño tras warm-up; estadísticas de normalización solo de train sin daño; score principal elegido por AP de validación, umbral por F1 de validación. **El VAE se entrenó con `has_any_damage == 0`, no con todas las imágenes `has_D40 == 0`.** Las dos variantes disponibles usan el score máximo de error por parche.
- No ajustar hiperparámetros, modelos ni umbrales con test_id, Noruega, China_Drone o Guatemala. La auditoría del umbral no lo sustituye por uno nuevo.
- Comparar siempre sobre las mismas imágenes. Diferente resolución es parte del costo de cada variante y debe declararse; no atribuir una diferencia de latencia exclusivamente a la arquitectura.

## 3. Resultados disponibles y texto para el informe

Resultados recalculados desde `reports/experiments/vae/*/predictions.csv`, no reentrenados:

| Modelo | Dominio | N | Precisión | Recall | F1 | AP | % marcadas D40 |
|---|---|---:|---:|---:|---:|---:|---:|
| VAE base, 128 px | Test interno | 4174 | 0.1216 | 1.0000 | 0.2169 | 0.1083 | 98.87 |
| VAE patch, 512 px | Test interno | 4174 | 0.1202 | 0.9980 | 0.2146 | 0.1054 | 99.83 |
| VAE base | Noruega | 8161 | 0.0314 | 1.0000 | 0.0608 | 0.0349 | 100.00 |
| VAE patch | Noruega | 8161 | 0.0314 | 1.0000 | 0.0608 | 0.0484 | 100.00 |
| VAE base | China_Drone | 2401 | 0.0277 | 0.9688 | 0.0539 | 0.0249 | 93.17 |
| VAE patch | China_Drone | 2401 | 0.0303 | 0.8438 | 0.0585 | 0.0216 | 74.26 |

> Se observó una caída de F1 entre el test interno y Noruega de 0.1561 para VAE base y 0.1538 para VAE patch512 (15.61 y 15.38 puntos porcentuales). Sin embargo, el problema no aparece únicamente al cambiar de país: en el test interno ambos modelos ya clasifican como D40 casi todas las imágenes. En Noruega producen 256 verdaderos positivos, 7905 falsos positivos y cero verdaderos negativos/falsos negativos. Por ello, recall 1 no demuestra discriminación útil. La precisión coincide con la prevalencia del dominio (3.14 %).
>
> El error de reconstrucción no es específico de baches y puede responder a cambios generales de apariencia. No atribuimos estos errores a sombras, clima o vehículos sin inspeccionar las imágenes. La prevalencia también cambia: 12.03 % en test interno, 3.14 % en Noruega y 2.67 % en China_Drone. Por tanto, ΔF1 no aísla causalmente el cambio visual. La evaluación de Guatemala y la comparación con ResNet permanecen pendientes; no se concluye generalización a vías guatemaltecas.

`pr_auc` representa **average precision (AP)** de sklearn, no la integral trapezoidal de la curva PR. Una referencia útil es la prevalencia, aunque una diferencia pequeña no prueba significancia estadística. No se estimaron intervalos de confianza ni se comprobó independencia por tramo/secuencia.

### Auditoría de una discrepancia numérica

Las seis matrices de los conjuntos test reproducen las archivadas. En `vae_base/val`, el archivo de métricas declara 503 TP/1 FN y el CSV recalculado produce 502 TP/2 FN. La fila `India__India_007420` queda aproximadamente 1.073 × 10⁻¹¹ por debajo del umbral al leer en float64, pero queda igual al convertir ambos números a float32. Esto es consistente con la serialización del score float32 con menos dígitos que el umbral; no se verificó de nuevo el checkpoint original.

Se preservan los CSV y el umbral y se publica la diferencia en `archive_discrepancies.csv`, en lugar de corregir los resultados manualmente. No cambia la conclusión sobre los conjuntos externos. Para una publicación definitiva, solicitar al responsable el reexportado con precisión numérica explícita y comprobación de igualdad antes/después de escribir el CSV.

## 4. Cómo ejecutar y revisar los resultados

Desde la raíz del repositorio, con las dependencias de `requirements.txt` instaladas:

```bash
python3 -m src.evaluation.cross_domain
python3 -m unittest discover -s tests -v
```

También se puede ejecutar `notebooks/04_xai_evaluation.ipynb` de principio a fin. No descarga datos ni entrena modelos.

Salidas en `reports/xai_evaluation/`:

- `metrics.csv`, `domain_shift.csv`, `by_country.csv`: comparación, degradación y desagregación.
- `<experimento>/confusion.png`, `precision_recall.png`, `error_groups.json`: matrices, curvas y errores por daño/país.
- `cases.csv`, `case_coverage.csv`: hasta tres ejemplos por TP/TN/FP/FN y dominio, semilla 42, estables ante reordenamiento del CSV. Los grupos con menos de tres ejemplos se declaran incompletos; no duplicarlos ni cambiar el umbral.
- `archive_discrepancies.csv`, `status.json`: discrepancias, hashes de los insumos y entregables pendientes.
- `latency_reported.csv`: **latencias históricas**, no benchmark nuevo.

Revisar cada foto y registrar zona observada, región activada, concordancia con cajas D40 y limitación. La revisión posterior B0 está en `reports/xai_evaluation/efficientnet_b0_pretrained/gradcam/visual_review.md`; los campos vacíos de las exportaciones automáticas no sustituyen esta revisión. Distinguir observación («la activación está en el borde superior») de hipótesis («podría usar contexto»). Los grupos `other_damage_classes` se solapan; no sumar sus conteos como imágenes únicas.

## 5. Grad-CAM de ResNet y EfficientNet

Al recibir `config.yaml`, `metrics.json`, `predictions.csv`, `history.csv`, `best.pt` y las imágenes, colocar los reportes bajo `reports/experiments/resnet/<ID>/` o `reports/experiments/efficientnet/<ID>/`. Consolidar primero y consultar `selected_classifier` en `status.json` (`selected_resnet` para la selección solo entre ResNet).

Sustituir `ID_RESNET` y las rutas por valores reales:

```bash
python3 -m src.xai.gradcam \
  --experiment reports/experiments/resnet/ID_RESNET \
  --checkpoint checkpoints/resnet/ID_RESNET/best.pt \
  --dataset-root data/raw/RDD2022 \
  --output reports/xai_evaluation/ID_RESNET/gradcam
```

Para EfficientNet usar las rutas de su experimento/checkpoint indicadas en `EFFICIENTNET.md`; el mismo comando reconoce la familia. Se explica **el logit D40, no la salida auxiliar ni la clase ganadora**, también para FN. Se promedian espacialmente sus gradientes respecto de `layer4` (ResNet) o `features[-1]` (EfficientNet), se ponderan los mapas y se aplica ReLU. El mapa se interpola al tamaño de entrada y normaliza por imagen. La escala de color no permite comparar magnitud absoluta entre fotos. Un mapa nulo/constante no significa ausencia demostrada de baches.

Cada caso incluye imagen con cajas D40 oficiales, superposición, mapa nativo, probabilidad, umbral, outcome, matriz `.npy` y procedencia. La probabilidad debe reproducir la del CSV con tolerancia absoluta 0.002 (posibles diferencias AMP/float32); si falla, investigar checkpoint/preprocesamiento antes de interpretar. El resize es el mismo estiramiento cuadrado del entrenamiento, no un crop nuevo.

Se calcula masa de activación en la **unión** de cajas D40, fracción de área y razón masa/área; no cuenta dos veces cajas solapadas. Es un diagnóstico aproximado de localización, **no IoU de segmentación, exactitud de detección ni prueba causal**. En fotos sin cajas la métrica es indefinida. Guatemala solo requiere etiquetas de imagen: no se inventan cajas.

Antes de presentar: inspeccionar al menos tres TP/FP/FN en cada dominio donde existan, conservar los casos seleccionados y mostrar también fallos. Con los VAE actuales no existen FN ni TN en Noruega; se debe explicar esta ausencia, no fabricar ejemplos. Grad-CAM aplica a ambos clasificadores; el VAE utiliza error de reconstrucción y no debe presentarse como si tuviera un logit D40.

## 6. Latencia comparable y Guatemala

Benchmark sobre checkpoint entrenado (el ejemplo usa VAE base):

```bash
python3 -m src.evaluation.inference \
  --experiment reports/experiments/vae/vae_base \
  --checkpoint checkpoints/vae/vae_base/best.pt \
  --dataset-root data/raw/RDD2022 \
  --device cpu --threads 4 --count 200 --warmup 20 \
  --output reports/xai_evaluation/vae_base/benchmark_cpu
```

Repetir para cada modelo en la **misma máquina**, con los mismos IDs ordenados de test_id, dispositivo, hilos, warm-up y cantidad. Guardar marca/modelo exactos del equipo junto al JSON si el sistema solo informa la arquitectura. Reportar media, mediana y p95, batch 1, dimensiones y versiones. El benchmark mide **forward más score D40**, sin lectura, transforms ni traslado inicial de la imagen al dispositivo; no es latencia de una aplicación completa. Sincroniza CUDA/MPS. La red usa float32; el score normalizado de VAE usa float64 salvo MPS (float32, limitación del dispositivo). No mezclar dispositivos o scopes en una razón de velocidad.

Los tiempos históricos del repositorio son: VAE base 9.321 ms CPU / 2.284 ms CUDA; VAE patch512 95.570 ms CPU / 5.158 ms CUDA. Excluyen scoring y no registran el hardware exacto ni todos los parámetros de medición. **No prueban una comparación estandarizada** y no son resultados ejecutados aquí. Mantenerlos separados del nuevo `latency.json`.

Para Guatemala seguir `docs/GUATEMALA_PROTOCOL.md`: obtener permisos/procedencia, retirar datos personales, eliminar duplicados y revisar similitud con RDD2022; documentar dependencia por tramo y no afirmar representatividad nacional. El código valida IDs y metadatos, **no sustituye la revisión visual ni detecta por sí solo duplicados por contenido**.

Manifest CSV separado, con estas columnas:

```text
sample_id,image_path,has_D40,has_any_damage,source,capture_session,road_segment,camera,lighting,weather,labeler,reviewer,review_status,notes
```

`review_status` debe ser `approved`, con etiquetas binarias y procedencia/revisor no vacíos; `image_path` es relativo al root externo. Mantener fotos y manifest privado fuera de Git, en `data/external/Guatemala/`. No incluir fotos inciertas sin revisión. No se requiere ninguna partición de entrenamiento Guatemala.

```bash
python3 -m src.evaluation.inference \
  --experiment reports/experiments/vae/vae_base \
  --checkpoint checkpoints/vae/vae_base/best.pt \
  --dataset-root data/external/Guatemala \
  --guatemala-manifest data/external/Guatemala/manifest.csv \
  --device cpu --output data/external/Guatemala/results/vae_base

python3 -m src.evaluation.cross_domain \
  --guatemala-manifest data/external/Guatemala/manifest.csv \
  --guatemala-results data/external/Guatemala/results
```

Repetir inferencia para cada experimento sin modificar sus umbrales. La consolidación incorpora las predicciones disponibles y enumera los modelos aún pendientes; produce matrices y errores por cámara, clima, iluminación y sesión/tramo. Cuando un subconjunto contiene una sola clase, AP queda indefinida; precision/recall/F1 sin denominador usan la convención cero, no evidencia de calidad. Para Grad-CAM externo añadir `--guatemala-manifest` y `--guatemala-predictions` al comando ResNet, con el root externo. Las salidas pueden incluir IDs, procedencia y rutas de fotos: revisar privacidad antes de publicarlas.

## 7. Guion de defensa y presentación/video

**Apertura (20 segundos):** «Mi responsabilidad fue comprobar si el desempeño se mantiene al cambiar de dominio y si las explicaciones visuales son coherentes con la tarea. Separé resultados medidos de etapas todavía pendientes y no ajusté umbrales sobre los conjuntos externos».

**Protocolo (30 segundos):** «Usamos las mismas particiones congeladas para todos. El umbral se decide en validación y se conserva al evaluar Noruega, perspectiva aérea y posteriormente Guatemala. Así evitamos utilizar el examen para preparar al modelo».

**Resultado (40 segundos):** «En Noruega los VAE encontraron todos los positivos, pero también marcaron todas las imágenes negativas. Son 256 TP y 7905 FP: recall 100 %, pero precisión 3.14 % y F1 0.061. La matriz demuestra por qué no basta con presentar recall. El desempeño interno ya era deficiente».

**Explicabilidad (30 segundos):** «Ya inspeccionamos 36 mapas del EfficientNet entrenado. En Norway_007551 el modelo omite el bache y activa el horizonte, fuera de las cajas. En China_Drone_001847 hay activación en la caja, pero el score no supera el umbral. Un mapa atractivo no garantiza acierto ni demuestra causalidad».

**Cierre (20 segundos):** «No podemos recomendar despliegue en Guatemala todavía. Falta completar ResNet, las capturas etiquetadas de Guatemala y la latencia de todos los modelos en el mismo equipo. La contribución permite ejecutar esas etapas de manera reproducible y deja explícito lo que no está medido».

### Distribución sugerida del máximo de 10 diapositivas del equipo

1. Problema, objetivo y unidad de clasificación.
2. Dataset, etiquetas y limitaciones de muestreo.
3. Particiones congeladas y prevención de fuga de información.
4. ResNet y variantes: resultados solo cuando estén entregados.
5. VAE y variantes: normalidad, score y selección.
6. **Felipe:** tabla consolidada y matrices, no solo recall.
7. **Felipe:** domain shift, prevalencias y situación de Guatemala.
8. **Felipe:** Grad-CAM B0 y TP/FP/FN reales; relacionarlos con el fuerte deterioro de Noruega.
9. **Felipe/equipo:** latencia comparable y limitaciones.
10. Conclusiones, pendientes, repositorio/video y atribución de apoyo IA.

En el video mostrar notebook ejecutado, matrices y comando reproducible. Si no se han recibido insumos, decirlo en voz alta; no sustituirlos por mapas sintéticos. La duración total debe acordarse con el equipo según la rúbrica, no deducirse de este fragmento de guion.

### Preguntas probables

- **¿Por qué AP además de F1?** F1 depende del umbral fijado; AP evalúa el ordenamiento de scores a través de distintos cortes. Ambas dependen del conjunto y deben interpretarse con su prevalencia.
- **¿Por qué no mejorar el umbral en Noruega?** Eso convertiría el test externo en validación. Sería otro protocolo, no esta medición de generalización.
- **¿Por qué los VAE fallan?** Observamos poca discriminación D40 y demasiados positivos. La reconstrucción mide anomalía general; la causa visual exacta requiere revisión de fotos, no puede afirmarse solo con estos números.
- **¿Un FP es una calle sana?** No necesariamente. No-D40 incluye grietas y otros daños. Por eso se separan otros daños de imágenes sin daño anotado.
- **¿ΔF1 mide solo cambio de país?** No. Mezcla cambios de apariencia, frecuencia de clases y composición de muestras. China_Drone añade cambio de perspectiva; no debe confundirse con Noruega ni Guatemala.
- **¿Grad-CAM demuestra que el modelo entiende un bache?** No. Localiza evidencia asociada a una salida bajo una aproximación por gradientes; no es prueba causal ni segmentación.
- **¿Cómo explicas un falso negativo?** Se calcula igualmente el mapa del logit D40; explicar la clase negativa respondería otra pregunta. Un mapa vacío se registra, no se rellena.
- **¿Cuál modelo gana?** B0 mejora F1/AP frente al VAE en test interno y dron, pero falla en recall de Noruega. No hay resultados ResNet para compararla. No escoger ni ajustar variantes por su mejor resultado externo tras observar los tests.
- **¿Qué falta para cerrar tu parte?** Reportes/checkpoint ResNet, checkpoints VAE para repetir latencia comparable y Guatemala revisada. RDD2022, entrenamiento B0, sus Grad-CAM y latencia CPU/MPS ya están completos; falta la integración y revisión colectiva del video/presentación.

## 8. Trazabilidad del apoyo IA

Se utilizó Amp como apoyo para inspección del repositorio, implementación, pruebas, consolidación y borrador de este texto. La conversación de origen es [este hilo](https://ampcode.com/threads/T-01a1362c-fd57-7340-bc49-5db870c0902c). Conservar la exportación real del historial para la entrega exigida por el proyecto; este resumen no reemplaza el historial de prompts y no inventa intercambios adicionales.

Petición relevante del usuario: trabajar en el proyecto del repositorio `Pothole-Detection-Guatemala`, siguiendo `Proyecto Final.pdf` y el documento del equipo, para la responsabilidad de Felipe: «XAI + evaluación cross-domain» y «Grad-CAM, error analysis, matrices de confusión, domain shift, latencia, comparación consolidada y apoyo en presentación/video».

La autoría de los datos/resultados VAE corresponde al trabajo existente del equipo. Las pruebas sintéticas verifican software, no desempeño en baches. Felipe debe revisar resultados, interpretar las fotos y asumir la defensa académica del trabajo antes de presentarlo.
