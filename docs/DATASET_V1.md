# dataset_v1 — decisiones y evidencia de entrega

Fecha: 23 de octubre de 2026. Preparado a solicitud de Vianka para habilitar
el trabajo del equipo. Las decisiones OOD/perspectiva son decisiones técnicas
de esta entrega; no se afirma que Ricardo, Diego o Felipe las hayan aprobado.
Si el equipo cambia esas decisiones, crear otra versión antes de comparar modelos.

## EDA revisado

Hay 38,385 imágenes etiquetadas y 65,712 objetos. D40 aparece en 3,674 imágenes
(9.57%); las otras 34,711 no contienen D40, pero pueden contener otros daños.
India tiene 19.85% D40 y Japan 13.23%; United_States tiene 2.41%. Las cajas D40
también varían: mediana de área relativa 0.008690 en India y 0.000460 en Norway.
Conservar las doce etiquetas encontradas, sin convertir automáticamente D43,
D44 u otras etiquetas en potholes. El objetivo sigue siendo presencia de D40,
con la ambigüedad propia de las anotaciones originales; no es detección de cajas.

Tablas: `reports/tables/country_summary.csv`, `ood_candidate_analysis.csv` y
`road_status_by_country.csv`. Figuras: `reports/figures/01` a `07`.
El notebook 01 contiene el análisis y ejemplos; sus muestras usan seed 42.

## OOD y perspectiva

**Norway completo → test_ood.** Aporta 8,161 imágenes, 256 D40 y 5,247 sin daño
anotado. Permite evaluar clasificación y los tres grupos del VAE, conservando
India y Japan como fuentes amplias de positivos en entrenamiento. Su menor
prevalencia y el tamaño relativo pequeño de D40 son cambios de dominio relevantes.
No se escogió usando el rendimiento de ningún modelo ni por similitud demostrada
con Guatemala. La diferencia de prevalencia debe considerarse al comparar PR-AUC.

**China_Drone completo → test_perspective.** Sus 2,401 imágenes y 64 D40 sirven
para evaluación adicional de perspectiva aérea. No mezclar con el entrenamiento
principal ni con el OOD geográfico. Solo tiene cinco imágenes sin daño anotado:
no permite conclusiones sólidas sobre ese subgrupo del VAE.

El test original de RDD2022 (9,035 imágenes sin etiquetas en esta copia) queda
fuera de los manifests y de estos experimentos. `test_id` y `test_ood` provienen
del train original etiquetado; no confundir ambos usos de la palabra test.

## Particiones congeladas

Los otros cinco dominios se dividen aproximadamente 70/15/15, estratificando por
país y estado (`pothole`, `other_damage`, `no_annotated_damage`). Se ordenan los
identificadores antes del muestreo, seed 42, usando dos divisiones estratificadas
de scikit-learn. Primero se agrupan copias con el mismo SHA256 de imagen.
Las proporciones exactas pueden variar ligeramente por redondeo y grupos.

| Split | Imágenes | D40 | Sin D40 | Sin daño anotado |
|---|---:|---:|---:|---:|
| train | 19,476 | 2,348 | 17,128 | 4,531 |
| val | 4,173 | 504 | 3,669 | 970 |
| test_id | 4,174 | 502 | 3,672 | 971 |
| test_ood | 8,161 | 256 | 7,905 | 5,247 |
| test_perspective | 2,401 | 64 | 2,337 | 5 |
| Total | 38,385 | 3,674 | 34,711 | 11,724 |

Para ResNet: `pos_weight = 17128 / 2348 = 7.294718909710392` en train.
Para VAE: 4,531 candidatos iniciales sin daño anotado en train. No usar la
ausencia de D40 como sinónimo de normalidad. No rebalancear val ni los tests.

## Auditoría y protección

Se verificaron los 38,385 JPG y XML: existencia, apertura/verificación con Pillow,
dimensiones vs manifest, lectura XML, conteos de objetos/clases y SHA256.
`make_splits.py --check` verifica integridad referencial, etiquetas, cobertura
uno a uno, separación de dominios y ausencia de copias idénticas entre splits.

Se encontró **un par de JPG idénticos**, ambos de United_States:
`United_States__United_States_001623` y `United_States__United_States_003917`.
Ambos quedan en train, con target 0, para evitar fuga entre particiones.

`data/manifests/dataset_v1.lock.json` protege config, código de generación,
manifests y tablas de auditoría con SHA256. `.gitattributes` fija LF para que
las huellas del contrato se conserven al clonar en Windows/Linux. Los JPG/XML
mantienen su huella binaria original en `file_checksums.csv`.

El generador rechaza sobrescribir la versión. No modificar manifests por
integrante. Guardar la versión y el SHA256 de splits con los resultados.
Las pruebas en `tests/test_dataset_release.py` cubren reproducibilidad aun con
entrada reordenada, agrupación de duplicados, bloqueo de sobrescritura,
detección de cambios, rechazo de duplicados entre países y empaquetado ZIP con rutas y huellas verificadas.

## Límites y pendientes reales

La auditoría no es una revisión manual de las etiquetas ni decodificación
exhaustiva de todos los píxeles. No se detectan todas las copias visualmente
similares ni se dispone de identificadores fiables de secuencia/tramo para
agrupar escenas cercanas; el test ID puede contener escenas correlacionadas.
Reportar esta limitación; no prometer independencia por carretera.

Guatemala está pendiente de recolección/etiquetado; su protocolo está en
[GUATEMALA_PROTOCOL.md](GUATEMALA_PROTOCOL.md). La fase de preparación RDD2022
permite que Ricardo, Diego y Felipe comiencen. La publicación de estos cambios
y la carga de ZIPs a Drive deben hacerse antes de entregar acceso al equipo;
no se han realizado automáticamente. Ver [HANDOFF.md](HANDOFF.md).
