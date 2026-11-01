# Proyecto de baches: reporte y estado de implementación

Esta rama `main` contiene el reporte del proyecto y sus figuras. El código experimental se mantiene separado en las ramas de implementación; esta entrega no lo fusiona a `main`.

## Reporte

- [Informe actualizado en PDF](docs/Informe_Proyecto_Baches.pdf)
- [Informe editable en Word](docs/Informe_Proyecto_Baches.docx)
- [Fuente del informe en Markdown](docs/INFORME_PROYECTO_BACHES.md)

El documento sigue las ocho secciones de `Proyecto Final.pdf`: problema, análisis, propuesta, descripción de la solución, herramientas y uso crítico de IA, resultados, conclusión y bibliografía. Incluye matrices de confusión, evaluación entre dominios, prompts relevantes, requisitos pendientes y una guía de defensa.

**Por qué añadir EfficientNet-B0:** el VAE de reconstrucción no produjo una decisión D40 útil con el umbral seleccionado. Un clasificador supervisado aprende directamente la presencia de baches y permite comparar otra arquitectura con ResNet18 usando los mismos datos y controles. Se conserva el VAE como resultado negativo; todavía no se ha demostrado que EfficientNet supere a ResNet.

## Ramas de implementación

- [`feature/xai-evaluation`](https://github.com/Vann06/Pothole-Detection-Guatemala/tree/feature/xai-evaluation): auditoría de predicciones, evaluación cross-domain, Grad-CAM, latencia y consolidación de resultados VAE.
- [`feature/efficientnet`](https://github.com/Vann06/Pothole-Detection-Guatemala/tree/feature/efficientnet): parte de la anterior y añade el entrenamiento de EfficientNet-B0 con el protocolo compartido, sus pruebas y documentación integrada.

Grad-CAM es una técnica de explicación, no un segundo clasificador. ResNet y VAE corresponden al trabajo previo del equipo.

## Qué está comprobado y qué falta

**EfficientNet-B0 ya está entrenada con datos reales.** Se verificaron por SHA256 las 38,385 imágenes y sus XML de `data/raw/RDD2022/`. El entrenamiento usó 19,476 imágenes, se detuvo en la época 6 y seleccionó la época 2 por AP de validación. El umbral 0.7523648738861084 se fijó en validación, sin ajustarlo en los tests.

| Test | Precision | Recall | F1 | AP |
|---|---:|---:|---:|---:|
| Interno | 0.6179 | 0.6733 | 0.6444 | 0.7306 |
| Noruega | 0.1020 | 0.0586 | 0.0744 | 0.0586 |
| China_Drone | 0.1508 | 0.5938 | 0.2405 | 0.2940 |

El modelo mejora F1/AP frente al VAE en test interno y dron, pero **omite 241 de 256 baches en Noruega**. No está validado para despliegue en Guatemala. No hay resultados verificables del control ResNet para afirmar que B0 lo supera.

Se generaron e inspeccionaron 36 mapas Grad-CAM reales. La latencia mediana B0 fue 96.467 ms CPU y 8.748 ms MPS en Apple M4 Pro, con batch 1, cuatro hilos CPU y 200 mediciones tras 20 calentamientos. Son tiempos de forward más score, no de la aplicación completa ni una comparación con el hardware histórico del VAE.

Las 59 pruebas automatizadas pasaron y las métricas se auditaron contra los CSV completos. Ya no se requiere acceso a OneDrive para esta copia local. Los datos crudos y checkpoints permanecen fuera de Git; los reportes completos de la corrida están en el checkout de `feature/efficientnet`.

**Corte experimental del informe: 31 de octubre de 2026, hora de Guatemala.** Esta entrega publica los resultados reales en `feature/efficientnet` y el informe actualizado en `main`, sin fusionar el código experimental. Las menciones del documento a resultados preparados localmente o publicación pendiente describen su corte anterior a esta publicación; no indican que el entrenamiento esté pendiente. Quedan pendientes ResNet, Guatemala, comparación de latencias entre modelos, revisión colectiva, video, presentación y requisitos administrativos.
