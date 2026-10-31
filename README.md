# Proyecto de baches: reporte y estado de implementación

Esta rama `main` contiene el reporte del proyecto y sus figuras. El código experimental se mantiene separado en las ramas de implementación; esta entrega no lo fusiona a `main`.

## Reporte

- [Informe en PDF — 14 páginas](docs/Informe_Proyecto_Baches.pdf)
- [Informe editable en Word](docs/Informe_Proyecto_Baches.docx)
- [Fuente del informe en Markdown](docs/INFORME_PROYECTO_BACHES.md)

El documento sigue las ocho secciones de `Proyecto Final.pdf`: problema, análisis, propuesta, descripción de la solución, herramientas y uso crítico de IA, resultados, conclusión y bibliografía. Incluye matrices de confusión, evaluación entre dominios, prompts relevantes, requisitos pendientes y una guía de defensa.

**Por qué añadir EfficientNet-B0:** el VAE de reconstrucción no produjo una decisión D40 útil con el umbral seleccionado. Un clasificador supervisado aprende directamente la presencia de baches y permite comparar otra arquitectura con ResNet18 usando los mismos datos y controles. Se conserva el VAE como resultado negativo; todavía no se ha demostrado que EfficientNet supere a ResNet.

## Ramas de implementación

- [`feature/xai-evaluation`](https://github.com/Vann06/Pothole-Detection-Guatemala/tree/feature/xai-evaluation): auditoría de predicciones, evaluación cross-domain, Grad-CAM, latencia y consolidación de resultados VAE.
- [`feature/efficientnet`](https://github.com/Vann06/Pothole-Detection-Guatemala/tree/feature/efficientnet): parte de la anterior y añade el entrenamiento de EfficientNet-B0 con el protocolo compartido, sus pruebas y documentación integrada.

Grad-CAM es una técnica de explicación, no un segundo clasificador. ResNet y VAE corresponden al trabajo previo del equipo.

## Qué está comprobado y qué falta

La integración pasó 59 pruebas automatizadas. Las métricas reales disponibles corresponden a dos variantes VAE; las pruebas sintéticas de EfficientNet verifican software, no precisión sobre baches. Faltan el entrenamiento real de EfficientNet, los resultados del control ResNet disponibles para comparar, Guatemala, Grad-CAM sobre fotografías reales y latencia estandarizada.

Los paquetes de `dataset_v1` están en el OneDrive institucional indicado por el equipo, pero su contenido no pudo verificarse sin autenticación UVG. No se incluyen imágenes del dataset, checkpoints ni enlaces de acceso restringido en esta entrega. Las instrucciones de ejecución están en las ramas de implementación.

**Corte del informe: 30 de octubre de 2026, hora de Guatemala.** Se conserva el documento tal como fue revisado: las menciones a commits locales y al reporte aún no publicado describen su estado antes de esta entrega. Ahora el reporte se versiona en `main` y los dos commits de implementación se publican en sus respectivas ramas. Esto actualiza el estado de publicación, no los resultados experimentales ni los pendientes académicos.
