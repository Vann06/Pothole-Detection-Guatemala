# Revisión visual de Grad-CAM — 31 de octubre de 2026

Se inspeccionaron los 36 mapas reales: tres ejemplos por TP, TN, FP y FN en cada test, seleccionados uniformemente con seed 42. No se eligieron nuevos ejemplos después de ver los mapas. `cases.csv` conserva la salida automática y sus diagnósticos; esta nota registra la revisión posterior, que no realiza el generador y debe revisar también el equipo.

Checkpoint: época 2; umbral fijo 0.7523648738861084. Todos los mapas explican el logit D40; ninguno es constante. Los colores se normalizan por imagen: un rojo intenso no indica por sí mismo un score alto, ni permite comparar magnitudes absolutas entre imágenes. Las observaciones siguientes no prueban causalidad ni justifican cambiar etiquetas.

## Test interno

- FN `Japan_010336`: activación lateral derecha sobre vegetación/borde, fuera de D40; masa en cajas 0. FN `Japan_005587` y `Japan_008908`: máximos fuera de las cajas, aunque existe solapamiento parcial de la activación interpolada.
- FP `India_001449` y `India_005911`: activación sobre sectores del asfalto sin D40 anotado. FP `Japan_005211`: activación inferior sobre una tapa circular y también en el fondo. No concluir que la etiqueta está equivocada solo por la textura visible.
- TN `United_States_004405`: contexto lateral derecho; `Czech_001345`: cielo/estructuras superiores; `United_States_004509`: esquina inferior derecha. Un TN también puede tener un mapa de color intenso porque se explica D40 y se normaliza cada mapa.
- TP `India_004553`: activación sobre la zona de daño y pavimento adyacente; 10.13 % de masa en cajas que ocupan 3.26 % del mapa. `Japan_012931` y `India_009071`: activación próxima a los baches, pero los máximos no quedan dentro de las cajas. Una clasificación correcta no garantiza localización correcta.

## Noruega

- FN `Norway_000702`, `Norway_000130` y `Norway_007551`: activación principalmente en contexto superior/lateral; masa en cajas D40 igual a cero en los tres. Los dos primeros tienen cajas de solo 0.0179 % del mapa redimensionado. Esto motiva estudiar escala y contexto, pero no demuestra que expliquen todos los 241 FN.
- FP `Norway_003863`: borde con nieve; `Norway_005487`: talud lateral; `Norway_001306`: orilla derecha. Son asociaciones visibles del mapa, no una prueba causal del origen de la falsa alarma.
- TN `Norway_001496`, `Norway_002882` y `Norway_003650`: activación lateral, perimetral o superior, aun con scores bajos.
- TP `Norway_005345` y `Norway_000009`: activación amplia próxima a grupos de cajas, sin delimitar cada bache. TP `Norway_006655`: máximo fuera de la caja y hacia la orilla derecha; el acierto binario no basta para validar la explicación.

## China_Drone

- FN `China_Drone_001045` y `China_Drone_001847`: hay activación en las regiones anotadas pese a scores inferiores al umbral. En el segundo, 41.42 % de masa está en cajas que ocupan 4.38 % del mapa; score 0.2325. FN `China_Drone_000057`: máximo hacia acera/contexto, con score 0.7411, todavía inferior al corte fijo.
- FP `China_Drone_000124` y `China_Drone_002244`: actividad en acera/alcorque; `China_Drone_000754`: sombras de vegetación. No hay D40 anotado en estos ejemplos.
- TN `China_Drone_000920`, `China_Drone_002163` y `China_Drone_001693`: activaciones en bordes/esquinas, sin cajas D40.
- TP `China_Drone_002143` y `China_Drone_000205`: activación próxima al daño y extendida a su entorno. TP `China_Drone_002245`: máximo sobre tapa circular fuera de las cajas; solo 1.99 % de masa dentro de ellas.

## Alcance y privacidad

Es una muestra diagnóstica equilibrada por resultado, no representativa de las frecuencias reales de error. Usar las matrices completas para esas frecuencias. La fracción de masa no es IoU ni prueba de que el modelo comprenda daño. El mapa nativo es 7 × 7.

No se identificaron matrículas o textos personales legibles en las láminas revisadas. `Norway_000130` contiene retratos publicitarios de baja resolución; se omite del informe ilustrado. No publicar fotografías adicionales sin revisar licencia y privacidad. Esta inspección asistida requiere revisión crítica del equipo antes de la entrega académica.
