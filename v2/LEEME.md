# Proyecto Interciclo RVH, versión 2 (revisada y corregida)

Predicción de la velocidad de toda Cuenca a 1 hora, detección automática de incidentes (AID) en los
corredores C1, C2 y C3, y validación en bucle cerrado con SUMO y avisos V2X.

## Cómo verlo

```bash
cd "Proyecto Interciclo/v2"
~/venv_cpu/bin/python servir_dashboard.py      # abre http://localhost:8097
~/venv_cpu/bin/python -m pytest pruebas/ -q     # banco de pruebas (resultado también en el dashboard)
```

## Qué hace cada archivo

| Archivo | Qué hace | Tiempo |
|---|---|---|
| `comun.py` | Carga los datos en dos modos: `gt` (sensores, 0 % V2X) y `obs` (solo autos V2X, conteos corregidos por la penetración). Promedio histórico, grafo vial y métricas | — |
| `entrenar_prediccion.py --modo gt\|obs` | GCN-GRU sobre el grafo de calles: predice +15, +30, +45 y +60 min como corrección del promedio histórico. Evalúa una sola vez en test y en los días de estrés | ~1 h por modo |
| `generar_dias_incidentes.py` | 20 días extra (101 a 120) con el mismo escenario de SUMO y más incidentes en los corredores, para el AID | ~5 h, 5 en paralelo |
| `entrenar_aid.py --modo gt\|obs` | Detector de incidentes: desvío frente a lo habitual (z-score), firma aguas arriba y aguas abajo, contexto de toda la red en ese minuto, lluvia, tendencia y persistencia. Umbral seco y umbral de lluvia elegidos sin mirar el test (predicciones fuera de pliegue en 51 días, FAR ≤ 3 %, meta del curso 3–5 %; también se reporta el punto FAR ≤ 1 %) y comparación con un detector al azar | ~6 min |
| `sil_sumo.py --inc inc119_3` | Bucle cerrado en SUMO real (TraCI): el AID corre en línea y, al confirmar la alarma, envía un DENM a los autos conectados que están a menos de 2 km de recorrido del incidente; su navegador suma 10 min a esa calle, usa los tiempos actuales de la red y solo se desvía si la alternativa es más rápida | ~12 min |
| `validar_datos.py` | Revisa que una carpeta de datos (sintética o real) tenga el formato del proyecto | ~5 s |
| `construir_dashboard.py` | Arma `dashboard/datos.json` y los binarios desde los resultados | ~1 min |
| `pruebas/test_proyecto.py` | Banco de pruebas, reglas y metas: fugas de información, líneas base, metas del curso, SIL y dashboard | ~5 s |
| `pruebas/test_integridad.py` | Banco de pruebas, integridad: causalidad, métricas recalculadas, límites físicos, reproducibilidad, SIL contra el dataset, tablero en Chrome, documentos citan las cifras reales y compilan | ~35 s |

Orden completo: `entrenar_prediccion` (gt y obs) → `generar_dias_incidentes` → `entrenar_aid` (gt y obs)
→ `sil_sumo` → `construir_dashboard` → `pytest`. Todo junto: `../repetir_experimento.sh`.

## Reglas de evaluación

| Regla | Cómo se cumple |
|---|---|
| Ninguna cifra escrita a mano | Todo se calcula y se guarda en `resultados/*.json`; el informe, las figuras y el tablero solo leen esos archivos |
| Línea base justa en predicción | Promedio histórico por paso de 5 min y tipo de día, solo con train, y persistencia |
| Línea base justa en detección | Detector al azar con la misma tasa de falsas alarmas; umbral elegido sin test para FAR ≤ 3 % |
| Incidentes observables | 20 días extra con incidentes que sí forman cola; la DR se reporta sobre todos y sobre los observables |
| Acción medida en SUMO | SIL real con TraCI, misma demanda y semilla, con y sin aviso V2X |
| Robustez V2X | Se evalúan sensores y V2X 50 %, y 30 % y 70 % de penetración en los días de estrés |

## Decisiones de evaluación que conviene saber

- **Incidente con impacto**: su tramo muestra una anomalía sostenida mayor que el percentil 99 del ruido normal de train. La DR se reporta sobre todos y sobre estos.
- **Gracia de 15 min**: la cola que queda mientras se disipa un incidente no cuenta como falsa alarma ni como acierto.
- **Lluvia**: SUMO con lluvia forma colas en toda la red; el detector usa un umbral aparte (modo lluvia), elegido con 10 días lluviosos fuera de pliegue. La FAR se reporta separada por clima.
- **FAR en test mayor que la estimada**: con el umbral elegido para 3 % fuera de pliegue, el test da 3,7 % (días normales) y 6,5 % (días con 8 incidentes, donde las colas secundarias de un incidente duran más que la gracia). No se reajustó mirando el test.
- **SIL**: se usa `inc119_3` (cierre total en la Av. de las Américas, 14 min, día seco de test), porque forma una cola clara y medible.

## Informe, diapositivas y guion (`informe/`)

| Archivo | Qué es |
|---|---|
| `informe/informe.pdf` | Informe IEEE con estado del arte, metodología, resultados, discusión, replicabilidad y anexos |
| `informe/diapositivas.pdf` | Diapositivas Beamer (Metropolis en verde) para 10 min |
| `informe/guion.pdf` | Guion por integrante con tiempos, demo y preguntas probables |
| `informe/guia_dashboard.pdf` | Cómo abrir, leer y demostrar el tablero (con capturas) |
| `abrir_dashboard.bat` / `abrir_dashboard.sh` | Abren el tablero en Windows / Linux o Mac (solo Python 3) |
| `informe/figuras_informe.py` | Regenera las figuras desde `resultados/*.json` |
| `informe/figuras/dash_*.png` | Capturas del tablero (Chrome headless con `?captura`) |

Compilar sin red: `tectonic -C -b https://data1.fullyjustified.net/tlextras-2022.0r0.tar informe.tex` (igual para `diapositivas.tex` y `guion.tex`).
El tablero acepta vistas exactas por URL: `?t=14:05&capa=pred&fuente=obs&conj=test_incidentes&celda=592&paso=2&singuia#sil`.

## Banco de pruebas: 81 pruebas, 77 pasan

`~/venv_cpu/bin/python -m pytest pruebas/ -q`. Solo una corrida completa actualiza `pruebas/informe.json` (tablero)
e `informe/pruebas.tex` (el conteo que usan el informe y las diapositivas). Las 4 que fallan son metas reales no
alcanzadas: 3 del AID (MTTD en días normales, DR con V2X) y la precisión de los avisos del SIL (69 % por falsas alarmas previas).

Errores reales que encontró el banco y se corrigieron:
- La capa "Error de la IA" del tablero comparaba con celdas sin autos (se pintaba casi todo rojo).
- Los avisos DENM del SIL nunca caducaban: autos avisados por falsas alarmas esquivaban el cierre por casualidad.
- Los detectores del SIL contaban cada segundo que un auto estaba sobre el lazo (velocidad ~1 km/h); ahora cada
  auto cuenta una vez con su velocidad de cruce, igual que el E1 del dataset (`LAST_STEP_VEHICLE_DATA`).
