# Centro de tráfico predictivo para Cuenca

Predicción de velocidad a 1 hora, detección automática de incidentes y aviso V2X (DENM), validado en bucle cerrado con SUMO.

**Proyecto Interciclo de Redes Vehiculares, Universidad de Cuenca.** Angel Apolo, Fernando Terán, Stalyn Ochoa y Karla Barrera.

**Versión evaluada:** etiqueta `v1.1-entrega` en https://github.com/angelrampolo/proyecto-interciclo-rvh

> Todos los resultados vienen de un **gemelo digital de Cuenca en SUMO** y son **datos sintéticos**, no medidas de campo. La justificación de los datos, sus parámetros con fuente y la validación contra datos reales están en el anexo del informe (`v2/informe/informe.pdf`) y en `FORMATO_DATOS.md`.

## Qué hace

El sistema cierra un lazo de cinco pasos:

| Paso | Qué es |
|---|---|
| **Dato** | Detectores por carril en tres avenidas y autos conectados (V2X al 50 %) |
| **Modelo** | Red GCN-GRU que predice la velocidad de 1694 celdas a +60 min, y un detector de incidentes (Gradient Boosting) |
| **Acción** | Mensaje DENM a los autos conectados a menos de 2 km, que recalculan su ruta |
| **Medida** | SUMO mide la demora, la cola y los viajes, con y sin la IA |
| **Visualización** | Tablero web en `localhost:8097` |

| Resultado (test, datos sintéticos) | Línea base | Este proyecto |
|---|---|---|
| Error de predicción a 1 h (MAE) | 3,16 km/h (promedio histórico) | **1,82 km/h** (V2X 50 %: 1,86) |
| Incidentes con cola detectados | 14 % (detector al azar, mismas alarmas) | **90 %** (FAR 4,4 %, 4,5 min) |
| Demora de los autos afectados por un cierre | 314 s (sin aviso) | **227 s (−27,8 %)** |

## Requisitos (versiones usadas)

| Componente | Versión |
|---|---|
| SUMO | 1.12, en la imagen Docker `van3twin:jammy` del repositorio del curso [Pbarbecho/van3twin-docker](https://github.com/Pbarbecho/van3twin-docker) |
| Python | 3.13.12 |
| Librerías | las de `requirements.txt`: numpy 2.5.4, pandas 3.1.1, pyarrow 26.0.1, scikit-learn 1.9.1, torch 2.14.1 (CPU), joblib 1.6.0, matplotlib 3.11.2, pytest 9.2.0 |
| Sistema | Linux con Docker. Para las simulaciones SUMO se recomiendan 16 GB de RAM; el entrenamiento corre en CPU |
| LaTeX (opcional) | `tectonic`, para recompilar el informe |

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

## Cómo repetir el experimento

Hay **un solo script**: `repetir_experimento.sh`.

| Comando | Qué hace | Tiempo |
|---|---|---|
| `./repetir_experimento.sh --solo-figuras` | Regenera tablas, figuras, tablero y pruebas desde los resultados guardados en `v2/resultados/` | ~2 min |
| `./repetir_experimento.sh` | Entrena y evalúa todo (predicción, detector y bucle cerrado en SUMO) desde el dataset | ~2,5 h |
| `./repetir_experimento.sh --desde-cero` | Además construye la red desde OSM y genera el dataset sintético en SUMO | ~13 h + lo anterior |

`PY=venv/bin/python PARALELO=4 ./repetir_experimento.sh` elige el intérprete y cuántas simulaciones corren a la vez.

**Datos de entrada.** El dataset completo pesa 1,5 GB y no cabe en Git, así que se regenera con `--desde-cero`, que es determinista. En el repositorio van:

- la red (`red/cuenca_urbana.net.xml`) y el OSM de origen (`osm/`);
- las tablas pequeñas del dataset (`dias`, `incidentes`, `celdas`, `aristas`);
- todas las salidas que respaldan las cifras del informe (`v2/resultados/`).

**Ver el aviso V2X en SUMO** (ventana gráfica `sumo-gui`):

```bash
./ver_en_sumo.sh            # corrida con IA: naranja = iba al cierre, azul = recibió el DENM y recalculó su ruta
./ver_en_sumo.sh base       # la misma corrida sin IA (nadie avisa)
./ver_en_sumo.sh replicar   # repite las dos corridas sin ventana y regenera resultados y tablero (~30 min)
```

**Abrir el tablero** (solo Python, sin instalar nada más):

```bash
python3 v2/servir_dashboard.py   # http://localhost:8097
```

## Semillas y parámetros

| Qué | Valor |
|---|---|
| Calendario de días (tipo, lluvia, factor de demanda) | semilla 2026 (`dataset/calendario.json`) |
| Simulación de cada día en SUMO y duarouter | `semilla = 1000 + día` |
| 20 días extra con incidentes (101 a 120) | calendario con semilla 2027; cada día con su semilla |
| Barrido de saturación V0 | semillas 101, 202 y 303 → V0 = 17 000 veh/h |
| Entrenamiento GCN-GRU | `torch.manual_seed(42)`; AdamW, lr 2·10⁻³, weight decay 10⁻⁴, oculto 32, 12 épocas (sensores) y 8 (V2X), lotes de 16 |
| Detector (HistGradientBoosting) | `random_state=42`, 300 iteraciones, lr 0,06, 48 hojas, peso de clase 1:20, 4 pliegues fuera de pliegue |
| Bucle cerrado (SIL) | incidente `inc119_3`, radio DENM 2 km, penalización 600 s, caducidad 3 min, calentamiento 60 min, umbral estricto 0,18 |
| División de días | 28 de entrenamiento, 7 de validación, 7 de test (36 a 42) y 4 de estrés (43 a 46); para el detector, 12, 4 y 4 (101 a 120) |

Todos los parámetros del generador sintético y su fuente están en el anexo del informe y en `DOCUMENTACION.md`.

## Qué script genera cada tabla y figura del informe

| Tabla o figura del informe | Script | Archivo de resultados |
|---|---|---|
| Error de predicción por horizonte, desglose y días de estrés | `v2/entrenar_prediccion.py` → `v2/informe/figuras_informe.py` | `v2/resultados/metricas_prediccion_{gt,obs}.json` |
| Detector frente al azar y curva DR–FAR | `v2/entrenar_aid.py` → `figuras_informe.py` | `v2/resultados/metricas_aid_{gt,obs}.json` |
| Resultados del bucle cerrado y cola | `v2/sil_sumo.py` → `figuras_informe.py` | `v2/resultados/sil_inc119_3_gt.json` |
| Validación del sintético frente a datos reales | `v2/validar_sintetico.py` | `v2/resultados/validacion_sintetico.json` |
| Capturas del tablero | `v2/construir_dashboard.py` + `v2/servir_dashboard.py` | `v2/dashboard/` |
| Conteo de pruebas citado en el informe | `pytest v2/pruebas` | `v2/pruebas/informe.json`, `v2/informe/pruebas.tex` |

**Ninguna cifra está escrita a mano.** El banco de pruebas (`v2/pruebas/`, 81 pruebas) recalcula las métricas y verifica que el informe y las diapositivas citen las cifras de los JSON.

## Usar datos reales

Ver `FORMATO_DATOS.md`. Se convierte el dato real al mismo formato, se revisa con `python v2/validar_datos.py <carpeta>` y se usa con `RVH_DATOS=<carpeta>`. El código no cambia.

## Estructura

```
osm/, red/          red vial de Cuenca (OSM → netconvert → poda)
escenario/          información de la red para Python y mapa
etapa2/             generador del dataset sintético en SUMO y su validación
dataset/            tablas pequeñas (las grandes se regeneran)
v2/                 modelos, bucle cerrado, tablero, pruebas e informe
  comun.py               carga de datos y métricas
  entrenar_prediccion.py GCN-GRU
  entrenar_aid.py        detector de incidentes
  sil_sumo.py            SUMO + TraCI + DENM
  dashboard/             tablero
  informe/               informe IEEE, diapositivas y figuras
repetir_experimento.sh  script único
```
