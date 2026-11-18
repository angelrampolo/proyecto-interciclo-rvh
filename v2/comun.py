"""Datos comunes del Proyecto Interciclo (v2).

Dos modos de entrada, el objetivo siempre es la verdad de terreno (gt_vel):
  - 'gt'  : sensores de infraestructura que ven a todos los autos (0 % V2X).
  - 'obs' : solo autos conectados V2X (50 % en los dias normales); conteos corregidos por la penetracion.
Reglas de la documentacion del dataset que se respetan aqui:
  division por dias (columna split), estadisticas y promedio historico solo con train,
  ventanas dentro de cada dia, nada de columnas de etiqueta como entrada.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

import os

RAIZ = Path(__file__).resolve().parent.parent
# carpeta de datos: por defecto el dataset sintetico; para usar datos reales (EMOV, camaras) con el mismo formato
# basta con RVH_DATOS=/ruta/al/dato_real (ver FORMATO_DATOS.md y validar_datos.py). El codigo no cambia.
DS = Path(os.environ.get("RVH_DATOS", RAIZ / "dataset"))
RED = RAIZ / "escenario" / "red_info.json"
SALIDA = Path(__file__).resolve().parent / "resultados"
SALIDA.mkdir(exist_ok=True)

PASOS = 288            # intervalos de 5 min por dia
HIST = 12              # 1 h de historia
HORIZONTES = (3, 6, 9, 12)   # +15, +30, +45 y +60 min; el objetivo principal es +60
H = HORIZONTES[-1]
DESDE_HORA = 5         # ventanas cuyo objetivo cae antes de las 05:00 no se usan
TIPOS = ("laborable", "sabado", "domingo")
PICO = [(7 * 12, 9 * 12), (12 * 12 + 6, 14 * 12), (17 * 12 + 6, 19 * 12 + 6)]


def dias():
    d = pd.read_parquet(DS / "dias.parquet")
    if (DS / "dias_aid.parquet").exists():   # dias extra centrados en incidentes (solo para el AID)
        d = pd.concat([d, pd.read_parquet(DS / "dias_aid.parquet")], ignore_index=True)
    return d.set_index("dia_id")


def incidentes():
    i = pd.read_parquet(DS / "incidentes.parquet")
    if (DS / "incidentes_aid.parquet").exists():
        i = pd.concat([i, pd.read_parquet(DS / "incidentes_aid.parquet")], ignore_index=True)
    return i


def celdas():
    return pd.read_parquet(DS / "celdas.parquet").sort_values("celda").reset_index(drop=True)


def ids(split):
    d = dias()
    return d.index[d.split == split].tolist()


def es_pico(t):
    return any(a <= t < b for a, b in PICO)


CANALES = ["vel_rel", "ocupacion", "densidad", "flujo", "hay_dato", "desvio_ha"]


def cargar_dias(lista, modo):
    """Devuelve X [D,288,N,5] (sin desvio_ha), Y [D,288,N] gt_vel km/h, M [D,288,N] hay autos (gt), meta."""
    cel = celdas()
    pos = {c: i for i, c in enumerate(cel.celda)}
    N, D = len(cel), len(lista)
    vff = cel.v_ff.to_numpy(np.float32)
    X = np.zeros((D, PASOS, N, 5), np.float32)
    Y = np.repeat(vff[None, None], D * PASOS, 0).reshape(D, PASOS, N).copy()
    M = np.zeros((D, PASOS, N), bool)
    tabla = dias()
    p = "gt" if modo == "gt" else "obs"
    for k, d in enumerate(lista):
        df = pd.read_parquet(DS / "celdas_5min" / f"dia_{d:02d}.parquet")
        ti = (df.t.to_numpy() // 300).astype(int)
        ci = df.celda_id.map(pos)
        ok = ci.notna().to_numpy()
        df, ti, ci = df[ok], ti[ok], ci[ok].astype(int).to_numpy()
        pen = 1.0 if modo == "gt" else float(tabla.loc[d, "penetracion_v2x"])
        hay = (df[f"{p}_n_veh"] > 0).to_numpy()
        vel = df[f"{p}_vel"].where(hay).fillna(df.v_ff).to_numpy()   # sin autos -> flujo libre, nunca 0
        X[k, ti, ci, 0] = vel / df.v_ff.to_numpy()
        X[k, ti, ci, 1] = df[f"{p}_ocupacion"].fillna(0).to_numpy() / (1.0 if modo == "gt" else pen)
        X[k, ti, ci, 2] = df[f"{p}_densidad"].fillna(0).to_numpy() / pen
        X[k, ti, ci, 3] = df[f"{p}_flujo"].fillna(0).to_numpy() / pen
        X[k, ti, ci, 4] = hay
        gt_hay = (df.gt_n_veh >= 1).to_numpy()
        Y[k, ti, ci] = df.gt_vel.where(gt_hay).fillna(df.v_ff).to_numpy()
        M[k, ti, ci] = gt_hay
    meta = tabla.loc[lista, ["tipo_dia", "lluvia", "dow", "penetracion_v2x"]].reset_index()
    return X, Y, M, meta


def promedio_historico(Y, meta):
    """Promedio historico por tipo de dia y paso del dia [3, 288, N] (solo se llama con train)."""
    out = np.zeros((len(TIPOS),) + Y.shape[1:], np.float32)
    for i, t in enumerate(TIPOS):
        sel = (meta.tipo_dia == t).to_numpy()
        out[i] = Y[sel].mean(0)
    return out


def indice_ventanas(D):
    """(dia, t) con t = ultimo paso observado; el objetivo principal es t+12 (60 min)."""
    return [(d, t) for d in range(D) for t in range(HIST - 1, PASOS - H) if (t + H) * 300 >= DESDE_HORA * 3600]


def rasgos_tiempo(t_obj, meta_fila):
    """Rasgos conocidos en la vida real para el instante objetivo: hora (sin, cos), tipo de dia, lluvia."""
    a = 2 * np.pi * t_obj / PASOS
    tipo = [float(meta_fila.tipo_dia == x) for x in TIPOS]
    return np.array([np.sin(a), np.cos(a), *tipo, float(meta_fila.lluvia)], np.float32)


def adyacencia(modo="red"):
    """Adyacencia celda->celda por la topologia vial real (una arista sale de una celda y llega a otra)."""
    cel = celdas()
    pos = {c: i for i, c in enumerate(cel.celda)}
    red = json.load(open(RED))
    cel_de = dict(pd.read_parquet(DS / "aristas.parquet", columns=["id", "celda"]).itertuples(index=False))
    sale = {}
    for e in red["edges"]:
        if e["id"] in cel_de:
            sale.setdefault(e["from"], []).append(pos.get(cel_de[e["id"]]))
    filas, cols = [], []
    for e in red["edges"]:
        i = pos.get(cel_de.get(e["id"]))
        if i is None:
            continue
        for j in sale.get(e["to"], []):
            if j is not None and j != i:
                filas.append(i); cols.append(j)
    ij = np.unique(np.array([filas, cols]), axis=1)
    return ij  # [2, E] dirigido


def metricas(y, p, m):
    """MAE y RMSE en km/h; MAPE solo donde y > 2 km/h (como el avance) para evitar divisiones espurias."""
    d = (p - y)[m]
    y2 = y[m]
    mm = y2 > 2
    return {"MAE": round(float(np.abs(d).mean()), 3), "RMSE": round(float(np.sqrt((d ** 2).mean())), 3),
            "MAPE": round(float(np.mean(np.abs(d[mm]) / y2[mm]) * 100), 2), "n": int(m.sum())}
