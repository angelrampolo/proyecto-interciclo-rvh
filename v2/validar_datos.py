"""Valida que una carpeta de datos (sintetica o real) tenga el formato que usa el proyecto.

Sirve para la transicion al dato real: cuando lleguen los datos de la EMOV o de una camara, se convierten
a estas tablas y este script dice si el codigo los puede usar tal cual (solo cambia RVH_DATOS).

Uso:  python v2/validar_datos.py [carpeta]        (por defecto: la de RVH_DATOS o dataset/)
"""
import os, sys
from pathlib import Path
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
ESQUEMA = {
    "dias.parquet": {"dia_id": "int", "tipo_dia": "str", "lluvia": "bool", "penetracion_v2x": "float", "split": "str"},
    "incidentes.parquet": {"inc_id": "str", "dia_id": "int", "edge": "str", "corredor": "str", "celda_id": "int",
                           "t_ini": "float", "t_fin": "float", "severidad": "str"},
    "celdas.parquet": {"celda": "int", "v_ff": "float", "fila": "int", "col": "int", "corredor": "str"},
    "aristas.parquet": {"id": "str", "lanes": "int", "speed": "float", "celda": "int", "corredor": "str"},
    "celdas_5min": {"dia_id": "int", "t": "int", "celda_id": "int", "gt_n_veh": "float", "gt_vel": "float", "gt_densidad": "float",
                    "gt_flujo": "float", "gt_ocupacion": "float", "obs_n_veh": "float", "obs_vel": "float", "v_ff": "float"},
    "corredor_1min": {"dia_id": "int", "t": "int", "corredor": "str", "edge": "str", "celda_id": "int", "carriles": "int",
                      "v_ff": "float", "gt_n": "int", "gt_flujo": "float", "gt_ocupacion": "float", "gt_vel": "float",
                      "obs_n": "int", "obs_vel": "float", "obs_ocupacion": "float"},
}
RANGOS = {"gt_vel": (0, 150), "obs_vel": (0, 150), "gt_ocupacion": (0, 100), "obs_ocupacion": (0, 100), "v_ff": (5, 150)}
PASO = {"celdas_5min": 300, "corredor_1min": 60}


def tipo_ok(serie, t):
    k = serie.dtype.kind
    return {"int": k in "iu", "float": k in "fiu", "bool": k == "b", "str": k in "OUT" or str(serie.dtype).startswith("str")}[t]


def main():
    ds = Path(sys.argv[1] if len(sys.argv) > 1 else os.environ.get("RVH_DATOS", RAIZ / "dataset"))
    errores = []
    for nombre, cols in ESQUEMA.items():
        p = ds / nombre
        if not p.exists():
            errores.append(f"falta {nombre}"); continue
        archivos = sorted(p.glob("*.parquet")) if p.is_dir() else [p]
        if not archivos:
            errores.append(f"{nombre}: carpeta vacia"); continue
        df = pd.read_parquet(archivos[0])   # basta un archivo por tabla para revisar el formato
        for c, t in cols.items():
            if c not in df.columns:
                errores.append(f"{nombre}: falta la columna {c}")
            elif not tipo_ok(df[c], t):
                errores.append(f"{nombre}: {c} deberia ser {t} y es {df[c].dtype}")
        for c, (lo, hi) in RANGOS.items():
            if c in df.columns:
                v = df[c].dropna()
                if len(v) and (v.min() < lo or v.max() > hi):
                    errores.append(f"{nombre}: {c} fuera de rango [{lo}, {hi}] ({v.min():.1f} a {v.max():.1f})")
        if nombre in PASO and "t" in df.columns and (df.t % PASO[nombre]).any():
            errores.append(f"{nombre}: t debe ser multiplo de {PASO[nombre]} s (inicio del intervalo)")
        print(f"{'ok ' if not any(e.startswith(nombre) for e in errores) else 'ERR'} {nombre:20s} {len(archivos)} archivo(s), {len(df.columns)} columnas")
    if errores:
        print("\nProblemas:\n  " + "\n  ".join(errores)); sys.exit(1)
    print(f"\nFormato correcto: el proyecto puede usar {ds} con RVH_DATOS={ds}")


if __name__ == "__main__":
    main()
