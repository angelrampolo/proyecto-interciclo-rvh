"""Dias extra centrados en incidentes para entrenar y evaluar el AID (dias 101-120).

Motivo: en el dataset original 26 de 39 incidentes de val+test no cambian el trafico mas que el ruido
normal (bloquean 1 de 2-3 carriles con demanda moderada), asi que no hay nada que detectar.
Aqui se usa EXACTAMENTE el mismo escenario (red, demanda, tipos de auto, detectores, SUMO 1.12 sin
rerouting); solo cambia el plan de incidentes:
  - 8 incidentes por dia, todos en los corredores C1, C2 y C3, entre 06:30 y 20:30, separados 40 min
    dentro de cada corredor;
  - severidad: 45 % bloquea 1 carril, 40 % bloquea 2 (cierre total si la via tiene 2), 15 % cierre total;
    los cierres totales duran 10-15 min (sin rerouting, uno largo en hora pico traba la ciudad por horas);
  - se simula de 04:00 a 21:30 (la red se llena sola antes de las 06:00) y solo se guardan los detectores
    de corredor cada minuto y los incidentes.
Uso:  python generar_dias_incidentes.py --paralelo 5        (reanudable)
"""
import argparse, gzip, importlib.util, json, shutil, time, traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np, pandas as pd

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent
spec = importlib.util.spec_from_file_location("etapa2_comun", RAIZ / "etapa2" / "comun.py")
E2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(E2)
DS = RAIZ / "dataset"
CAL = E2.CALENTAMIENTO
INI, FIN = 4 * 3600, int(21.5 * 3600)
DIAS = list(range(101, 121))
SPLIT = {**{d: "aid_train" for d in range(101, 113)}, **{d: "aid_val" for d in range(113, 117)}, **{d: "aid_test" for d in range(117, 121)}}


def calendario():
    rng = np.random.default_rng(2027)
    out = []
    for d in DIAS:
        out.append({"dia_id": d, "semana": 0, "dow": int(d % 5), "tipo_dia": "laborable", "factor_demanda": round(float(rng.uniform(0.92, 1.05)), 3),
                    "lluvia": bool(rng.random() < 0.2), "penetracion_v2x": 0.5, "n_incidentes": 8, "split": SPLIT[d], "semilla": 5000 + d,
                    "v0": float((RAIZ / "etapa2/resultados/V0.txt").read_text())})
    return out


def plan(dia_id, rng, uso):
    r = E2.red_info()
    grupos = {}
    for c in E2.CORREDORES:
        es = [e for e in r["edges"] if e["corredor"] == c and e["len"] >= 120 and uso.get(e["id"], 0) > 0]
        u = np.array([uso[e["id"]] for e in es], float); m = u >= np.median(u)
        es = [e for e, k in zip(es, m) if k]; u = u[m]
        grupos[c] = (es, u / u.sum())
    ocupado = {c: [] for c in E2.CORREDORES}
    out, k = [], 0
    while len(out) < 8 and k < 200:
        k += 1
        c = list(E2.CORREDORES)[rng.integers(3)]
        t = int(rng.uniform(6.5 * 3600 if FIN > 10 * 3600 else 5.2 * 3600, min(20.5 * 3600, FIN - 1500)))
        if any(abs(t - x) < 2400 for x in ocupado[c]):
            continue
        es, p = grupos[c]; e = es[rng.choice(len(es), p=p)]
        L = e["lanes"]; x = rng.random()
        if L == 1 or x >= 0.85:
            carriles, sev, dur = list(range(L)), "total", int(rng.uniform(600, 900))
        elif x < 0.45:
            carriles, sev, dur = [int(rng.integers(L))], "parcial", int(rng.uniform(600, 1500))
        elif L == 2:
            carriles, sev, dur = [0, 1], "total", int(rng.uniform(600, 900))
        else:
            l0 = int(rng.integers(L - 1)); carriles, sev, dur = [l0, l0 + 1], "parcial", int(rng.uniform(600, 1500))
        ocupado[c].append(t)
        out.append({"inc_id": f"inc{dia_id}_{len(out)}", "dia_id": dia_id, "edge": e["id"], "nombre_via": e["name"], "corredor": c,
                    "celda_id": e["celda"], "uso_rutas": int(uso[e["id"]]), "pos": float(rng.uniform(0.35, 0.65) * e["len"]),
                    "carriles": carriles, "n_carriles_via": L, "severidad": sev, "t_ini_plan": t, "duracion": dur})
    return out


def correr_dia(dia):
    i = dia["dia_id"]
    trabajo = RAIZ / "trabajo" / f"aid_{i}"
    shutil.rmtree(trabajo, ignore_errors=True); trabajo.mkdir(parents=True)
    log = trabajo / "log.txt"; t0 = time.time()
    for intento in range(3):
        rng = np.random.default_rng(dia["semilla"])
        factor = dia["factor_demanda"] * (1.04 if dia["lluvia"] else 1.0) * 0.95 ** intento
        tasas = E2._tasas_por_segundo(E2.PERFILES["laborable"], dia["v0"], factor, FIN + CAL)
        tasas[:INI + CAL] = 0                                  # nadie sale antes de las 04:00
        n = E2.generar_viajes(trabajo / "viajes.trips.xml", tasas, dia["semilla"])
        E2.escribir_vtypes(trabajo / "vtypes.add.xml", 0.88 if dia["lluvia"] else 1.0)
        E2.escribir_salidas(trabajo / "salidas.add.xml", con_corredores=True)
        s = "".join(l for l in open(trabajo / "salidas.add.xml") if "edgeData" not in l)
        open(trabajo / "salidas.add.xml", "w").write(s)
        E2.duarouter(trabajo, dia["semilla"], log)
        (trabajo / "viajes.trips.xml").unlink()
        incs = plan(i, rng, E2.uso_rutas(trabajo / "rutas.rou.xml.gz"))
        E2.escribir_incidentes(trabajo / "incidentes.rou.xml", incs)
        try:
            E2.correr_sumo(trabajo, dia["semilla"], FIN + CAL, con_tripinfo=False, extra_rutas=["/t/incidentes.rou.xml"],
                           log_file=log, limite_en_red=10000)
            break
        except E2.Colapso as e:
            E2.log(f"dia {i}: {e}; se repite con 5 % menos de demanda")
            if intento == 2: raise
            for f in trabajo.iterdir():
                if f.name != "log.txt": f.unlink()
    (trabajo / "rutas.rou.xml.gz").unlink(missing_ok=True)
    st = E2.leer_stops(trabajo / "stops.xml")
    for inc in incs:
        s_ = st[st.veh.str.startswith(inc["inc_id"] + "_L")]
        inc["t_ini"] = float(s_.inicio.min() - CAL) if len(s_) else np.nan
        inc["t_fin"] = float(s_.fin.max() - CAL) if len(s_) else np.nan
        inc["carriles_bloqueados"] = len(inc["carriles"]); inc["carriles"] = ",".join(map(str, inc["carriles"]))
    ar = E2.tabla_aristas().set_index("id")
    cor = None
    for vista, f in (("gt", "e1_gt.xml.gz"), ("obs", "e1_v2x.xml.gz")):
        e1 = E2.leer_e1(trabajo / f); (trabajo / f).unlink()
        e1["t"] = e1.t_sim - CAL
        e1["inv"] = np.where(e1.n > 0, e1.n / e1.vel_arm.clip(lower=0.1), 0)
        g = e1.groupby(["t", "edge"]).agg(n=("n", "sum"), inv=("inv", "sum"), flujo=("flujo", "mean"), ocupacion=("ocupacion", "mean")).reset_index()
        g["vel"] = np.where(g.n > 0, g.n / g.inv.replace(0, np.nan) * 3.6, np.nan)
        g["densidad"] = np.where(g.vel > 0, g.flujo / g.vel, 0.0)
        g = g.drop(columns=["inv"]).rename(columns={c: f"{vista}_{c}" for c in ("n", "flujo", "ocupacion", "vel", "densidad")})
        cor = g if cor is None else cor.merge(g, on=["t", "edge"], how="outer")
    cor = cor.join(ar[["corredor", "celda", "lanes", "speed"]], on="edge")
    cor["v_ff"] = cor.pop("speed") * 3.6
    cor = cor.rename(columns={"lanes": "carriles"})
    # el resto del dia (sin simular) queda vacio: el AID lo ve como "sin autos", igual que de madrugada
    cor["gt_critico"] = ((cor.gt_vel <= 0.55 * cor.v_ff) & (cor.gt_densidad >= 28)).fillna(False)
    cor["incidente_en_celda"] = False; cor["incidente_en_arista"] = False; cor["inc_id"] = ""
    for inc in incs:
        if np.isnan(inc["t_ini"]): continue
        m = (cor.celda == inc["celda_id"]) & (cor.t < inc["t_fin"]) & (cor.t + 60 > inc["t_ini"])
        cor.loc[m, "incidente_en_celda"] = True; cor.loc[m, "inc_id"] = inc["inc_id"]
        m2 = (cor.edge == inc["edge"]) & (cor.t < inc["t_fin"]) & (cor.t + 60 > inc["t_ini"])
        cor.loc[m2, "incidente_en_arista"] = True
    cor = cor[(cor.t >= 0) & (cor.t < 86400)]
    cor.insert(0, "dia_id", i)
    cols = ["dia_id", "t", "corredor", "edge", "celda", "carriles", "v_ff"] + [c for c in cor.columns if c.startswith(("gt_", "obs_"))] + \
           ["incidente_en_arista", "incidente_en_celda", "inc_id"]
    cor = cor[cols].rename(columns={"celda": "celda_id"}).sort_values(["t", "edge"])
    cor["t"] = cor.t.astype("int32")
    for c in cor.columns:
        if cor[c].dtype == "float64": cor[c] = cor[c].astype("float32")
    cor.to_parquet(DS / "corredor_1min" / f"dia_{i}.parquet", index=False, compression="zstd")
    sm = E2.leer_summary(trabajo / "summary.xml.gz")
    res = dict(dia, factor_efectivo=round(factor, 4), reintentos=intento, n_viajes_generados=n,
               max_en_red=int(sm.en_red.max()), teleports=int(sm.teleports.max()), seg_total=round(time.time() - t0))
    json.dump({"dia": res, "incidentes": incs}, open(DS / "dias_aid_json" / f"dia_{i}.json", "w"), default=float, indent=1)
    shutil.rmtree(trabajo)
    return res


def consolidar():
    dias, incs = [], []
    for f in sorted((DS / "dias_aid_json").glob("dia_*.json")):
        j = json.load(open(f)); dias.append(j["dia"]); incs += j["incidentes"]
    pd.DataFrame(dias).to_parquet(DS / "dias_aid.parquet", index=False)
    pd.DataFrame(incs).to_parquet(DS / "incidentes_aid.parquet", index=False)
    return len(dias), len(incs)


def main():
    global FIN
    ap = argparse.ArgumentParser(); ap.add_argument("--paralelo", type=int, default=5); ap.add_argument("--dias", type=int, nargs="*")
    ap.add_argument("--prueba", action="store_true", help="dia corto (hasta las 07:00) para probar el pipeline")
    a = ap.parse_args()
    (DS / "dias_aid_json").mkdir(exist_ok=True)
    if a.prueba:
        FIN = 7 * 3600
        d = calendario()[0]; d["dia_id"] = 199; d["split"] = "prueba"
        r = correr_dia(d); print(r)
        print(json.load(open(DS / "dias_aid_json" / "dia_199.json"))["incidentes"][:2])
        print(pd.read_parquet(DS / "corredor_1min" / "dia_199.parquet").query("t > 21000").head())
        return
    (DS / "dias_aid_json").mkdir(exist_ok=True)
    cal = [d for d in calendario() if not a.dias or d["dia_id"] in a.dias]
    pend = [d for d in cal if not (DS / "dias_aid_json" / f"dia_{d['dia_id']}.json").exists()]
    E2.log(f"dias pendientes {len(pend)}, en paralelo {a.paralelo}")
    with ProcessPoolExecutor(a.paralelo) as ex:
        fut = {ex.submit(correr_dia, d): d for d in pend}
        for f in as_completed(fut):
            d = fut[f]
            try:
                r = f.result()
                E2.log(f"dia {r['dia_id']} ({r['split']}, x{r['factor_demanda']}{' lluvia' if r['lluvia'] else ''}): max en red {r['max_en_red']}, "
                       f"teleports {r['teleports']}, reintentos {r['reintentos']}, {r['seg_total']} s")
            except Exception:
                E2.log(f"ERROR dia {d['dia_id']}:\n{traceback.format_exc()}")
    print("consolidado:", consolidar())


if __name__ == "__main__":
    main()
