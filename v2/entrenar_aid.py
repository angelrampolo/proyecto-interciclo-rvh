"""Deteccion automatica de incidentes (AID) en los corredores C1, C2 y C3, minuto a minuto.

Lo que separa un incidente de la congestion normal de hora pico (y que el avance no usaba):
  1. desvio respecto a lo habitual en esa arista, a esa hora y en ese tipo de dia (promedio de train);
  2. la firma espacial del incidente: cola aguas arriba y via vacia aguas abajo (tipo algoritmo California);
  3. la tendencia de los ultimos 1, 3 y 5 min y la persistencia de la alarma.

Evaluacion por incidente (DR, MTTD) y por minuto (FAR), con intervalos de confianza por bootstrap.
Uso:  python entrenar_aid.py --modo gt | obs
"""
import argparse, json, time
from collections import defaultdict
import numpy as np, pandas as pd, joblib
from sklearn.ensemble import HistGradientBoostingClassifier
import comun as C

RASGOS = ["vel_rel", "ocup", "flujo", "sin_autos",
          "dv1", "dv3", "dv5", "do1", "do3", "do5", "vel_media5", "ocup_media5",
          "dev_vel", "dev_ocup", "dev_flujo",
          "arriba_vel", "abajo_vel", "arriba_ocup", "abajo_ocup",
          "dif_ocup_arriba_abajo", "dif_vel_abajo_arriba", "hora_sin", "hora_cos",
          "z_vel", "z_ocup", "z_vel_sostenido", "z_ocup_sostenido", "z_vel_arriba", "z_ocup_arriba",
          "z_vel_red", "z_ocup_red", "z_vel_local", "z_ocup_local", "lluvia",
          "z_vel_60", "z_ocup_60", "novedad_vel", "novedad_ocup", "caida10"]
FAR_META = 3.0   # % ; la documentacion del curso pide FAR <= 3-5 %
GRACIA = 15   # min tras el fin del incidente en que la cola aun se disipa: ni positivo ni falsa alarma


def vecinos_corredor():
    red = {e["id"]: e for e in json.load(open(C.RED))["edges"]}
    ari = pd.read_parquet(C.DS / "aristas.parquet")
    ari = ari[ari.corredor.fillna("ninguno") != "ninguno"]
    por_desde = defaultdict(list); por_hasta = defaultdict(list)
    for e in ari.id:
        if e in red:
            por_desde[(red[e]["from"])].append(e); por_hasta[(red[e]["to"])].append(e)
    arriba = {e: [u for u in por_hasta[red[e]["from"]] if red[u]["from"] != red[e]["to"]] for e in ari.id if e in red}
    abajo = {e: [d for d in por_desde[red[e]["to"]] if red[d]["to"] != red[e]["from"]] for e in ari.id if e in red}
    return arriba, abajo


ARRIBA, ABAJO = vecinos_corredor()


def tramo(edge):
    """La arista del incidente, dos aristas aguas arriba y una aguas abajo (mismo sentido)."""
    out, fr = [edge], [edge]
    for _ in range(2):
        fr = [u for e in fr for u in ARRIBA.get(e, [])]; out += fr
    return out + ABAJO.get(edge, [])[:1]


def impacto_incidentes():
    """Impacto observable de cada incidente: la mayor anomalia sostenida (5 min) de su tramo durante el
    incidente, en desviaciones habituales: max(-z_vel, z_ocup). Se considera observable si supera el
    percentil 99 de esa misma anomalia en minutos SIN incidente de los dias de train (el ruido normal de
    un tramo con semaforos). Se mide con gt; solo sirve para etiquetar y desglosar, nunca es entrada."""
    cache = C.SALIDA / "impacto_incidentes.json"
    if cache.exists():
        return json.load(open(cache))
    HA = promedio_historico("gt")
    inc = C.incidentes()
    inc = inc[inc.corredor != "ninguno"]
    fondo, valor = [], {}
    for d in sorted(inc.dia_id.unique()):
        X = rasgos_desde_dia_gt(d, HA)
        a = np.maximum(-X["z_vel_sostenido"], X["z_ocup_sostenido"])
        if d in C.ids("train"):
            y = matrices_dia(d, "gt")["y"]
            fondo.append(a[y == 0][::11])
        for _, r in inc[inc.dia_id == d].iterrows():
            t0, t1 = int(r.t_ini // 60), int(np.ceil(r.t_fin / 60))
            filas = [X["ie"][e] for e in tramo(r.edge) if e in X["ie"]]
            valor[r.inc_id] = round(float(a[filas, t0:t1 + 1].max()), 3) if filas else 0.0
    f = np.concatenate(fondo); f = f[np.isfinite(f)]
    umbral = float(np.percentile(f, 99))
    out = {"umbral_p99_fondo_train": round(umbral, 3), "valor": valor, "impacto": {k: v >= umbral for k, v in valor.items()}}
    json.dump(out, open(cache, "w"), indent=1)
    return out


def rasgos_desde_dia_gt(d, HA):
    m = matrices_dia(d, "gt")
    X = rasgos_desde(m, C.dias().loc[d, "tipo_dia"], HA)
    return {"ie": {e: i for i, e in enumerate(m["edges"])},
            "z_vel_sostenido": X[..., RASGOS.index("z_vel_sostenido")], "z_ocup_sostenido": X[..., RASGOS.index("z_ocup_sostenido")]}


IMPACTO = {}


def matrices_dia(d, modo):
    df = pd.read_parquet(C.DS / "corredor_1min" / f"dia_{d:02d}.parquet")
    p = "gt" if modo == "gt" else "obs"
    pen = 1.0 if modo == "gt" else float(C.dias().loc[d, "penetracion_v2x"])
    edges = sorted(df.edge.unique())
    ie = {e: i for i, e in enumerate(edges)}
    E, T = len(edges), 1440
    ei = df.edge.map(ie).to_numpy(); ti = (df.t.to_numpy() // 60).astype(int)
    def mat(col, rel=None, fill=0.0, div=1.0):
        a = np.full((E, T), np.nan, np.float32)
        v = df[col].to_numpy(np.float64) / div
        if rel is not None:
            v = v / df[rel].to_numpy()
        a[ei, ti] = v
        return a
    n = mat(f"{p}_n")
    vel = mat(f"{p}_vel", rel="v_ff")
    sin = (np.nan_to_num(n) == 0)
    vel = np.where(sin, np.nan, vel)
    # sin autos en un minuto: se arrastra la ultima velocidad medida (no se inventa flujo libre: puede ser un cierre)
    vel = pd.DataFrame(vel.T).ffill().fillna(1.0).to_numpy().T.astype(np.float32)
    ocup = np.nan_to_num(mat(f"{p}_ocupacion", div=1.0 if modo == "gt" else pen))
    flujo = np.nan_to_num(mat(f"{p}_flujo", div=pen))
    y_celda = np.zeros((E, T), np.int8); y_celda[ei, ti] = df.incidente_en_celda.to_numpy()
    if d >= 100:
        y_celda[:, :5 * 60] = 2; y_celda[:, int(21.5 * 60):] = 2
    celda = df.groupby("edge").celda_id.first().reindex(edges).to_numpy()
    return dict(edges=edges, vel=vel, ocup=ocup, flujo=flujo, sin=sin.astype(np.float32), y=y_celda, celda=celda,
                lluvia=float(C.dias().loc[d, "lluvia"]))


def promedio_historico(modo):
    """Perfil habitual por arista, tipo de dia y bloque de 15 min (solo train): media y desviacion.
    Ademas un perfil 'laborable_lluvia' con los laborables lluviosos de train y aid_train: la lluvia se conoce
    en tiempo real (servicio meteorologico) y en SUMO frena toda la red, no es un incidente."""
    cache = C.SALIDA / f"perfil_habitual_{modo}.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        return z["perfil"].item()
    tabla = C.dias()
    tr = C.ids("train")
    lluviosos = [d for d in tr + C.ids("aid_train") if tabla.loc[d, "lluvia"] and tabla.loc[d, "tipo_dia"] == "laborable"]
    acum = {}
    for d in sorted(set(tr) | set(lluviosos)):
        tipos = ([tabla.loc[d, "tipo_dia"]] if d in tr else []) + (["laborable_lluvia"] if d in lluviosos else [])
        m = matrices_dia(d, modo)
        for k in ("vel", "ocup", "flujo"):
            v = m[k].reshape(len(m["edges"]), 96, 15)
            for i, e in enumerate(m["edges"]):
                for tipo in tipos:
                    s_ = acum.setdefault((tipo, e, k), [np.zeros(96), np.zeros(96), 0])
                    s_[0] += v[i].mean(-1); s_[1] += (v[i] ** 2).mean(-1); s_[2] += 1
    out = {}
    for key, (s1, s2, n) in acum.items():
        mu = s1 / n
        out[key] = mu.astype(np.float32)
        out[key + ("sd",)] = np.sqrt(np.maximum(s2 / n - mu ** 2, 0)).astype(np.float32)
    np.savez(cache, perfil=np.array(out, dtype=object))
    return out


def rasgos_dia(d, modo, HA):
    m = matrices_dia(d, modo)
    return rasgos_desde(m, C.dias().loc[d, "tipo_dia"], HA), m


def rasgos_desde(m, tipo, HA):
    """Rasgos causales (solo usan el pasado) a partir de matrices [aristas, 1440]; lo usa tambien el bucle SIL."""
    E, T = m["vel"].shape
    ie = {e: i for i, e in enumerate(m["edges"])}
    vel, ocup, flujo = m["vel"], m["ocup"], m["flujo"]
    def dif(a, k):
        out = np.zeros_like(a); out[:, k:] = a[:, k:] - a[:, :-k]; return out
    def media5(a):
        c = np.cumsum(np.pad(a, ((0, 0), (5, 0))), 1); return (c[:, 5:] - c[:, :-5]) / 5
    def media(a, n):
        c = np.cumsum(np.pad(a, ((0, 0), (n, 0))), 1); return (c[:, n:] - c[:, :-n]) / n
    def atras(a, k):
        out = np.zeros_like(a); out[:, k:] = a[:, :-k]; return out
    def vec(lista, a):
        out = a.copy()
        for e, i in ie.items():
            js = [ie[u] for u in lista.get(e, []) if u in ie]
            if js:
                out[i] = a[js].mean(0)
        return out
    def ha(k, sd=False):
        out = np.full((E, T), 1.0 if sd else 0.0, np.float32)
        for e, i in ie.items():
            h = HA.get((tipo, e, k, "sd") if sd else (tipo, e, k))
            if h is not None:
                out[i] = np.repeat(h, 15)
        return out
    def maxmin(a, k, f):
        out = a.copy()
        for j in range(1, k):
            out[:, j:] = f(out[:, j:], a[:, :-j])
        return out
    v5, o5 = media5(vel), media5(ocup)
    av, bv = vec(ARRIBA, v5), vec(ABAJO, v5)
    ao, bo = vec(ARRIBA, o5), vec(ABAJO, o5)
    hora = np.arange(T) / T * 2 * np.pi
    zv = (v5 - ha("vel")) / (ha("vel", True) + 0.05)
    zo = (o5 - ha("ocup")) / (ha("ocup", True) + 1.0)
    # contexto de todos los corredores en el mismo minuto (lluvia, dia cargado): un incidente es local
    zvr = np.broadcast_to(np.median(zv, 0), (E, T)); zor = np.broadcast_to(np.median(zo, 0), (E, T))
    F = {"vel_rel": vel, "ocup": ocup, "flujo": flujo, "sin_autos": m["sin"],
         "dv1": dif(vel, 1), "dv3": dif(vel, 3), "dv5": dif(vel, 5),
         "do1": dif(ocup, 1), "do3": dif(ocup, 3), "do5": dif(ocup, 5),
         "vel_media5": v5, "ocup_media5": o5,
         "dev_vel": v5 - ha("vel"), "dev_ocup": o5 - ha("ocup"), "dev_flujo": media5(flujo) - ha("flujo"),
         "arriba_vel": av - v5, "abajo_vel": bv - v5, "arriba_ocup": ao - o5, "abajo_ocup": bo - o5,
         "dif_ocup_arriba_abajo": ao - bo, "dif_vel_abajo_arriba": bv - av,
         "hora_sin": np.broadcast_to(np.sin(hora), (E, T)), "hora_cos": np.broadcast_to(np.cos(hora), (E, T)),
         "z_vel": zv, "z_ocup": zo, "z_vel_sostenido": maxmin(zv, 4, np.maximum), "z_ocup_sostenido": maxmin(zo, 4, np.minimum),
         "z_vel_arriba": vec(ARRIBA, zv), "z_ocup_arriba": vec(ARRIBA, zo),
         "z_vel_red": zvr, "z_ocup_red": zor, "z_vel_local": zv - zvr, "z_ocup_local": zo - zor,
         "lluvia": np.full((E, T), float(m.get("lluvia", 0.0)), np.float32),
         # novedad: un incidente es un cambio brusco; un tramo que va lento toda la hora (dia cargado) no lo es
         "z_vel_60": media(zv, 60), "z_ocup_60": media(zo, 60),
         "novedad_vel": zv - atras(media(zv, 30), 5), "novedad_ocup": zo - atras(media(zo, 30), 5),
         "caida10": v5 - atras(v5, 10)}
    return np.stack([F[k] for k in RASGOS], -1).astype(np.float32)   # [E, T, F]


def etiqueta_limpia(d, m):
    """1 = tramo de un incidente con impacto, mientras dura; -1 = ignorar (resto de la celda o incidente sin firma)."""
    y = m["y"].astype(np.int8).copy()
    fuera = y == 2
    ignorar = (y == 1) | fuera
    y[:] = 0
    ie = {e: i for i, e in enumerate(m["edges"])}
    inc = C.incidentes()
    for _, r in inc[(inc.dia_id == d) & (inc.corredor != "ninguno")].iterrows():
        t0, t1 = int(r.t_ini // 60), int(np.ceil(r.t_fin / 60))
        filas = [ie[e] for e in tramo(r.edge) if e in ie]
        ignorar[filas, t1 + 1:t1 + 1 + GRACIA] = True
        if IMPACTO.get(r.inc_id):
            y[filas, t0 + 1:t1 + 1] = 1
        else:
            ignorar[filas, t0:t1 + 1] = True
    y[ignorar & (y == 0)] = -1
    return y


def tabla_train(modo, HA, rng, lista=None):
    Xs, ys = [], []
    for d in (lista if lista is not None else C.ids("train") + C.ids("aid_train")):
        X, m = rasgos_dia(d, modo, HA)
        y = etiqueta_limpia(d, m)
        dificil = (X[..., 0] < 0.7) | (X[..., RASGOS.index("ocup")] > 15)    # congestion: los negativos dificiles
        u = rng.random(y.shape)
        sel = (y == 1) | ((y == 0) & ((dificil & (u < 0.5)) | (u < 0.03)))       # mas negativos de congestion normal
        Xs.append(X[sel]); ys.append(y[sel])
    return np.concatenate(Xs), np.concatenate(ys)


def alarmas(prob, umbral, pers):
    a = prob >= umbral
    if pers > 1:
        c = np.cumsum(np.pad(a.astype(np.int16), ((0, 0), (pers, 0))), 1)
        a = (c[:, pers:] - c[:, :-pers]) == pers
    return a


def evaluar(dias_ids, probs, mats, umbral, pers, detalle=False):
    inc = C.incidentes()
    inc = inc[inc.dia_id.isin(dias_ids) & (inc.corredor != "ninguno")]
    fa = neg = 0
    det, tiempos, filas = 0, [], []
    lluv = C.dias().lluvia
    for d in dias_ids:
        a = alarmas(probs[d], umbral_de(umbral, lluv.loc[d]), pers)
        y = mats[d]["y"]
        a = a & (y != 2)
        y = y.copy()
        ie = {e: i for i, e in enumerate(mats[d]["edges"])}
        for _, r in inc[inc.dia_id == d].iterrows():
            t1 = int(np.ceil(r.t_fin / 60))
            filas_g = [ie[e] for e in tramo(r.edge) if e in ie]
            y[filas_g, t1 + 1:t1 + 1 + GRACIA] = np.where(y[filas_g, t1 + 1:t1 + 1 + GRACIA] == 0, 3, y[filas_g, t1 + 1:t1 + 1 + GRACIA])
        fa += int((a & (y == 0)).sum()); neg += int((y == 0).sum())
        for _, r in inc[inc.dia_id == d].iterrows():
            filas_c = np.where(mats[d]["celda"] == r.celda_id)[0]
            t0, t1 = int(r.t_ini // 60), int(np.ceil(r.t_fin / 60))
            hit = np.where(a[filas_c][:, t0:t1 + 1].any(0))[0] if len(filas_c) else []
            ok = len(hit) > 0
            tt = float(hit[0] + t0 - r.t_ini / 60) if ok else None
            det += ok
            if ok: tiempos.append(max(tt, 0.0))
            filas.append({"impacto": IMPACTO.get(r.inc_id), "inc_id": r.inc_id, "dia": d, "corredor": r.corredor, "via": r.nombre_via, "severidad": r.severidad,
                          "duracion_min": round(r.duracion / 60, 1), "detectado": bool(ok), "ttd_min": None if tt is None else round(max(tt, 0), 1)})
    n = len(inc)
    con = [f for f in filas if IMPACTO.get(f["inc_id"])]
    res = {"incidentes_con_impacto": len(con), "DR_con_impacto": round(100 * sum(f["detectado"] for f in con) / max(len(con), 1), 2),
           "MTTD_con_impacto_min": round(float(np.mean([f["ttd_min"] for f in con if f["detectado"]])), 2) if any(f["detectado"] for f in con) else None,
           "incidentes": n, "detectados": det, "DR": round(100 * det / max(n, 1), 2),
           "FAR": round(100 * fa / max(neg, 1), 4), "MTTD_min": round(float(np.mean(tiempos)), 2) if tiempos else None,
           "falsas_alarmas_por_dia": round(fa / len(dias_ids), 1)}
    if detalle:
        res["por_incidente"] = filas
    return res


def umbral_de(umbral, lluvia):
    """Umbral 'modo lluvia': con lluvia (dato meteorologico en tiempo real) se usa otro umbral elegido sin test."""
    if isinstance(umbral, dict):
        return umbral["lluvia" if lluvia else "seco"]
    return umbral


def bootstrap(filas, B=2000, rng=None):
    rng = rng or np.random.default_rng(0)
    det = np.array([f["detectado"] for f in filas]); tt = np.array([f["ttd_min"] if f["ttd_min"] is not None else np.nan for f in filas])
    drs, mt = [], []
    for _ in range(B):
        s = rng.integers(0, len(filas), len(filas))
        drs.append(det[s].mean() * 100); v = tt[s][det[s]]
        if len(v): mt.append(v.mean())
    return {"DR_IC95": [round(float(np.percentile(drs, 2.5)), 1), round(float(np.percentile(drs, 97.5)), 1)],
            "MTTD_IC95": [round(float(np.percentile(mt, 2.5)), 2), round(float(np.percentile(mt, 97.5)), 2)]}


def predecir_dias(clf, lista, modo, HA):
    probs, mats = {}, {}
    for d in lista:
        X, m = rasgos_dia(d, modo, HA)
        E, T, F = X.shape
        probs[d] = clf.predict_proba(X.reshape(-1, F))[:, 1].reshape(E, T).astype(np.float32)
        m.pop("vel"); m.pop("ocup"); m.pop("flujo"); m.pop("sin")
        mats[d] = m
    return probs, mats


def cargar_impacto():
    IMPACTO.update(impacto_incidentes()["impacto"])


def main():
    cargar_impacto()
    ap = argparse.ArgumentParser(); ap.add_argument("--modo", choices=["gt", "obs"], default="gt")
    a = ap.parse_args()
    rng = np.random.default_rng(42); t0 = time.time()
    HA = promedio_historico(a.modo)
    Xtr, ytr = tabla_train(a.modo, HA, rng)
    print(f"[{a.modo}] train {len(ytr):,} filas, {ytr.sum():,} positivas ({time.time()-t0:.0f}s)", flush=True)
    def modelo():
        return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, max_leaf_nodes=48, min_samples_leaf=40,
                                              l2_regularization=1.0, class_weight={0: 1, 1: 20}, random_state=42)
    clf = modelo(); clf.fit(Xtr, ytr); del Xtr
    # Seleccion del umbral sin mirar test: predicciones fuera de pliegue (4 pliegues por dia) en train+aid_train
    # mas validacion. Con lluvia hay muchas mas colas normales, asi que se elige un umbral aparte (modo lluvia).
    dias_tr = C.ids("train") + C.ids("aid_train")
    pliegues = [dias_tr[k::4] for k in range(4)]
    pool_p, pool_m = {}, {}
    for k, pl in enumerate(pliegues):
        resto = [d for d in dias_tr if d not in pl]
        Xk, yk = tabla_train(a.modo, HA, np.random.default_rng(100 + k), resto)
        mk = modelo(); mk.fit(Xk, yk); del Xk
        pk, mmk = predecir_dias(mk, pl, a.modo, HA); pool_p.update(pk); pool_m.update(mmk)
        print(f"  pliegue {k + 1}/4 listo ({time.time()-t0:.0f}s)", flush=True)
    dias_val = C.ids("val") + C.ids("aid_val")
    pv, mv = predecir_dias(clf, dias_val, a.modo, HA); pool_p.update(pv); pool_m.update(mv)
    lluv = C.dias().lluvia
    pool = sorted(pool_p)
    grupos = {"seco": [d for d in pool if not lluv.loc[d]], "lluvia": [d for d in pool if lluv.loc[d]]}
    curva, tabla_c = [], {}
    for pers in (2, 3):
        for g, dd in grupos.items():
            for u in np.round(np.arange(0.04, 0.991, 0.02), 2):
                r = evaluar(dd, pool_p, pool_m, u, pers)
                # la FAR se exige por separado en dias normales y en dias con muchos incidentes (como el test de incidentes)
                sub = [evaluar(x, pool_p, pool_m, u, pers)["FAR"] for x in ([d for d in dd if d < 100], [d for d in dd if d >= 100]) if x]
                r["FAR_peor_tipo_dia"] = max(sub)
                curva.append({"umbral": float(u), "persistencia": pers, "clima": g, **r}); tabla_c[(pers, g, float(u))] = r

    def seleccionar(lim):
        """Regla (sin test): FAR <= lim en cada clima y en cada tipo de dia (normal / con incidentes), maxima DR en
        incidentes con impacto; desempate DR y MTTD."""
        mejor = None
        for pers in (2, 3):
            elegido = {}
            for g in grupos:
                best = None
                for (pp, gg, u), r in tabla_c.items():
                    if pp == pers and gg == g and r["FAR_peor_tipo_dia"] <= lim and (best is None or (r["DR_con_impacto"], r["DR"], -(r["MTTD_min"] or 99)) > (best[1]["DR_con_impacto"], best[1]["DR"], -(best[1]["MTTD_min"] or 99))):
                        best = (u, r)
                elegido[g] = best[0]
            r = evaluar(pool, pool_p, pool_m, elegido, pers)
            cand = {"umbral": elegido["seco"], "umbral_lluvia": elegido["lluvia"], "persistencia": pers, "FAR_maxima": lim, **r}
            if mejor is None or (r["DR_con_impacto"], -r["FAR"]) > (mejor["DR_con_impacto"], -mejor["FAR"]):
                mejor = cand
        mejor["dias_seleccion"] = {g: len(v) for g, v in grupos.items()}
        return mejor
    # punto de operacion: meta del curso (FAR <= 3-5 %), se toma el extremo exigente 3 %; ademas se reporta FAR <= 1 %
    mejor, estricto = seleccionar(FAR_META), seleccionar(1.0)
    print("SELECCION (fuera de pliegue + val):", mejor, flush=True)
    print("SELECCION estricta FAR<=1:", estricto, flush=True)
    U = {"seco": mejor["umbral"], "lluvia": mejor["umbral_lluvia"]}
    U1 = {"seco": estricto["umbral"], "lluvia": estricto["umbral_lluvia"]}
    del pool_p, pool_m, pv, mv
    salida = {"modo": a.modo, "rasgos": RASGOS, "seleccion_en_val": mejor, "curva_val": curva,
              "dias_entrenamiento": len(C.ids("train") + C.ids("aid_train"))}
    probs_dash = {}
    for nombre, dias_eval in (("test", C.ids("test")), ("test_incidentes", C.ids("aid_test"))):
        if not dias_eval:
            continue
        pt, mt = predecir_dias(clf, dias_eval, a.modo, HA)
        test = evaluar(dias_eval, pt, mt, U, mejor["persistencia"], detalle=True)
        test.update(bootstrap(test["por_incidente"]))
        lluv = C.dias().lluvia
        test["por_clima"] = {g: evaluar(dd, pt, mt, U, mejor["persistencia"]) for g, dd in
                             (("seco", [d for d in dias_eval if not lluv.loc[d]]), ("lluvia", [d for d in dias_eval if lluv.loc[d]])) if dd}
        # linea base: detector al azar con la misma tasa de alarmas (cuanto de la DR es suerte)
        rng2 = np.random.default_rng(7)
        azar_p = {d: rng2.random(mt[d]["y"].shape).astype(np.float32) for d in mt}
        objetivo = test["FAR"]
        mejor_az = min((abs(evaluar(dias_eval, azar_p, mt, u, mejor["persistencia"])["FAR"] - objetivo), u) for u in np.arange(0.80, 0.9995, 0.002))[1]
        test["azar_misma_FAR"] = evaluar(dias_eval, azar_p, mt, mejor_az, mejor["persistencia"])
        salida[nombre] = test
        e1 = evaluar(dias_eval, pt, mt, U1, estricto["persistencia"])
        salida.setdefault("estricto_far1", {"seleccion": estricto})[nombre] = e1
        salida["curva_" + nombre] = [{"umbral": float(u), **evaluar(dias_eval, pt, mt, u, mejor["persistencia"])} for u in np.round(np.arange(0.06, 0.991, 0.04), 2)]
        print(nombre.upper() + ":", {k: v for k, v in test.items() if k != "por_incidente"}, flush=True)
        if nombre == "test":
            probs_dash.update({f"p_{d}": (pt[d] * 255).astype(np.uint8) for d in pt}); probs_dash.update({f"edges_{d}": np.array(mt[d]["edges"]) for d in mt})
        del pt, mt, azar_p
    ps, ms = predecir_dias(clf, C.ids("test_estres"), a.modo, HA)
    estres = {f"dia_{d}": evaluar([d], ps, ms, U, mejor["persistencia"]) for d in C.ids("test_estres")}
    estres["todos"] = evaluar(C.ids("test_estres"), ps, ms, U, mejor["persistencia"], detalle=True)
    salida["estres"] = estres
    print("ESTRES:", {k: {kk: vv for kk, vv in v.items() if kk != "por_incidente"} for k, v in estres.items()}, flush=True)
    joblib.dump({"modelo": clf, "umbral": mejor["umbral"], "umbral_lluvia": mejor["umbral_lluvia"], "persistencia": mejor["persistencia"], "rasgos": RASGOS, "HA": HA, "modo": a.modo},
                C.SALIDA / f"modelo_aid_{a.modo}.joblib", compress=3)
    np.savez_compressed(C.SALIDA / f"aid_test_{a.modo}.npz", **probs_dash)
    salida["seg"] = round(time.time() - t0)
    json.dump(salida, open(C.SALIDA / f"metricas_aid_{a.modo}.json", "w"), indent=2, ensure_ascii=False, default=float)

if __name__ == "__main__":
    main()
