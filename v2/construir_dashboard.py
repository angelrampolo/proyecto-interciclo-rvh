"""Arma los datos del dashboard (dashboard/datos.json + binarios) a partir de los resultados reales.
Uso:  python construir_dashboard.py [--dia 37] [--inc inc37_5]
"""
import argparse, json, importlib.util
from pathlib import Path
import numpy as np, pandas as pd
import comun as C
import entrenar_aid as A

OUT = Path(__file__).resolve().parent / "dashboard"
spec = importlib.util.spec_from_file_location("etapa2_comun", C.RAIZ / "etapa2" / "comun.py")
E2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(E2)
TIPO = {"highway.residential": 0, "highway.unclassified": 0, "highway.living_street": 0, "highway.tertiary": 1, "highway.tertiary_link": 1,
        "highway.secondary": 2, "highway.secondary_link": 2, "highway.primary": 3, "highway.primary_link": 3,
        "highway.trunk": 4, "highway.trunk_link": 4, "highway.motorway": 4, "highway.motorway_link": 4}
NOMBRE = {"C1_Americas": "Av. de las Américas", "C2_Circunvalacion_Sur": "Circunvalación Sur", "C3_RemigioCrespo_Solano": "Remigio Crespo – Solano"}


def leer(p, defecto=None):
    p = C.SALIDA / p
    return json.load(open(p)) if p.exists() else defecto


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dia", type=int, default=37); ap.add_argument("--inc", default="inc119_3")
    a = ap.parse_args()
    red = E2.red_info(); b = red["boundary"]; x0, y0 = b[0], b[1]
    rel = lambda pts: [v for p in pts for v in (round(p[0] - x0), round(p[1] - y0))]
    por_id = red["por_id"]
    cel = C.celdas(); N = len(cel)
    corr_code = {None: 0, "ninguno": 0, "C1_Americas": 1, "C2_Circunvalacion_Sur": 2, "C3_RemigioCrespo_Solano": 3}
    nx = red["nx"]
    celdas = [[int(r.col * 200 + 100), int(r.fila * 200 + 100), round(float(r.v_ff), 1), corr_code.get(r.corredor, 0), int(r.celda)] for r in cel.itertuples()]
    calles = {"t": [], "p": []}
    for e in red["edges"]:
        calles["t"].append(TIPO.get(e["type"], 0)); calles["p"].append(rel(e["shape"]))
    ari = pd.read_parquet(C.DS / "aristas.parquet")
    corredores = []
    for cid, nom in NOMBRE.items():
        ids = ari[ari.corredor == cid].id
        eds = [{"id": e, "p": rel(por_id[e]["shape"])} for e in ids if e in por_id]
        mids = np.array([por_id[e]["mid"] for e in ids if e in por_id])
        k = int(np.argsort(mids[:, 0])[len(mids) // 3])
        corredores.append({"id": cid, "nombre": nom, "edges": eds, "etiqueta": [round(mids[k, 0] - x0 + 60), round(mids[k, 1] - y0 + 60)]})
    # ---- datos de cada dia de test (el tablero permite elegir el dia; el elegido con --dia es el inicial)
    tabla = C.dias()
    tipo = tabla.loc[a.dia, "tipo_dia"]
    if not A.IMPACTO:
        A.IMPACTO.update(leer("impacto_incidentes.json", {}).get("impacto", {}))
    inc = pd.read_parquet(C.DS / "incidentes.parquet")
    zp = {f: dict(np.load(C.SALIDA / f"pred_test_{f}.npz")) for f in ("gt", "obs")}   # dict: cargar cada arreglo una sola vez
    za = {f: np.load(C.SALIDA / f"aid_test_{f}.npz") for f in ("gt", "obs")}
    vff = cel.v_ff.to_numpy()
    pos = {c: i for i, c in enumerate(cel.celda)}
    (OUT / "dias").mkdir(exist_ok=True)
    SEMANA = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]
    nombre_dia = lambda d: SEMANA[int(tabla.loc[d, "dow"])] + (", con lluvia" if bool(tabla.loc[d, "lluvia"]) else "")
    geom_aid = {}

    def dia_datos(d):
        """Escribe los binarios del dia en dashboard/dias/ y devuelve su parte del JSON."""
        tp = tabla.loc[d, "tipo_dia"]
        df = pd.read_parquet(C.DS / "celdas_5min" / f"dia_{d:02d}.parquet", columns=["t", "celda_id", "gt_n_veh", "gt_vel", "obs_vel", "obs_n_veh", "v_ff"])
        act = np.zeros((288, N), np.uint8)
        act[(df.t // 300).astype(int), df.celda_id.map(pos)] = (df.gt_n_veh >= 1).to_numpy()
        err = {}
        for f in ("gt", "obs"):
            z = zp[f]
            di = list(z["dias"]).index(d)
            arr = np.zeros((288, N, 7), np.uint8)
            arr[..., 0] = z["Y"][di]
            arr[..., 1] = z["HA"][C.TIPOS.index(tp)]
            ei, eh = [], []
            for w, (dd, t) in enumerate(z["idx"]):
                if dd == di:
                    arr[t, :, 2:6] = z["P"][w]
                    if t + 12 < 288:
                        m = act[t + 12] > 0
                        y = z["Y"][di][t + 12][m].astype(float)
                        ei.append(np.abs(z["P"][w][m, 3] - y)); eh.append(np.abs(z["HA"][C.TIPOS.index(tp)][t + 12][m] - y))
            arr[..., 6] = act
            arr.tofile(OUT / "dias" / f"d{d}_pred_{f}.bin")
            err[f] = {"ia": round(float(np.concatenate(ei).mean()), 2), "ha": round(float(np.concatenate(eh).mean()), 2)} if ei else None
        Y = zp["gt"]["Y"][list(zp["gt"]["dias"]).index(d)]
        serie = []
        for p in range(288):
            m = act[p] > 0
            serie.append(round(float((Y[p][m] / vff[m]).mean()) if m.any() else 1.0, 3))
        aidd = {"edges": {}, "celda_edges": {}}
        inc_dia = inc[inc.dia_id == d]
        for f in ("gt", "obs"):
            z = za[f]
            eds = [str(e) for e in z[f"edges_{d}"]]
            z[f"p_{d}"].astype(np.uint8).tofile(OUT / "dias" / f"d{d}_aid_{f}.bin")
            aidd["edges"][f] = eds
            ie = {e: i for i, e in enumerate(eds)}
            aidd["celda_edges"][f] = {r.inc_id: [ie[e] for e in ari[ari.celda == r.celda_id].id if e in ie] for r in inc_dia.itertuples()}
            for e in eds:
                if e in por_id:
                    geom_aid[e] = rel(por_id[e]["shape"])
        incidentes = []
        for r in inc_dia.itertuples():
            e = por_id.get(r.edge)
            incidentes.append({"inc_id": r.inc_id, "edge": r.edge, "via": r.nombre_via or "vía local", "corredor": r.corredor != "ninguno",
                               "t_ini": float(r.t_ini), "t_fin": float(r.t_fin), "x": round(e["mid"][0] - x0), "y": round(e["mid"][1] - y0),
                               "severidad": r.severidad, "impacto": A.IMPACTO.get(r.inc_id)})
        v2x_corr = []
        for h in range(5, 24):
            sdf = df[(df.t // 3600 == h) & (df.gt_n_veh >= 1) & (df.obs_n_veh >= 1)]
            if len(sdf) > 50:
                v2x_corr.append([f"{h}h", round(float(np.corrcoef(sdf.gt_vel, sdf.obs_vel)[0, 1]), 3), "#7aa7ff"])
        # ficha: lo que caracteriza al dia
        fila = tabla.loc[d]
        dia_h = [(p, serie[p]) for p in range(6 * 12, 22 * 12)]
        p_lento = min(dia_h, key=lambda x: x[1])[0]
        ficha = {"dia": int(d), "semana": SEMANA[int(fila.dow)], "tipo": tp, "lluvia": bool(fila.lluvia),
                 "demanda_pct": round((float(fila.factor_demanda) - 1) * 100, 1), "penetracion_v2x": round(float(fila.penetracion_v2x) * 100),
                 "viajes": int(fila.n_viajes_completados) if "n_viajes_completados" in fila and fila.n_viajes_completados == fila.n_viajes_completados else None,
                 "incidentes": len(inc_dia), "incidentes_avenidas": int((inc_dia.corredor != "ninguno").sum()),
                 "incidentes_con_cola": int(sum(bool(A.IMPACTO.get(i)) for i in inc_dia.inc_id)),
                 "vel_rel_media": round(float(np.mean([v for _, v in dia_h])), 3), "hora_mas_lenta": f"{p_lento // 12:02d}:{p_lento % 12 * 5:02d}",
                 "vel_rel_min": round(dia_h[p_lento - 6 * 12][1], 3), "error_60": err,
                 "vel_viaje_kmh": round(float(fila.vel_media_viaje_kmh), 1), "demora_media_s": round(float(fila.perdida_media_s)),
                 "congestion_pico_pct": round(float(fila.pct_celdas_congestion_pico), 1)}
        return {"meta": {"dia": int(d), "tipo": tp, "lluvia": bool(fila.lluvia), "nombre_dia": nombre_dia(d)},
                "serie_red": {"rel": serie}, "incidentes": incidentes, "aid": aidd, "v2x_corr": v2x_corr, "ficha": ficha}

    dias_test = [int(x) for x in zp["gt"]["dias"]]
    por_dia = {}
    for d in dias_test:
        por_dia[d] = dia_datos(d)
        json.dump(por_dia[d], open(OUT / "dias" / f"d{d}.json", "w"), separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    # el dia inicial tambien queda en la raiz (compatibilidad y carga rapida)
    import shutil
    for f in ("gt", "obs"):
        shutil.copyfile(OUT / "dias" / f"d{a.dia}_pred_{f}.bin", OUT / f"pred_{f}.bin")
        shutil.copyfile(OUT / "dias" / f"d{a.dia}_aid_{f}.bin", OUT / f"aid_{f}.bin")
    metricas_pred = {f: leer(f"metricas_prediccion_{f}.json") for f in ("gt", "obs")}
    dia_ini = por_dia[a.dia]
    serie_rel, incidentes = dia_ini["serie_red"]["rel"], dia_ini["incidentes"]
    aid = {"edges": dia_ini["aid"]["edges"], "celda_edges": dia_ini["aid"]["celda_edges"], "geom": geom_aid}
    allp = np.array([por_id[e]["mid"] for e in geom_aid])
    aid["bbox"] = [float(allp[:, 0].min() - x0 - 400), float(allp[:, 1].min() - y0 - 400), float(allp[:, 0].max() - x0 + 400), float(allp[:, 1].max() - y0 + 400)]
    # ---- SIL
    sil = None
    for f in ("gt", "obs"):
        s = leer(f"sil_{a.inc}_{f}.json")
        if s:
            sil = s; break
    if sil:
        e0 = sil["incidente"]["edge"]
        tramo, fr = [e0], [e0]
        for _ in range(3):
            fr = [u for e in fr for u in A.ARRIBA.get(e, [])]; tramo += fr
        # linea de la cola: desde el punto del incidente hacia atras por las aristas de aguas arriba
        shp = por_id[e0]["shape"]
        posinc = float(C.incidentes().set_index("inc_id").loc[a.inc, "pos"])
        acc, linea = 0.0, []
        for p, q in zip(shp[:-1], shp[1:]):
            L = float(np.hypot(q[0] - p[0], q[1] - p[1]))
            if acc + L >= posinc:
                u = (posinc - acc) / L; linea = [[p[0] + (q[0] - p[0]) * u, p[1] + (q[1] - p[1]) * u]] + [list(pp) for pp in reversed(shp[:shp.index(q)])]
                break
            acc += L
        actual = e0
        for _ in range(6):
            ups = A.ARRIBA.get(actual, [])
            if not ups: break
            actual = ups[0]
            linea += [list(pp) for pp in reversed(por_id[actual]["shape"][:-1])]
        mx, my = linea[0]
        sil["geo"] = {"tramo": [rel(por_id[e]["shape"]) for e in tramo if e in por_id], "cola_linea": rel(linea),
                      "x": round(mx - x0), "y": round(my - y0), "vff": float(por_id[e0]["speed"] * 3.6),
                      "bbox": [mx - x0 - 650, my - y0 - 420, mx - x0 + 650, my - y0 + 420]}
        al = [p["min"] for p in sil["ia"]["serie"] if p["alarma_incidente"] and p["min"] >= sil["incidente"]["t_ini_min"]]
        sil["ia"]["alarma_min"] = al[0] if al else None
    # ---- contexto
    dv = tabla[tabla.split.isin(["train", "val", "test", "test_estres"])].reset_index()   # los dias extra del AID no tienen viajes completos
    color = {"train": "#2ad4c4", "val": "#ffb547", "test": "#ff4d5e", "test_estres": "#8fa0b6"}
    dias_viajes = [[int(r.dia_id), round(r.n_viajes_completados / 1000, 1), color[r.split]] for r in dv.itertuples()]
    v2x_corr = dia_ini["v2x_corr"]
    # ---- la IA frente a la referencia justa de cada parte (todo leido de los resultados)
    c = lambda x, n=2: f"{x:.{n}f}".replace(".", ",")
    tabla_ref = []
    mp = metricas_pred["gt"]["test"]; mo = metricas_pred["obs"]["test"]
    ia60, ha60 = mp["modelo"]["+60min"]["MAE"], mp["promedio_historico"]["+60min"]["MAE"]
    tabla_ref.append(["Predicción a 1 h (error)", f"{c(ha60)} km/h", f"{c(ia60)} km/h", "ok", f"−{c(100 * (1 - ia60 / ha60), 0)} %"])
    tabla_ref.append(["Predicción con V2X 50 % (error)", f"{c(mo['promedio_historico']['+60min']['MAE'])} km/h", f"{c(mo['modelo']['+60min']['MAE'])} km/h", "ok", "casi igual"])
    for d in dias_test:
        f = por_dia[d]["ficha"]
        if f["lluvia"] and f["error_60"]["gt"]:
            e = f["error_60"]["gt"]
            tabla_ref.append([f"Predicción en día de lluvia ({d})", f"{c(e['ha'])} km/h", f"{c(e['ia'])} km/h", "ok", f"−{c(100 * (1 - e['ia'] / e['ha']), 0)} %"])
    ma = leer("metricas_aid_gt.json")
    if ma:
        ti = ma["test_incidentes"]; tn = ma["test"]
        tabla_ref += [["Detector: accidentes con cola detectados", f"al azar {c(ti['azar_misma_FAR']['DR_con_impacto'], 0)} %", f"{c(ti['DR_con_impacto'], 0)} %", "ok" if ti["DR_con_impacto"] >= 85 else "med", "meta ≥ 85 %"],
                      ["Detector: falsas alarmas", "meta 3–5 %", f"{c(tn['FAR'], 1)} – {c(ti['FAR'], 1)} %", "ok" if ti["FAR"] <= 5 else "mal", "en la meta" if ti["FAR"] <= 5 else "fuera"],
                      ["Detector: minutos hasta la alarma", "meta < 5 min", f"{c(ti['MTTD_con_impacto_min'], 1)} – {c(tn['MTTD_con_impacto_min'], 1)} min", "ok" if tn["MTTD_con_impacto_min"] < 5 else "med", "días normales lento" if tn["MTTD_con_impacto_min"] >= 5 else "en la meta"]]
    if sil:
        bs, ia_ = sil["base"], sil["ia"]; cp = sil["comparacion_pct"]
        tabla_ref += [["Aviso V2X: demora de los afectados", f"sin IA {bs['afectados_perdida_media_s']:.0f} s", f"{ia_['afectados_perdida_media_s']:.0f} s", "ok", f"{c(cp['perdida_afectados'], 1)} %".replace("-", "−")],
                      ["Aviso V2X: demora de toda la red", f"sin IA {bs['perdida_media_s']:.0f} s", f"{ia_['perdida_media_s']:.0f} s", "ok", f"{c(cp['perdida_media_red'], 1)} %".replace("-", "−")]]
    pr = leer("../pruebas/informe.json")
    dia_sil = (sil or {}).get("incidente", {}).get("dia")
    datos = {"meta": {"dia": a.dia, "tipo": tipo, "lluvia": bool(tabla.loc[a.dia, "lluvia"]), "aristas": len(red["edges"]),
                      "nombre_dia": nombre_dia(a.dia), "nombre_dia_sil": nombre_dia(dia_sil) if dia_sil in tabla.index else ""},
             "dias": [por_dia[d]["ficha"] for d in dias_test], "ficha": dia_ini["ficha"],
             "bbox": [0, 0, b[2] - b[0], b[3] - b[1]], "celdas": celdas, "calles": calles, "corredores": corredores,
             "serie_red": {"rel": serie_rel}, "incidentes": incidentes, "aid": aid, "sil": sil,
             "dias_viajes": dias_viajes, "v2x_corr": v2x_corr, "pruebas": pr,
             "metricas": {"pred": metricas_pred, "aid": {f: leer(f"metricas_aid_{f}.json") for f in ("gt", "obs")},
                          "auditoria_tabla": tabla_ref,
                          "auditoria_nota": "Medido una sola vez en los días de test. La referencia es la forma simple de hacerlo sin IA: repetir lo habitual, disparar alarmas al azar o no avisar."}}
    def limpio(o):   # JSON valido para el navegador: NaN e infinito -> null
        if isinstance(o, float):
            return None if (o != o or o in (float("inf"), float("-inf"))) else o
        if isinstance(o, dict):
            return {k: limpio(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [limpio(v) for v in o]
        return o
    json.dump(limpio(datos), open(OUT / "datos.json", "w"), separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    print({p.name: f"{p.stat().st_size/1e6:.1f} MB" for p in OUT.iterdir() if p.is_file()}, "dias:", dias_test)


if __name__ == "__main__":
    main()
