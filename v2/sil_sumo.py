"""Validacion en bucle cerrado (SIL) con SUMO real y TraCI, sobre un incidente del TEST.

Dos corridas con el mismo dia, la misma demanda y la misma semilla:
  base : el AID corre en linea y registra sus alarmas, pero nadie actua.
  ia   : cuando el AID confirma la alarma, se envia un aviso tipo DENM a los autos CONECTADOS (V2X)
         que todavia tienen la arista afectada en su ruta; cada uno recalcula su ruta evitandola.
         Los autos no conectados no se enteran (realista con 50 % de penetracion).

Nada esta escrito a mano: las colas, demoras y tiempos de viaje salen de SUMO (TraCI + tripinfo).
Uso:  python sil_sumo.py --inc inc37_5 --aid gt
"""
import argparse, gzip, importlib.util, json, os, subprocess, sys, time
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np, pandas as pd, joblib
import comun as C
import entrenar_aid as A

RAIZ = C.RAIZ
spec = importlib.util.spec_from_file_location("etapa2_comun", RAIZ / "etapa2" / "comun.py")
E2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(E2)
import traci  # 1.12, igual que el SUMO del contenedor

CAL = E2.CALENTAMIENTO
PUERTO = 8813
RADIO_DENM = 2000.0     # m de recorrido antes del incidente (area de relevancia del DENM, ETSI EN 302 637-3)
CALENTAMIENTO_SIL = 60 * 60   # s: red casi vacia al inicio y rasgos de 60 min del AID; en ese lapso no se actua
CADUCA_MIN = 3          # min sin alarma tras los cuales el DENM se cancela (validez del DENM, ETSI)
DEMORA_DENM = 600.0     # s que el navegador suma al tiempo actual de la arista bloqueada (duracion esperada)


def preparar(inc, trabajo, t0, t1):
    """Reproduce el dia del dataset: mismos viajes (misma semilla) y rutas del DIA COMPLETO con duarouter
    (misma semilla), igual que generar_dataset.py; despues se recortan los autos que salen en la ventana."""
    extra = inc.dia_id >= 100      # dias extra de generar_dias_incidentes.py (04:00 a 21:30, lluvia ya en el factor)
    dia = json.load(open(C.DS / ("dias_aid_json" if extra else "dias_json") / f"dia_{inc.dia_id:02d}.json"))["dia"]
    if (trabajo / "rutas.rou.xml.gz").exists() and (trabajo / "rutas_dia.ok").exists():
        return dia
    T_FIN = int(21.5 * 3600) + CAL if extra else 25 * 3600
    factor = dia["factor_efectivo"] * (1.04 if dia["lluvia"] and not extra else 1.0)
    tasas = E2._tasas_por_segundo(E2.PERFILES[dia["tipo_dia"]], dia["v0"], factor, T_FIN)
    if extra:
        tasas[:4 * 3600 + CAL] = 0
    E2.PENETRACION_V2X = dia["penetracion_v2x"]
    E2.generar_viajes(trabajo / "viajes.trips.xml", tasas, dia["semilla"])
    E2.escribir_vtypes(trabajo / "vtypes.add.xml", 0.88 if dia["lluvia"] else 1.0)
    E2.escribir_salidas(trabajo / "salidas.add.xml", con_corredores=True)
    lineas = [l for l in open(trabajo / "salidas.add.xml") if "edgeData" not in l]
    open(trabajo / "salidas.add.xml", "w").write("".join(lineas).replace('file="/t/e1_gt.xml.gz"', 'file="/dev/null"').replace('file="/t/e1_v2x.xml.gz"', 'file="/dev/null"'))
    E2.duarouter(trabajo, dia["semilla"])
    (trabajo / "viajes.trips.xml").unlink()
    completo = trabajo / "rutas.rou.xml.gz"
    tmp = trabajo / "ventana.rou.xml.gz"
    with gzip.open(completo) as f, gzip.open(tmp, "wt") as g:
        g.write("<routes>\n")
        for ev, el in ET.iterparse(f, events=("end",)):
            if el.tag == "vehicle":
                if t0 <= float(el.get("depart")) < t1:
                    g.write(ET.tostring(el, encoding="unicode"))
                el.clear()
        g.write("</routes>\n")
    tmp.replace(completo)
    (trabajo / "rutas_dia.ok").write_text("ok")
    info = {"inc_id": inc.inc_id, "edge": inc.edge, "pos": float(inc.pos), "carriles": [int(c) for c in str(inc.carriles).split(",")],
            "t_ini_plan": int(inc.t_ini_plan), "duracion": int(inc.duracion)}
    E2.escribir_incidentes(trabajo / "incidentes.rou.xml", [info])
    return dia


def arrancar_sumo(trabajo, t0, t1, semilla, nombre):
    subprocess.run(["docker", "rm", "-f", nombre], capture_output=True)
    cmd = ["docker", "run", "-d", "--rm", "--name", nombre, "-p", f"{PUERTO}:{PUERTO}", "-u", f"{os.getuid()}:{os.getgid()}",
           "-e", "SUMO_HOME=/usr/share/sumo", "-v", f"{RAIZ}:/w:ro", "-v", f"{trabajo}:/t", "-w", "/w", E2.IMG,
           "sumo", "-n", E2.RED, "-r", "/t/rutas.rou.xml.gz,/t/incidentes.rou.xml", "-a", "/t/vtypes.add.xml,/t/salidas.add.xml",
           "--begin", str(t0), "--end", str(t1), "--seed", str(semilla), "--no-step-log", "--no-warnings",
           "--time-to-teleport", "300", "--max-depart-delay", "900", "--ignore-junction-blocker", "30",
           "--tripinfo-output", f"/t/tripinfo_{nombre}.xml", "--remote-port", str(PUERTO), "--num-clients", "1"]
    subprocess.run(cmd, check=True, capture_output=True)
    for _ in range(60):
        try:
            traci.init(PUERTO); return
        except Exception:
            time.sleep(1)
    raise RuntimeError("SUMO no respondio")


def corrida(modo_ia, inc, trabajo, t0, t1, dia, aid, tipo, U):
    nombre = "sil_ia" if modo_ia else "sil_base"
    arrancar_sumo(trabajo, t0, t1, dia["semilla"], nombre)
    lluvia_dia = float(bool(dia["lluvia"]))
    vff = pd.read_parquet(C.DS / "corredor_1min" / f"dia_{inc.dia_id:02d}.parquet", columns=["edge", "v_ff"]).groupby("edge").v_ff.first()
    edges = sorted(vff.index); ie = {e: i for i, e in enumerate(edges)}
    E = len(edges)
    vista = "gt" if aid["modo"] == "gt" else "v2x"
    pen = 1.0 if aid["modo"] == "gt" else float(dia["penetracion_v2x"])
    dets = [d for d in traci.inductionloop.getIDList() if d.startswith(vista + "|")]
    por_arista = {}
    for d in dets:
        _, e, _ = d.split("|"); por_arista.setdefault(e, []).append(d)
    m = {k: np.full((E, 1440), np.nan, np.float32) for k in ("vel", "ocup", "flujo")}
    m["sin"] = np.zeros((E, 1440), np.float32)
    # arista del incidente y tramo de aguas arriba (hasta 3 aristas) para medir la cola
    tramo, frontera = [inc.edge], [inc.edge]
    for _ in range(3):
        frontera = [u for e in frontera for u in A.ARRIBA.get(e, [])]
        tramo += frontera
    largo = {e: traci.lane.getLength(e + "_0") for e in tramo}
    serie, alarmas_on, primera_alarma, avisados, afectados = [], np.zeros(E, bool), None, set(), set()
    racha = np.zeros(E, int)
    penalizados, sin_alarma, cancelados = {}, 0, 0      # DENM vigentes: auto -> aristas penalizadas
    aristas_celda = set(pd.read_parquet(C.DS / "aristas.parquet", filters=[("celda", "==", int(inc.celda_id))]).id)
    celda_inc = [ie[e] for e in edges if e in aristas_celda]
    i_inc = ie.get(inc.edge)
    import traci.constants as tc
    det_lista = [d for e in edges for d in por_arista.get(e, [])]
    for d in det_lista:
        traci.inductionloop.subscribe(d, (tc.LAST_STEP_VEHICLE_DATA, tc.LAST_STEP_OCCUPANCY))
    # igual que un detector E1 (nVehContrib, harmonicMeanSpeed): cada auto cuenta UNA vez, cuando termina de cruzar el
    # lazo, con su velocidad de cruce = largo / tiempo sobre el lazo. (Contar pasos hacia pesar 30 veces a un auto
    # detenido en un semaforo y la velocidad salia ~1 km/h; el banco de pruebas lo detecto.)
    acc = {d: [0, 0.0, 0.0, 0] for d in det_lista}      # n, sum(1/v), sum(ocupacion), pasos
    t = t0
    while t < t1:
        traci.simulationStep()
        t += 1
        # afectados: autos cuya ruta ORIGINAL pasa por la arista del incidente (misma definicion en ambas corridas)
        for v in traci.simulation.getDepartedIDList():
            if inc.edge in traci.vehicle.getRoute(v):
                afectados.add(v)
        for d, r in traci.inductionloop.getAllSubscriptionResults().items():
            a_ = acc[d]
            a_[2] += r[tc.LAST_STEP_OCCUPANCY]; a_[3] += 1
            for (_v, largo, entra, sale, _tipo) in r[tc.LAST_STEP_VEHICLE_DATA]:
                if sale >= 0:
                    a_[0] += 1; a_[1] += 1.0 / max(largo / max(sale - entra, 1e-3), 0.1)
        if t % 30:
            continue
        tdia = t - CAL
        minuto = int(tdia // 60)
        if t % 60 == 0 and 0 <= minuto - 1 < 1440:
            k = minuto - 1                        # el minuto que acaba de cerrar, agregado como los detectores E1 del dataset
            for e, ds in por_arista.items():
                if e not in ie: continue
                i = ie[e]
                n = sum(acc[d][0] for d in ds); inv = sum(acc[d][1] for d in ds)
                m["vel"][i, k] = (n / inv * 3.6) / vff[e] if n > 0 and inv > 0 else np.nan
                m["ocup"][i, k] = np.mean([acc[d][2] / max(acc[d][3], 1) for d in ds]) / (1.0 if aid["modo"] == "gt" else pen)
                m["flujo"][i, k] = n / len(ds) * 60 / pen
                m["sin"][i, k] = float(n == 0)
            for d in det_lista:
                acc[d] = [0, 0.0, 0.0, 0]
            # mismo preprocesamiento que en el entrenamiento (arrastrar la ultima velocidad medida)
            mm = {"vel": pd.DataFrame(m["vel"][:, :k + 1].T).ffill().fillna(1.0).to_numpy().T, "ocup": np.nan_to_num(m["ocup"][:, :k + 1]),
                  "flujo": np.nan_to_num(m["flujo"][:, :k + 1]), "sin": m["sin"][:, :k + 1]}
            mm = {kk: np.pad(v, ((0, 0), (0, 1440 - v.shape[1])), constant_values=0 if kk != "vel" else 1.0) for kk, v in mm.items()}
            mm["edges"] = edges
            mm["lluvia"] = lluvia_dia
            X = A.rasgos_desde(mm, tipo, aid["HA"])[:, k]
            p = aid["modelo"].predict_proba(X)[:, 1]
            racha = np.where(p >= A.umbral_de(U, lluvia_dia), racha + 1, 0)
            nuevas = racha >= aid["persistencia"]
            alarmas_on = nuevas
            alarma_inc = bool(alarmas_on[celda_inc].any()) if celda_inc else False
            if alarma_inc and primera_alarma is None and tdia >= inc.t_ini:
                primera_alarma = tdia
            n_avisos = 0
            actua = modo_ia and tdia >= t0 - CAL + CALENTAMIENTO_SIL
            sin_alarma = 0 if alarma_inc else sin_alarma + 1
            if actua and sin_alarma == CADUCA_MIN and penalizados:
                # el DENM caduca: los autos olvidan la penalizacion y vuelven a la ruta mas rapida con el trafico actual
                vivos = set(traci.vehicle.getIDList())
                for v, aristas in penalizados.items():
                    if v in vivos:
                        for b in aristas:
                            traci.vehicle.setAdaptedTraveltime(v, b)
                        traci.vehicle.rerouteTraveltime(v, currentTravelTimes=True)
                cancelados += len(penalizados); penalizados = {}
            if actua and alarma_inc:
                # DENM: los autos conectados que aun tienen la arista bloqueada por delante recalculan su ruta
                malas = [edges[j] for j in celda_inc if alarmas_on[j]] or [inc.edge]
                demora = {b: traci.edge.getTraveltime(b) + DEMORA_DENM for b in malas}
                for v in traci.vehicle.getIDList():
                    if traci.vehicle.getTypeID(v) != "auto_v2x" or v in penalizados: continue
                    ruta = traci.vehicle.getRoute(v); pos = traci.vehicle.getRouteIndex(v)
                    resto = ruta[pos + 1:]
                    adelante = [b for b in malas if b in resto]
                    if not adelante:
                        continue
                    dist = traci.vehicle.getDrivingDistance(v, adelante[0], 0.0)
                    if not (0 < dist <= RADIO_DENM):
                        continue          # fuera del area de relevancia: el DENM no le llega o no le aplica
                    for b in adelante:
                        traci.vehicle.setAdaptedTraveltime(v, b, demora[b])
                    traci.vehicle.rerouteTraveltime(v, currentTravelTimes=True)    # tiempos actuales de toda la red (como un navegador con trafico); solo se desvia si la alternativa es mas rapida
                    avisados.add(v); penalizados[v] = adelante; n_avisos += 1
            serie.append({"min": round(tdia / 60, 1), "prob_max_celda": round(float(p[celda_inc].max()), 3) if celda_inc else 0,
                          "alarmas_corredor": int(alarmas_on.sum()), "alarma_incidente": alarma_inc,
                          "vel_arista_kmh": None if np.isnan(m["vel"][i_inc, k]) else round(float(m["vel"][i_inc, k] * vff[inc.edge]), 1),
                          "avisos_denm": n_avisos})
        if True:
            det = sum(traci.edge.getLastStepHaltingNumber(e) for e in tramo)
            cola = max(traci.edge.getLastStepHaltingNumber(e) * 7.0 / max(traci.edge.getLaneNumber(e), 1) for e in tramo)
            vel_tramo = np.mean([traci.edge.getLastStepMeanSpeed(e) * 3.6 for e in tramo])
            if t % 60 == 0 and serie:
                serie[-1].update({"detenidos_tramo": int(det), "cola_m_por_carril": round(float(cola), 1),
                                  "vel_tramo_kmh": round(float(vel_tramo), 1), "autos_en_red": traci.vehicle.getIDCount()})
    traci.close()
    time.sleep(2)
    subprocess.run(["docker", "rm", "-f", nombre], capture_output=True)
    trip = E2.leer_tripinfo(trabajo / f"tripinfo_{nombre}.xml")
    (trabajo / f"tripinfo_{nombre}.xml").unlink()
    lo, hi = int((t0 - CAL) // 60), int((t1 - CAL) // 60)    # solo la ventana simulada, para comparar con el dataset
    np.savez_compressed(C.SALIDA / f"sil_matrices_{inc.inc_id}_{nombre}.npz", edges=np.array(edges), desde=lo,
                        **{k: v[:, lo:hi].astype(np.float16) for k, v in m.items()})
    return serie, trip, primera_alarma, len(avisados), afectados, cancelados


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inc", default="inc119_3")
    ap.add_argument("--punto", choices=["estricto", "meta"], default="estricto",
                    help="umbral para ACTUAR: estricto = FAR <= 1 %% (por defecto, un DENM falso desvia autos reales); meta = FAR <= 3 %%")
    ap.add_argument("--aid", choices=["gt", "obs"], default="gt")
    a = ap.parse_args()
    inc = C.incidentes().set_index("inc_id").loc[a.inc]
    inc["inc_id"] = a.inc
    aid = joblib.load(C.SALIDA / f"modelo_aid_{a.aid}.joblib")
    met = json.load(open(C.SALIDA / f"metricas_aid_{a.aid}.json"))
    sel = met["estricto_far1"]["seleccion"] if a.punto == "estricto" else met["seleccion_en_val"]
    U = {"seco": sel["umbral"], "lluvia": sel["umbral_lluvia"]}
    aid = dict(aid, persistencia=sel["persistencia"])
    tipo = C.dias().loc[inc.dia_id, "tipo_dia"]
    trabajo = Path(os.environ.get("SIL_TRABAJO", "/tmp/claude-1000/sil")) / a.inc
    trabajo.mkdir(parents=True, exist_ok=True)
    t0 = int((inc.t_ini + CAL - 5400) // 60 * 60)      # 1 h 30 min antes: la red se llena y se estabiliza
    t1 = int(inc.t_fin + CAL + 2700)      # 45 min despues del fin: se mide la recuperacion
    dia = preparar(inc, trabajo, t0, t1 - 900)
    res = {"incidente": {"inc_id": a.inc, "dia": int(inc.dia_id), "via": inc.nombre_via, "corredor": inc.corredor, "edge": inc.edge,
                         "carriles_bloqueados": int(inc.carriles_bloqueados), "carriles_via": int(inc.n_carriles_via),
                         "t_ini_min": round(inc.t_ini / 60, 1), "t_fin_min": round(inc.t_fin / 60, 1), "duracion_min": round(inc.duracion / 60, 1)},
           "aid": a.aid, "punto_actuacion": {"nombre": a.punto, **U, "persistencia": sel["persistencia"], "FAR_maxima": sel["FAR_maxima"]},
           "calentamiento_min": CALENTAMIENTO_SIL // 60, "denm_caduca_min": CADUCA_MIN, "ventana_min": [round((t0 - CAL) / 60, 1), round((t1 - CAL) / 60, 1)]}
    ap_solo = os.environ.get("SIL_SOLO_BASE") == "1"      # depuracion: solo la corrida base
    for etq, ia in ((("base", False),) if ap_solo else (("base", False), ("ia", True))):
        t = time.time()
        serie, trip, pa, n_av, afect, n_can = corrida(ia, inc, trabajo, t0, t1, dia, aid, tipo, U)
        t_ini_m = inc.t_ini / 60
        previa = any(q["alarma_incidente"] for q in serie if t_ini_m - 3 <= q["min"] < t_ini_m)
        tr_af = trip[trip.veh_id.isin(afect)]
        res[etq] = {"serie": serie, "mttd_min": None if pa is None else round((pa - inc.t_ini) / 60, 2), "autos_avisados_denm": n_av, "denm_cancelados": n_can,
                    "alarma_previa": previa, "avisos_antes_del_incidente": int(sum(q["avisos_denm"] for q in serie if q["min"] < t_ini_m)),
                    "viajes": int(len(trip)), "vht_h": round(float(trip.duracion.sum() / 3600), 1),
                    "tiempo_viaje_medio_s": round(float(trip.duracion.mean()), 1), "perdida_media_s": round(float(trip.perdida.mean()), 1),
                    "afectados": int(len(tr_af)), "afectados_tiempo_viaje_medio_s": round(float(tr_af.duracion.mean()), 1) if len(tr_af) else None,
                    "afectados_perdida_media_s": round(float(tr_af.perdida.mean()), 1) if len(tr_af) else None,
                    "afectados_perdida_total_veh_h": round(float(tr_af.perdida.sum() / 3600), 1),
                    "cola_max_m_por_carril": max(s.get("cola_m_por_carril", 0) for s in serie),
                    "detenidos_max": max(s.get("detenidos_tramo", 0) for s in serie),
                    "vel_tramo_min_kmh": min(s.get("vel_tramo_kmh", 99) for s in serie if s["min"] >= res["incidente"]["t_ini_min"]),
                    "seg": round(time.time() - t)}
        print(etq, {k: v for k, v in res[etq].items() if k != "serie"}, flush=True)
    if ap_solo:
        return
    b, i = res["base"], res["ia"]
    pct = lambda x, y: round(100 * (y - x) / x, 1) if x else None
    res["comparacion_pct"] = {"perdida_afectados": pct(b["afectados_perdida_media_s"], i["afectados_perdida_media_s"]),
                              "tiempo_viaje_afectados": pct(b["afectados_tiempo_viaje_medio_s"], i["afectados_tiempo_viaje_medio_s"]),
                              "cola_max": pct(b["cola_max_m_por_carril"], i["cola_max_m_por_carril"]),
                              "vht_red": pct(b["vht_h"], i["vht_h"]), "perdida_media_red": pct(b["perdida_media_s"], i["perdida_media_s"])}
    print("comparacion", res["comparacion_pct"])
    json.dump(res, open(C.SALIDA / f"sil_{a.inc}_{a.aid}.json", "w"), indent=1, ensure_ascii=False, default=float)


if __name__ == "__main__":
    main()
