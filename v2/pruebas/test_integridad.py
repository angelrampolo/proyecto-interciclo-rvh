"""Pruebas de integridad: causalidad, recalculo independiente de las metricas, sanidad fisica,
reproducibilidad, SIL contra el dataset, tablero en un navegador real y coherencia de los documentos.

Complementa a test_proyecto.py (que revisa las reglas y las metas). Corre con:
    cd v2 && python -m pytest pruebas/ -q
"""
import functools, http.server, json, py_compile, re, shutil, socket, subprocess, sys, threading
from pathlib import Path
import numpy as np, pandas as pd, pytest

AQUI = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(AQUI))
import comun as C  # noqa: E402

R = C.SALIDA
DASH = AQUI / "dashboard"
INF = AQUI / "informe"


def j(nombre):
    p = R / nombre
    if not p.exists():
        pytest.skip(f"falta {nombre}")
    return json.load(open(p))


@functools.lru_cache(None)
def aid_mod():
    import entrenar_aid as A
    A.cargar_impacto()
    return A


def sil():
    ps = sorted(R.glob("sil_*.json"))
    if not ps:
        pytest.skip("falta correr sil_sumo.py")
    return json.load(open(ps[0]))


# ------------------------------------------------------------------ codigo
@pytest.mark.grupo("codigo")
def test_scripts_compilan():
    """Todos los programas de Python del proyecto compilan sin errores de sintaxis."""
    for p in list(AQUI.glob("*.py")) + list((AQUI / "pruebas").glob("*.py")) + list(INF.glob("*.py")):
        py_compile.compile(str(p), doraise=True)


@pytest.mark.grupo("codigo")
def test_metricas_unitarias():
    """La funcion de metricas da lo esperado en un caso calculado a mano (MAPE ignora velocidades <= 2 km/h)."""
    y = np.array([10.0, 20.0, 1.0, 40.0]); p = np.array([12.0, 18.0, 5.0, 40.0]); m = np.array([True, True, True, False])
    r = C.metricas(y, p, m)
    assert r["MAE"] == pytest.approx((2 + 2 + 4) / 3, abs=1e-3)
    assert r["RMSE"] == pytest.approx(np.sqrt((4 + 4 + 16) / 3), abs=1e-3)
    assert r["MAPE"] == pytest.approx((0.2 + 0.1) / 2 * 100, abs=1e-2)
    assert r["n"] == 3


@pytest.mark.grupo("codigo")
def test_alarma_exige_minutos_seguidos():
    """La alarma del AID solo se activa tras 'persistencia' minutos seguidos sobre el umbral."""
    A = aid_mod()
    p = np.array([[0.9, 0.1, 0.9, 0.9, 0.9, 0.1]])
    assert A.alarmas(p, 0.5, 1)[0].tolist() == [True, False, True, True, True, False]
    assert A.alarmas(p, 0.5, 2)[0].tolist() == [False, False, False, True, True, False]
    assert A.alarmas(p, 0.5, 3)[0].tolist() == [False, False, False, False, True, False]
    assert A.umbral_de({"seco": 0.1, "lluvia": 0.3}, True) == 0.3 and A.umbral_de(0.2, True) == 0.2


@pytest.mark.grupo("codigo")
def test_grafo_vial_valido():
    """El grafo de celdas es dirigido, sin lazos, con indices validos y cubre la mayoria de la ciudad."""
    ij = C.adyacencia()
    N = len(C.celdas())
    assert ij.min() >= 0 and ij.max() < N
    assert not np.any(ij[0] == ij[1])
    assert len(np.unique(ij[0])) > 0.8 * N
    pares = set(map(tuple, ij.T)); asim = sum((b, a) not in pares for a, b in pares)
    assert asim > 0, "el grafo deberia tener sentidos unicos (calles de una via)"


# ------------------------------------------------------------------ datos
@pytest.mark.grupo("datos")
def test_dias_extra_bien_formados():
    """Los 20 dias extra del AID tienen 8 incidentes cada uno, no se mezclan con los originales y respetan el plan."""
    d = C.dias()
    extra = d[d.index >= 100]
    assert len(extra) == 20
    assert set(extra.split) == {"aid_train", "aid_val", "aid_test"}
    assert not set(extra.index) & set(d[d.index < 100].index)
    inc = C.incidentes(); inc = inc[inc.dia_id >= 100]
    assert (inc.groupby("dia_id").size() == 8).all()
    assert inc.t_ini.between(6.5 * 3600 - 60, 20.5 * 3600 + 60).all()
    for (_, _), g in inc.groupby(["dia_id", "corredor"]):
        t = np.sort(g.t_ini_plan.to_numpy())
        assert (np.diff(t) >= 40 * 60 - 1).all(), "dos incidentes del mismo corredor a menos de 40 min"
    for i in extra.index:
        assert (C.DS / "corredor_1min" / f"dia_{i}.parquet").exists()


@pytest.mark.grupo("datos")
def test_detectores_fisicamente_validos():
    """En los datos por minuto: velocidades entre 0 y 1,5 veces la libre, ocupacion entre 0 y 100 %, flujos no negativos."""
    for d in (37, 119):
        df = pd.read_parquet(C.DS / "corredor_1min" / f"dia_{d:02d}.parquet")
        v = df.gt_vel.dropna()
        assert (v >= 0).all() and (v <= 1.5 * df.loc[v.index, "v_ff"]).all()
        assert df.gt_ocupacion.between(0, 100).all()
        assert (df.gt_flujo >= 0).all() and (df.obs_n <= df.gt_n).all()


# ------------------------------------------------------------------ prediccion
@pytest.fixture(scope="module")
def datos_dia36():
    torch = pytest.importorskip("torch")
    import entrenar_prediccion as E
    p = R / "modelo_gcngru_gt.pt"
    if not p.exists():
        pytest.skip("falta el modelo")
    ck = torch.load(p, weights_only=False)
    m = E.GCNGRU(len(C.celdas()), ck["ij"]); m.load_state_dict(ck["estado"]); m.eval()
    d = E.Datos([36], "gt", ck["HA"]); E.normalizar(d, ck["mu"], ck["sd"])
    return E, m, d


@pytest.mark.grupo("prediccion")
def test_prediccion_no_usa_el_futuro(datos_dia36):
    """Cambiar todo lo que pasa despues del instante de prediccion no cambia la prediccion (causalidad)."""
    E, m, d = datos_dia36
    i = 150; dd, t = d.idx[i]
    d.idx = [d.idx[i]]; d.tiempo = d.tiempo[i:i + 1]
    base = E.predecir(m, d)
    guardado = d.X[dd, t + 1:].copy()
    d.X[dd, t + 1:] = np.random.default_rng(0).normal(0, 5, guardado.shape)
    futuro = E.predecir(m, d)
    d.X[dd, t + 1:] = guardado
    assert np.abs(base - futuro).max() < 1e-3      # km/h; margen para el redondeo float32 de PyTorch con varios hilos
    guardado = d.X[dd, t].copy(); d.X[dd, t, :, 0] = 0.1          # en cambio, el presente si importa
    presente = E.predecir(m, d); d.X[dd, t] = guardado
    assert np.abs(base - presente).max() > 0.5


@pytest.mark.grupo("prediccion")
@pytest.mark.parametrize("modo", ["gt", "obs"])
def test_prediccion_metricas_recalculadas(modo):
    """Recalculando el MAE a +60 min desde las predicciones guardadas se obtiene lo mismo que reporta el JSON."""
    p = R / f"pred_test_{modo}.npz"
    if not p.exists():
        pytest.skip("faltan las predicciones")
    z = np.load(p)
    _, Y, M, _ = C.cargar_dias(list(z["dias"]), modo)
    P = z["P"][..., 3].astype(np.float32); idx = z["idx"]
    y = np.stack([Y[d, t + C.H] for d, t in idx]); mk = np.stack([M[d, t + C.H] for d, t in idx])
    mae = float(np.abs(P - y)[mk].mean())
    rep = j(f"metricas_prediccion_{modo}.json")["test"]["modelo"]["+60min"]["MAE"]
    assert abs(mae - rep) < 0.12, (mae, rep)      # tolerancia por guardar en enteros (uint8)


@pytest.mark.grupo("prediccion")
@pytest.mark.parametrize("modo", ["gt", "obs"])
def test_prediccion_fisica_y_sin_sesgo(modo):
    """Predicciones sin NaN, bajo el limite fisico de SUMO (factor de velocidad maximo 2), sin mas excesos que la
    realidad y sin sesgo sistematico (media del error < 1 km/h)."""
    z = np.load(R / f"pred_test_{modo}.npz")
    _, Y, M, _ = C.cargar_dias(list(z["dias"]), modo)
    vff = C.celdas().v_ff.to_numpy()
    P = z["P"].astype(np.float32)
    assert np.isfinite(P).all() and (P <= 2.0 * vff[None, :, None] + 1).all()
    exceso_pred = (P[..., 3] > 1.3 * vff).mean(); exceso_real = (z["Y"] > 1.3 * vff).mean()
    assert exceso_pred < 0.005 and exceso_pred < 5 * exceso_real + 1e-4, (exceso_pred, exceso_real)
    idx = z["idx"]
    e = np.stack([P[w, :, 3] - Y[d, t + C.H] for w, (d, t) in enumerate(idx)])
    mk = np.stack([M[d, t + C.H] for d, t in idx])
    assert abs(float(e[mk].mean())) < 1.0, float(e[mk].mean())


# ------------------------------------------------------------------ AID
@pytest.mark.grupo("aid")
def test_aid_rasgos_no_usan_el_futuro():
    """Los rasgos de un minuto calculados con el dia completo son identicos a los calculados solo con el pasado
    (asi los usa el bucle SIL en linea)."""
    A = aid_mod()
    HA = A.promedio_historico("gt")
    m = A.matrices_dia(37, "gt")
    tipo = C.dias().loc[37, "tipo_dia"]
    completo = A.rasgos_desde(m, tipo, HA)
    for k in (8 * 60, 12 * 60 + 7, 18 * 60 + 30):
        mm = {kk: np.pad(m[kk][:, :k + 1], ((0, 0), (0, 1440 - k - 1)), constant_values=1.0 if kk == "vel" else 0)
              for kk in ("vel", "ocup", "flujo", "sin")}
        mm["edges"] = m["edges"]; mm["lluvia"] = m["lluvia"]
        parcial = A.rasgos_desde(mm, tipo, HA)
        assert np.allclose(completo[:, k], parcial[:, k], atol=1e-4), f"minuto {k}"


@pytest.mark.grupo("aid")
def test_aid_reproducible():
    """Cargar el detector guardado y volver a predecir el dia 36 da las mismas probabilidades guardadas."""
    import joblib
    A = aid_mod()
    b = joblib.load(R / "modelo_aid_gt.joblib")
    probs, _ = A.predecir_dias(b["modelo"], [36], "gt", b["HA"])
    z = np.load(R / "aid_test_gt.npz")
    assert np.abs(probs[36] * 255 - z["p_36"].astype(np.float32)).max() <= 1.01


@pytest.mark.grupo("aid")
def test_aid_metricas_recalculadas():
    """Recalculando DR y FAR del test desde las probabilidades guardadas se obtiene lo que reporta el JSON."""
    A = aid_mod()
    r = j("metricas_aid_gt.json"); sel = r["seleccion_en_val"]
    z = np.load(R / "aid_test_gt.npz")
    dias = C.ids("test")
    probs = {d: z[f"p_{d}"].astype(np.float32) / 255 for d in dias}
    mats = {}
    for d in dias:
        m = A.matrices_dia(d, "gt"); assert list(m["edges"]) == list(z[f"edges_{d}"])
        mats[d] = {k: m[k] for k in ("edges", "y", "celda")}
    U = {"seco": sel["umbral"], "lluvia": sel["umbral_lluvia"]}
    e = A.evaluar(dias, probs, mats, U, sel["persistencia"])
    assert abs(e["DR"] - r["test"]["DR"]) <= 6.0 and abs(e["FAR"] - r["test"]["FAR"]) <= 0.3, (e["DR"], e["FAR"])


@pytest.mark.grupo("aid")
def test_aid_umbral_elegido_sin_test():
    """Los dias usados para elegir el umbral son exactamente train + aid_train (fuera de pliegue) + validacion."""
    sel = j("metricas_aid_gt.json")["seleccion_en_val"]
    esperado = len(C.ids("train") + C.ids("aid_train") + C.ids("val") + C.ids("aid_val"))
    assert sum(sel["dias_seleccion"].values()) == esperado
    lluv = C.dias().lluvia
    assert sel["dias_seleccion"]["lluvia"] == int(sum(lluv.loc[d] for d in C.ids("train") + C.ids("aid_train") + C.ids("val") + C.ids("aid_val")))


# ------------------------------------------------------------------ SIL
@pytest.mark.grupo("sil")
def test_sil_incidente_forma_cola():
    """El incidente del SIL forma una cola real (si no, la comparacion no mide nada)."""
    s = sil()
    assert s["base"]["cola_max_m_por_carril"] > 50 and s["base"]["detenidos_max"] >= 10


@pytest.mark.grupo("sil")
def test_sil_corridas_comparables():
    """Ambas corridas: mismos afectados (definidos por la ruta original), aviso solo en la de IA, ningun aviso durante el
    calentamiento y la mayoria de los avisos despues del inicio real del incidente."""
    s = sil(); b, i = s["base"], s["ia"]
    assert abs(b["afectados"] - i["afectados"]) / b["afectados"] < 0.02
    assert b["autos_avisados_denm"] == 0 and i["autos_avisados_denm"] > 0
    assert 0 <= i["mttd_min"] < 5
    assert all(p["avisos_denm"] == 0 for p in b["serie"])
    t_ini = s["incidente"]["t_ini_min"]; cal = s["ventana_min"][0] + s.get("calentamiento_min", 30)
    assert all(p["min"] >= cal for p in i["serie"] if p["avisos_denm"] > 0), "aviso durante el calentamiento"


@pytest.mark.grupo("sil")
def test_sil_avisos_mayormente_utiles():
    """Meta de operacion: la mayoria (75 %) de los avisos DENM corresponden al incidente real y no a falsas alarmas previas."""
    s = sil(); i = s["ia"]; t_ini = s["incidente"]["t_ini_min"]
    antes = sum(p["avisos_denm"] for p in i["serie"] if p["min"] < t_ini)
    assert antes <= 0.25 * i["autos_avisados_denm"], f"{antes} de {i['autos_avisados_denm']} avisos fueron por falsas alarmas previas"


@pytest.mark.grupo("sil")
def test_sil_porcentajes_consistentes():
    """Los porcentajes de la comparacion salen de los valores base e IA reportados."""
    s = sil(); b, i, c = s["base"], s["ia"], s["comparacion_pct"]
    pct = lambda a, x: round(100 * (x - a) / a, 1)
    assert c["perdida_afectados"] == pytest.approx(pct(b["afectados_perdida_media_s"], i["afectados_perdida_media_s"]), abs=0.15)
    assert c["vht_red"] == pytest.approx(pct(b["vht_h"], i["vht_h"]), abs=0.15)
    assert c["perdida_media_red"] == pytest.approx(pct(b["perdida_media_s"], i["perdida_media_s"]), abs=0.15)


@pytest.mark.grupo("sil")
def test_sil_reproduce_el_dia_del_dataset():
    """La corrida base del SIL reproduce el dia del dataset en la arista del incidente: misma distribucion de velocidad
    fuera del cierre y la arista se vacia durante el cierre total (el ruido minuto a minuto es otra realizacion aleatoria)."""
    s = sil(); inc = s["incidente"]
    df = pd.read_parquet(C.DS / "corredor_1min" / f"dia_{inc['dia']}.parquet", columns=["edge", "t", "gt_vel"])
    de = df[df.edge == inc["edge"]]
    ds = pd.Series(de.gt_vel.to_numpy(), index=(de.t // 60).astype(int).to_numpy())
    t0, t1 = inc["t_ini_min"], inc["t_fin_min"]
    fuera = [(p["vel_arista_kmh"], ds.get(int(p["min"]))) for p in s["base"]["serie"]
             if p["vel_arista_kmh"] is not None and p["min"] >= s["ventana_min"][0] + 30 and not (t0 - 1 <= p["min"] <= t1 + 3)]
    a, b = np.array([x for x in fuera if x[1] is not None and np.isfinite(x[1])]).T
    assert len(a) > 30
    assert abs(a.mean() - b.mean()) < 3, (a.mean(), b.mean())
    assert 0.6 < a.std() / b.std() < 1.6, (a.std(), b.std())
    dentro = [p for p in s["base"]["serie"] if t0 + 1 <= p["min"] <= t1 - 1]
    vacia_sil = sum(p["vel_arista_kmh"] is None or p["vel_arista_kmh"] < 5 for p in dentro)
    vacia_ds = sum(not np.isfinite(ds.get(int(p["min"]), np.nan)) or ds.get(int(p["min"])) < 5 for p in dentro)
    assert vacia_sil >= 0.8 * len(dentro) and vacia_ds >= 0.8 * len(dentro), (vacia_sil, vacia_ds, len(dentro))


@pytest.mark.grupo("sil")
def test_sil_detectores_como_el_dataset():
    """Lo que el AID lee en linea dentro de SUMO tiene la misma escala y variabilidad que los detectores del dataset
    (si no, el modelo veria datos distintos a los de su entrenamiento)."""
    s = sil(); inc = s["incidente"]
    p = R / f"sil_matrices_{inc['inc_id']}_sil_base.npz"
    if not p.exists():
        pytest.skip("falta la matriz del SIL")
    A = aid_mod()
    z = np.load(p); m = A.matrices_dia(inc["dia"], "gt")
    assert list(z["edges"]) == list(m["edges"])
    lo = int(z["desde"]) + s.get("calentamiento_min", 30); hi = int(inc["t_ini_min"]) - 2      # tras el calentamiento, antes del incidente
    a = z["vel"][:, lo - int(z["desde"]):hi - int(z["desde"])].astype(np.float32); b = m["vel"][:, lo:hi]
    assert abs(np.nanmean(a) - np.nanmean(b)) < 0.05, (np.nanmean(a), np.nanmean(b))
    assert 0.7 < np.nanmean(np.nanstd(a, 1)) / np.std(b, 1).mean() < 1.3
    for k in ("flujo", "ocup"):
        a = z[k][:, lo - int(z["desde"]):hi - int(z["desde"])].astype(np.float32); b = m[k][:, lo:hi]
        assert 0.7 < np.nanmean(a) / np.mean(b) < 1.3, (k, np.nanmean(a), np.mean(b))


# ------------------------------------------------------------------ tablero
@pytest.mark.grupo("dashboard")
def test_dashboard_coherente_con_resultados():
    """Las cifras que muestra el tablero son las mismas de los archivos de resultados."""
    D = json.load(open(DASH / "datos.json"))
    for f in ("gt", "obs"):
        assert D["metricas"]["pred"][f]["test"]["modelo"]["+60min"] == j(f"metricas_prediccion_{f}.json")["test"]["modelo"]["+60min"]
        a = j(f"metricas_aid_{f}.json")
        assert D["metricas"]["aid"][f]["test_incidentes"]["DR_con_impacto"] == a["test_incidentes"]["DR_con_impacto"]
        assert D["metricas"]["aid"][f]["seleccion_en_val"]["umbral"] == a["seleccion_en_val"]["umbral"]
    assert D["sil"]["comparacion_pct"] == sil()["comparacion_pct"]


@pytest.mark.grupo("dashboard")
def test_dashboard_binarios_coherentes():
    """Los binarios del tablero: la velocidad real coincide con la del test y la marca de 'hay autos' es 0/1."""
    D = json.load(open(DASH / "datos.json")); N = len(D["celdas"])
    a = np.fromfile(DASH / "pred_gt.bin", np.uint8).reshape(288, N, 7)
    assert set(np.unique(a[..., 6])) <= {0, 1}
    z = np.load(R / "pred_test_gt.npz")
    di = list(z["dias"]).index(D["meta"]["dia"])
    assert np.array_equal(a[..., 0], z["Y"][di])


@pytest.mark.grupo("dashboard")
def test_dashboard_dias_seleccionables():
    """Cada dia de test que se puede elegir en el tablero tiene sus datos completos y coinciden con el test."""
    D = json.load(open(DASH / "datos.json")); N = len(D["celdas"])
    z = np.load(R / "pred_test_gt.npz")
    assert sorted(f["dia"] for f in D["dias"]) == sorted(int(x) for x in z["dias"])
    for f in D["dias"]:
        d = f["dia"]; j = json.load(open(DASH / "dias" / f"d{d}.json"))
        assert j["meta"]["dia"] == d and j["ficha"]["semana"] in ("lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo")
        a = np.fromfile(DASH / "dias" / f"d{d}_pred_gt.bin", np.uint8).reshape(288, N, 7)
        assert np.array_equal(a[..., 0], z["Y"][list(z["dias"]).index(d)])
        for m in ("gt", "obs"):
            assert (DASH / "dias" / f"d{d}_aid_{m}.bin").stat().st_size == len(j["aid"]["edges"][m]) * 1440
        assert f["error_60"]["gt"]["ia"] < f["error_60"]["gt"]["ha"]   # la IA gana al promedio historico todos los dias


def _puerto_libre():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p


@pytest.fixture(scope="module")
def servidor():
    p = _puerto_libre()
    h = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(DASH))
    http.server.SimpleHTTPRequestHandler.log_message = lambda *a, **k: None
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", p), h)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{p}/"
    srv.shutdown()


VISTAS = ["#inicio", "#ciudad", "?t=08:00&capa=pred&celda=592#ciudad", "?t=08:00&capa=error#ciudad",
          "?fuente=obs&t=08:00&capa=futuro#ciudad", "?t=12:10#aid", "?t=12:10&conj=test_incidentes&fuente=obs#aid",
          "?t=14:20#sil", "?paso=3#como", "?paso=4#como", "#pruebas",
          "?dia=38&t=09:15&capa=pred#ciudad", "?dia=41&t=12:10&fuente=obs#aid"]


@pytest.mark.grupo("dashboard")
@pytest.mark.parametrize("vista", VISTAS)
def test_dashboard_carga_sin_errores(servidor, vista, tmp_path):
    """Cada vista del tablero se abre en Chrome sin errores de JavaScript y dibuja contenido."""
    chrome = shutil.which("google-chrome") or shutil.which("chromium")
    if not chrome:
        pytest.skip("no hay Chrome")
    url = servidor + ("?captura&" + vista[1:] if vista.startswith("?") else "?captura" + vista)
    shot = tmp_path / "v.png"
    r = subprocess.run([chrome, "--headless=new", "--no-sandbox", f"--user-data-dir={tmp_path / 'perfil'}", "--window-size=1400,850",
                        "--virtual-time-budget=6000", "--enable-logging=stderr", "--v=0", f"--screenshot={shot}", url],
                       capture_output=True, text=True, timeout=90)
    errores = [l for l in r.stderr.splitlines() if "CONSOLE" in l and re.search(r"Uncaught|Error|No se pudieron", l)]
    assert not errores, errores[:2]
    from PIL import Image
    im = np.asarray(Image.open(shot).convert("L"), np.float32)
    assert im.std() > 12, "la vista sale casi vacia"


@pytest.mark.grupo("dashboard")
def test_capa_error_solo_compara_celdas_con_autos():
    """En la capa de error, las celdas sin autos dentro de 1 h no se pintan (su 'velocidad real' es un relleno)."""
    html = (DASH / "index.html").read_text()
    rama = re.search(r'else \{const p=pv\(f,paso,i,5\);([^\n]*)colorErr', html)
    assert rama and "pv(f,pasoF,i,6)" in rama.group(1)


# ------------------------------------------------------------------ documentos
def _texto(nombre):
    p = INF / nombre
    if not p.exists():
        pytest.skip(f"falta {nombre}")
    return p.read_text()


def _es(x, n=1):
    return f"{x:.{n}f}".replace(".", ",")


@pytest.mark.grupo("documentos")
@pytest.mark.parametrize("doc", ["informe.tex", "diapositivas.tex", "guion.tex"])
def test_documentos_citan_las_cifras_reales(doc):
    """El informe, las diapositivas y el guion usan las cifras de los resultados (MAE, DR, FAR, SIL)."""
    t = _texto(doc)
    pg, po = j("metricas_prediccion_gt.json"), j("metricas_prediccion_obs.json")
    ag = j("metricas_aid_gt.json"); s = sil()
    cifras = [_es(pg["test"]["modelo"]["+60min"]["MAE"], 2), _es(po["test"]["modelo"]["+60min"]["MAE"], 2),
              _es(abs(s["comparacion_pct"]["perdida_afectados"])), _es(abs(s["comparacion_pct"]["perdida_media_red"])),
              f"{ag['test_incidentes']['DR_con_impacto']:.0f}", _es(ag["test_incidentes"]["FAR"])]
    def esta(c):
        if c in t:
            return True
        try:
            return f"{round(float(c.replace(',', '.'))):d}" in t       # el guion hablado redondea
        except ValueError:
            return False
    faltan = [c for c in cifras if not esta(c)]
    assert not faltan, f"no aparecen o no coinciden: {faltan}"


@pytest.mark.grupo("documentos")
def test_documentos_conteo_de_pruebas_automatico():
    """El informe y las diapositivas toman el numero de pruebas del ultimo banco (informe/pruebas.tex), sin escribirlo a mano."""
    for doc in ("informe.tex", "diapositivas.tex"):
        t = _texto(doc)
        assert "\\input{pruebas.tex}" in t and "\\PruebasOk" in t, f"{doc} no usa el conteo automatico"
        assert not re.search(r"(Pasan|pasan)\s+\d+\s+de\s+\d+", t), f"{doc} tiene el conteo escrito a mano"
    m = (INF / "pruebas.tex")
    if m.exists():
        casos = json.load(open(AQUI / "pruebas" / "informe.json"))["casos"]
        assert f"\\newcommand{{\\PruebasTotal}}{{{len(casos)}}}" in m.read_text()


@pytest.mark.grupo("documentos")
def test_figuras_al_dia():
    """Las figuras del informe son posteriores a los resultados de los que salen."""
    figs = list((INF / "figuras").glob("*.pdf"))
    if not figs:
        pytest.skip("sin figuras")
    ultima = max(p.stat().st_mtime for p in R.glob("metricas_*.json"))
    viejas = [p.name for p in figs if p.stat().st_mtime < ultima]
    assert not viejas, viejas


@pytest.mark.grupo("documentos")
@pytest.mark.parametrize("doc", ["informe.tex", "diapositivas.tex", "guion.tex"])
def test_documentos_compilan(doc, tmp_path):
    """Los documentos LaTeX compilan sin errores."""
    tec = shutil.which("tectonic")
    if not tec:
        pytest.skip("no hay tectonic")
    r = subprocess.run([tec, "-C", "-b", "https://data1.fullyjustified.net/tlextras-2022.0r0.tar", "--outdir", str(tmp_path), doc],
                       cwd=INF, capture_output=True, text=True, timeout=400)
    assert r.returncode == 0, r.stderr[-400:]
