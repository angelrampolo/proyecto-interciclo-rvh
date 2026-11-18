"""Banco de pruebas del Proyecto Interciclo.

Corre:   cd v2 && python -m pytest pruebas/ -q
Cada prueba revisa una regla del proyecto (fugas de informacion, lineas base, metas del docente,
validacion en SUMO, dashboard). Los resultados quedan en pruebas/informe.json para el dashboard.
"""
import json, re, subprocess, sys
from pathlib import Path
import numpy as np, pandas as pd, pytest

AQUI = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(AQUI))
import comun as C  # noqa: E402

R = C.SALIDA


def j(nombre):
    p = R / nombre
    if not p.exists():
        pytest.skip(f"falta {nombre}: corre primero el script que lo genera")
    return json.load(open(p))


# ------------------------------------------------------------------ datos y fugas
@pytest.mark.grupo("datos")
def test_division_por_dias_sin_cruces():
    """Train, val, test y estres son dias distintos; el estres nunca se usa para entrenar."""
    d = C.dias()
    grupos = {s: set(d.index[d.split == s]) for s in ("train", "val", "test", "test_estres")}
    for a in grupos:
        for b in grupos:
            if a < b:
                assert not grupos[a] & grupos[b], (a, b)
    assert grupos["train"] == set(range(1, 29)) and grupos["test"] == set(range(36, 43))


@pytest.mark.grupo("datos")
def test_ventanas_no_cruzan_dias():
    """Cada ventana tiene 1 h de historia y su objetivo a +60 min dentro del mismo dia, despues de las 05:00."""
    for d, t in C.indice_ventanas(3):
        assert t - C.HIST + 1 >= 0 and t + C.H < C.PASOS
        assert (t + C.H) * 300 >= C.DESDE_HORA * 3600


@pytest.mark.grupo("datos")
def test_entradas_sin_etiquetas():
    """Ningun canal de entrada usa columnas de etiqueta ni de configuracion del simulador."""
    import entrenar_aid as A
    prohibidas = ("incidente", "inc_id", "congestion", "critico", "factor", "semilla", "reintentos")
    for c in C.CANALES + A.RASGOS:
        assert not any(p in c for p in prohibidas), c


@pytest.mark.grupo("datos")
def test_sin_autos_no_es_atasco():
    """Si no pasa ningun auto, la velocidad se rellena con la velocidad libre, nunca con 0."""
    X, Y, M, _ = C.cargar_dias([36], "obs")
    vacio = X[..., 4] == 0
    assert vacio.any()
    assert np.allclose(X[..., 0][vacio], 1.0)


@pytest.mark.grupo("datos")
def test_correccion_por_penetracion():
    """Los conteos V2X divididos por la penetracion se parecen a los reales tambien con 30 % y 70 %."""
    for d in (45, 46):
        Xo, *_ = C.cargar_dias([d], "obs")
        Xg, *_ = C.cargar_dias([d], "gt")
        m = (Xg[..., 2] > 5)
        r = np.median(Xo[..., 2][m] / Xg[..., 2][m])
        assert 0.8 < r < 1.25, (d, r)


@pytest.mark.grupo("datos")
def test_promedio_historico_solo_con_train():
    """El promedio historico guardado en el modelo es el de train (no se filtra informacion del test)."""
    import torch
    p = R / "modelo_gcngru_gt.pt"
    if not p.exists():
        pytest.skip("falta el modelo")
    HA = torch.load(p, weights_only=False)["HA"]
    _, Y, _, meta = C.cargar_dias(C.ids("train")[:28], "gt")
    assert np.allclose(HA, C.promedio_historico(Y, meta), atol=1e-3)


# ------------------------------------------------------------------ prediccion a 1 h
@pytest.mark.grupo("prediccion")
@pytest.mark.parametrize("modo", ["gt", "obs"])
def test_modelo_supera_lineas_base(modo):
    """A +60 min la IA tiene menor MAE y RMSE que la persistencia y que el promedio historico (test)."""
    r = j(f"metricas_prediccion_{modo}.json")["test"]
    for base in ("persistencia", "promedio_historico"):
        assert r["modelo"]["+60min"]["MAE"] < r[base]["+60min"]["MAE"]
        assert r["modelo"]["+60min"]["RMSE"] < r[base]["+60min"]["RMSE"]


@pytest.mark.grupo("prediccion")
@pytest.mark.parametrize("modo", ["gt", "obs"])
def test_meta_error_1h(modo):
    """Meta del proyecto: MAE < 3 km/h y MAPE < 10 % a 60 min en el test."""
    m = j(f"metricas_prediccion_{modo}.json")["test"]["modelo"]["+60min"]
    assert m["MAE"] < 3.0 and m["MAPE"] < 10.0, m


@pytest.mark.grupo("prediccion")
@pytest.mark.parametrize("modo", ["gt", "obs"])
def test_mejor_en_hora_pico_e_incidentes(modo):
    """La IA gana al promedio historico justo donde importa: hora pico, congestion y celdas con incidente."""
    r = j(f"metricas_prediccion_{modo}.json")["test"]
    for k in ("hora_pico", "congestion_real(v<0.55vff)", "celdas_con_incidente"):
        assert r["modelo"]["desglose_60min"][k]["MAE"] < r["promedio_historico"]["desglose_60min"][k]["MAE"], k


@pytest.mark.grupo("prediccion")
@pytest.mark.parametrize("modo", ["gt", "obs"])
def test_robustez_estres(modo):
    """En los 4 dias de estres (mas demanda, lluvia, 30 % y 70 % V2X) sigue ganandole al promedio historico."""
    r = j(f"metricas_prediccion_{modo}.json")["estres"]
    for d, v in r.items():
        assert v["modelo"]["MAE"] < v["promedio_historico"]["MAE"], d


@pytest.mark.grupo("prediccion")
def test_metricas_son_del_test():
    """Las metricas reportadas salen de los 7 dias de test completos (no de validacion)."""
    r = j("metricas_prediccion_gt.json")
    n = r["test"]["modelo"]["+60min"]["n"]
    nv = r["validacion"]["modelo"]["+60min"]["n"]
    assert n > 0 and n != nv
    z = np.load(R / "pred_test_gt.npz")
    assert list(z["dias"]) == list(range(36, 43))


@pytest.mark.grupo("prediccion")
def test_prediccion_reproducible():
    """Cargar el modelo guardado y predecir de nuevo da lo mismo que lo reportado."""
    import torch, entrenar_prediccion as E
    p = R / "modelo_gcngru_gt.pt"
    if not p.exists():
        pytest.skip("falta el modelo")
    ck = torch.load(p, weights_only=False)
    m = E.GCNGRU(len(C.celdas()), ck["ij"]); m.load_state_dict(ck["estado"]); m.eval()
    d = E.Datos([36], "gt", ck["HA"]); E.normalizar(d, ck["mu"], ck["sd"])
    d.idx = d.idx[100:104]; d.tiempo = d.tiempo[100:104]
    P = E.predecir(m, d)
    z = np.load(R / "pred_test_gt.npz")
    assert np.abs(np.clip(P, 0, 120) - z["P"][100:104]).max() <= 1.0


# ------------------------------------------------------------------ AID
@pytest.mark.grupo("aid")
@pytest.mark.parametrize("conj", ["test", "test_incidentes"])
@pytest.mark.parametrize("modo", ["gt", "obs"])
def test_aid_far(modo, conj):
    """Meta del curso: FAR <= 3-5 %. El umbral se elige (sin test) para FAR <= 3 %; aqui se verifica el limite del curso, 5 %."""
    t = j(f"metricas_aid_{modo}.json")[conj]
    assert t["FAR"] <= 5.0, t["FAR"]


@pytest.mark.grupo("aid")
@pytest.mark.parametrize("conj", ["test", "test_incidentes"])
@pytest.mark.parametrize("modo", ["gt", "obs"])
def test_aid_mejor_que_el_azar(modo, conj):
    """Con la misma tasa de falsas alarmas, el AID detecta mas incidentes con impacto que un detector al azar."""
    t = j(f"metricas_aid_{modo}.json")[conj]
    assert t["DR_con_impacto"] > t["azar_misma_FAR"]["DR_con_impacto"], (t["DR_con_impacto"], t["azar_misma_FAR"]["DR_con_impacto"])


@pytest.mark.grupo("aid")
@pytest.mark.parametrize("conj", ["test", "test_incidentes"])
@pytest.mark.parametrize("modo", ["gt", "obs"])
def test_aid_meta_deteccion(modo, conj):
    """Meta del docente: DR >= 85 % y MTTD < 5 min (sobre los incidentes que cambian el trafico)."""
    t = j(f"metricas_aid_{modo}.json")[conj]
    assert t["DR_con_impacto"] >= 85 and (t["MTTD_con_impacto_min"] or 99) < 5, (t["DR_con_impacto"], t["MTTD_con_impacto_min"])


@pytest.mark.grupo("aid")
def test_aid_umbral_elegido_en_validacion():
    """El umbral y la persistencia se eligieron con validacion, no mirando el test."""
    r = j("metricas_aid_gt.json")
    sel = r["seleccion_en_val"]
    assert any(c["umbral"] == sel["umbral"] and c["persistencia"] == sel["persistencia"] for c in r["curva_val"])
    if "umbral_lluvia" in sel:
        assert any(c["umbral"] == sel["umbral_lluvia"] and c.get("clima") == "lluvia" for c in r["curva_val"])


# ------------------------------------------------------------------ SIL
def sil():
    ps = sorted(R.glob("sil_*.json"))
    if not ps:
        pytest.skip("falta correr sil_sumo.py")
    return json.load(open(ps[0]))


@pytest.mark.grupo("sil")
def test_sil_sale_de_sumo():
    """El SIL corre en SUMO con miles de viajes reales y la misma demanda en ambas corridas."""
    s = sil()
    assert s["base"]["viajes"] > 5000 and abs(s["base"]["viajes"] - s["ia"]["viajes"]) / s["base"]["viajes"] < 0.02
    assert len(s["base"]["serie"]) > 60


@pytest.mark.grupo("sil")
def test_sil_ia_mejora():
    """Con el aviso V2X baja la perdida de tiempo de los autos que iban hacia el incidente."""
    s = sil()
    assert s["ia"]["autos_avisados_denm"] > 0
    assert s["ia"]["afectados_perdida_media_s"] < s["base"]["afectados_perdida_media_s"]


@pytest.mark.grupo("sil")
def test_sil_no_empeora_la_red():
    """El desvio no traslada el problema: la perdida media de toda la red no sube mas de 1 %."""
    s = sil()
    assert s["ia"]["perdida_media_s"] <= s["base"]["perdida_media_s"] * 1.01


# ------------------------------------------------------------------ dashboard
@pytest.mark.grupo("dashboard")
def test_dashboard_datos():
    """El dashboard tiene sus datos y los binarios tienen el tamano esperado."""
    D = AQUI / "dashboard"
    datos = json.load(open(D / "datos.json"))
    N = len(datos["celdas"])
    for f in ("gt", "obs"):
        assert (D / f"pred_{f}.bin").stat().st_size == 288 * N * 7
        assert (D / f"aid_{f}.bin").stat().st_size == len(datos["aid"]["edges"][f]) * 1440


@pytest.mark.grupo("dashboard")
def test_dashboard_javascript_valido():
    """El JavaScript del dashboard compila (node --check)."""
    html = (AQUI / "dashboard" / "index.html").read_text()
    js = re.search(r"<script>(.*)</script>", html, re.S).group(1)
    tmp = AQUI / "pruebas" / "_tmp.js"
    tmp.write_text(js)
    r = subprocess.run(["node", "--check", str(tmp)], capture_output=True, text=True)
    tmp.unlink()
    assert r.returncode == 0, r.stderr


@pytest.mark.grupo("dashboard")
def test_sin_numeros_escritos_a_mano():
    """Las metricas del dashboard vienen de los JSON de resultados, no de constantes en el HTML."""
    html = (AQUI / "dashboard" / "index.html").read_text()
    for numero in ("88.24", "2.89", "54.55", "62.76", "4.79"):
        assert numero not in html
