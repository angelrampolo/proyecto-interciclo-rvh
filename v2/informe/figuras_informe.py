"""Figuras del informe y de las diapositivas, todas leidas de resultados/*.json (nada escrito a mano)."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

AQUI = Path(__file__).resolve().parent
R = AQUI.parent / "resultados"
OUT = AQUI / "figuras"; OUT.mkdir(exist_ok=True)
j = lambda n: json.load(open(R / n))
plt.rcParams.update({"font.family": "Lato", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": .25, "axes.titleweight": "bold", "figure.dpi": 200,
                     "savefig.bbox": "tight", "axes.titlesize": 11})
C_IA, C_HA, C_PER, C_AZ, C_V2X = "#0f8f7f", "#e0962a", "#8a94a3", "#b8c0cc", "#2f6fd6"
H = ["+15min", "+30min", "+45min", "+60min"]
Hx = [15, 30, 45, 60]


def guardar(fig, nombre):
    """Toda figura lleva la marca de que sus datos son sinteticos (simulacion SUMO), no medidas de campo."""
    fig.text(0.995, 0.005, "Datos sintéticos (SUMO)", ha="right", va="bottom", fontsize=7, color="#8a94a3", style="italic")
    fig.savefig(OUT / nombre); plt.close(fig)

pg, po = j("metricas_prediccion_gt.json"), j("metricas_prediccion_obs.json")

# 1. error por horizonte
fig, ax = plt.subplots(figsize=(6.2, 3.2))
ax.plot(Hx, [pg["test"]["persistencia"][h]["MAE"] for h in H], "o-", c=C_PER, label="Persistencia")
ax.plot(Hx, [pg["test"]["promedio_historico"][h]["MAE"] for h in H], "o-", c=C_HA, label="Promedio histórico")
ax.plot(Hx, [pg["test"]["modelo"][h]["MAE"] for h in H], "o-", c=C_IA, lw=2.2, label="GCN-GRU (sensores)")
ax.plot(Hx, [po["test"]["modelo"][h]["MAE"] for h in H], "s--", c=C_V2X, lw=1.8, label="GCN-GRU (V2X 50 %)")
ax.axhline(3, ls=":", c="#c0392b", lw=1); ax.text(15.5, 3.05, "meta MAE < 3 km/h", color="#c0392b", fontsize=8)
ax.set_xticks(Hx); ax.set_xticklabels(H); ax.set_ylabel("MAE (km/h)"); ax.set_ylim(0, None)
ax.set_title("Error de predicción en el test (días 36–42)"); ax.legend(fontsize=8, ncol=2, loc="lower right")
guardar(fig, "pred_horizonte.pdf")

# 2. desglose a +60
cats = [("hora_pico", "Hora pico"), ("valle", "Valle"), ("corredores", "Corredores"), ("celdas_con_incidente", "Con incidente"),
        ("congestion_real(v<0.55vff)", "Congestión"), ("dia_sabado", "Sábado"), ("dia_domingo", "Domingo")]
fig, ax = plt.subplots(figsize=(6.6, 3.0)); x = np.arange(len(cats)); w = .27
for k, (src, col, lab) in enumerate([("promedio_historico", C_HA, "Promedio histórico"), ("modelo", C_IA, "GCN-GRU sensores")]):
    ax.bar(x + (k - .5) * w, [pg["test"][src]["desglose_60min"][c]["MAE"] for c, _ in cats], w, color=col, label=lab)
ax.bar(x + 1.5 * w - .0, [po["test"]["modelo"]["desglose_60min"][c]["MAE"] for c, _ in cats], w, color=C_V2X, label="GCN-GRU V2X 50 %")
ax.set_xticks(x + .25 * w); ax.set_xticklabels([n for _, n in cats], fontsize=8.5); ax.set_ylabel("MAE a +60 min (km/h)")
ax.set_title("Error a +60 min según el contexto"); ax.legend(fontsize=8)
guardar(fig, "pred_desglose.pdf")

# 3. estres
dias = ["dia_43", "dia_44", "dia_45", "dia_46"]; et = ["43\n+demanda", "44\nlluvia", "45\n30 % V2X", "46\n70 % V2X"]
fig, ax = plt.subplots(figsize=(5.4, 2.9)); x = np.arange(4); w = .27
ax.bar(x - w, [pg["estres"][d]["promedio_historico"]["MAE"] for d in dias], w, color=C_HA, label="Promedio histórico")
ax.bar(x, [pg["estres"][d]["modelo"]["MAE"] for d in dias], w, color=C_IA, label="GCN-GRU sensores")
ax.bar(x + w, [po["estres"][d]["modelo"]["MAE"] for d in dias], w, color=C_V2X, label="GCN-GRU V2X")
ax.set_xticks(x); ax.set_xticklabels(et, fontsize=8.5); ax.set_ylabel("MAE a +60 min (km/h)"); ax.set_title("Días de estrés (nunca vistos al entrenar)"); ax.legend(fontsize=8)
guardar(fig, "pred_estres.pdf")

# 4. curva de entrenamiento
fig, ax = plt.subplots(figsize=(5.0, 2.8))
for r, c, lab in ((pg, C_IA, "sensores"), (po, C_V2X, "V2X 50 %")):
    hh = r["historial"]; ax.plot([h["epoca"] for h in hh], [h["val_MAE"] for h in hh], "o-", c=c, label=f"validación ({lab})")
ax.set_xlabel("Época"); ax.set_ylabel("MAE val (km/h)"); ax.set_title("Entrenamiento: error en validación"); ax.legend(fontsize=8)
guardar(fig, "pred_entrenamiento.pdf")

# 5. AID: IA frente al azar
ag, ao = j("metricas_aid_gt.json"), j("metricas_aid_obs.json")
fig, axs = plt.subplots(1, 2, figsize=(6.8, 2.9), sharey=True)
for ax, conj, tit in ((axs[0], "test", "Días normales (36–42)"), (axs[1], "test_incidentes", "Días con incidentes (117–120)")):
    lab = ["Sensores", "V2X 50 %"]; x = np.arange(2); w = .36
    ia = [ag[conj]["DR_con_impacto"], ao[conj]["DR_con_impacto"]]; az = [ag[conj]["azar_misma_FAR"]["DR_con_impacto"], ao[conj]["azar_misma_FAR"]["DR_con_impacto"]]
    b1 = ax.bar(x - w / 2, ia, w, color=C_IA, label="AID"); b2 = ax.bar(x + w / 2, az, w, color=C_AZ, label="Azar, misma FAR")
    for b, v in list(zip(b1, ia)) + list(zip(b2, az)): ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.0f}", ha="center", fontsize=8)
    far = [ag[conj]["FAR"], ao[conj]["FAR"]]
    ax.set_xticks(x); ax.set_xticklabels([f"{l}\nFAR {f:.1f} %".replace(".", ",") for l, f in zip(lab, far)], fontsize=8.5)
    ax.axhline(85, ls=":", c="#c0392b", lw=1); ax.set_title(tit, fontsize=10); ax.set_ylim(0, 110)
axs[0].set_ylabel("DR, incidentes con impacto (%)"); axs[0].legend(fontsize=8, loc="lower left")
guardar(fig, "aid_azar.pdf")

# 6. AID: curva DR-FAR
fig, ax = plt.subplots(figsize=(5.2, 3.0))
for r, c, lab in ((ag, C_IA, "sensores"), (ao, C_V2X, "V2X 50 %")):
    cv = sorted(r["curva_test_incidentes"], key=lambda q: q["FAR"])
    ax.plot([q["FAR"] for q in cv], [q["DR_con_impacto"] for q in cv], "-", c=c, label=f"AID {lab}")
    ax.plot(r["test_incidentes"]["FAR"], r["test_incidentes"]["DR_con_impacto"], "o", c=c, ms=7, mec="k")
for r, c in ((ag, C_IA), (ao, C_V2X)):
    az = r["test_incidentes"]["azar_misma_FAR"]; ax.plot(az["FAR"], az["DR_con_impacto"], "X", c=c, ms=8, mec="k")
ax.plot([], [], "X", c=C_PER, mec="k", label="azar a la misma FAR")
ax.axvspan(3, 5, color="#2ecc71", alpha=.10); ax.text(3.1, 8, "meta FAR\n3–5 %", fontsize=8, color="#1e8449")
ax.set_xlim(0, 12); ax.set_ylim(0, 102); ax.set_xlabel("FAR (%)"); ax.set_ylabel("DR con impacto (%)")
ax.set_title("Curva DR–FAR, días con incidentes"); ax.legend(fontsize=8, loc="lower right")
guardar(fig, "aid_curva.pdf")

# 7. SIL: cola y comparacion
s = j("sil_inc119_3_gt.json")
fig, ax = plt.subplots(figsize=(6.2, 2.8))
for k, c, lab in (("base", C_PER, "Sin IA"), ("ia", C_IA, "Con IA + DENM")):
    se = [p for p in s[k]["serie"] if "cola_m_por_carril" in p]
    ax.plot([p["min"] / 60 for p in se], [p["cola_m_por_carril"] for p in se], c=c, lw=2, label=lab)
ti, tf = s["incidente"]["t_ini_min"] / 60, s["incidente"]["t_fin_min"] / 60
ax.axvspan(ti, tf, color="#e74c3c", alpha=.10, label="incidente activo")
ax.set_xlim(ti - .6, tf + .6); ax.set_xlabel("Hora del día"); ax.set_ylabel("Cola (m por carril)")
ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{int(v):02d}:{int(round(v % 1 * 60)):02d}"))
ax.set_title(f"Cola aguas arriba de {s['incidente']['inc_id']} (SUMO)"); ax.legend(fontsize=8)
guardar(fig, "sil_cola.pdf")

cp = s["comparacion_pct"]
items = [("perdida_afectados", "Pérdida de tiempo\nafectados"), ("tiempo_viaje_afectados", "Tiempo de viaje\nafectados"),
         ("perdida_media_red", "Pérdida media\nde toda la red"), ("vht_red", "VHT de\ntoda la red"), ("cola_max", "Cola\nmáxima")]
fig, ax = plt.subplots(figsize=(6.2, 2.7))
v = [cp[k] for k, _ in items]; b = ax.barh(range(len(v)), v, color=[C_IA if x < 0 else "#c0392b" for x in v])
for i, x in enumerate(v): ax.text(x - .8 if x < 0 else x + .3, i, f"{x:+.1f} %".replace(".", ","), va="center", ha="right" if x < 0 else "left", fontsize=8.5)
ax.set_yticks(range(len(v))); ax.set_yticklabels([n for _, n in items], fontsize=8.5); ax.axvline(0, c="k", lw=.8)
ax.invert_yaxis(); ax.set_xlabel("Cambio con IA + aviso V2X (%)"); ax.set_xlim(min(v) - 6, 4); ax.set_title("Bucle cerrado: efecto del aviso DENM")
guardar(fig, "sil_comparacion.pdf")
print("figuras listas:", sorted(p.name for p in OUT.glob("*.pdf")))
