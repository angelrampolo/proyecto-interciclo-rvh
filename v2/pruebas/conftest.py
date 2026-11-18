"""Guarda el resultado de cada prueba en pruebas/informe.json para mostrarlo en el dashboard."""
import json, time
from pathlib import Path

CASOS = []


def pytest_configure(config):
    config.addinivalue_line("markers", "grupo(nombre): grupo de la prueba")


def pytest_runtest_logreport(report):
    if report.when != "call" and not (report.when == "setup" and report.skipped):
        return
    item = report.nodeid.split("::")[-1]
    estado = "ok" if report.passed else "aviso" if report.skipped else "falla"
    detalle = ""
    if report.failed:
        detalle = str(report.longrepr.reprcrash.message)[:220] if hasattr(report.longrepr, "reprcrash") else ""
    elif report.skipped:
        detalle = str(report.longrepr[2])[:200] if isinstance(report.longrepr, tuple) else ""
    CASOS.append({"id": item, "estado": estado, "detalle": detalle, "_nodo": report.nodeid})


def pytest_collection_modifyitems(items):
    for it in items:
        m = it.get_closest_marker("grupo")
        doc = (it.function.__doc__ or it.name).strip().split("\n")[0]
        param = it.callspec.params.get("modo") if hasattr(it, "callspec") else None
        if param:
            doc += " [" + ("sensores" if param == "gt" else "V2X")
            conj = it.callspec.params.get("conj")
            if conj:
                doc += ", " + ("días normales" if conj == "test" else "días con incidentes")
            doc += "]"
        elif hasattr(it, "callspec") and it.callspec.params:
            doc += " [" + ", ".join(str(v) for v in it.callspec.params.values()) + "]"
        META[it.nodeid] = {"grupo": m.args[0] if m else "otros", "titulo": doc}


META = {}


def pytest_sessionfinish(session, exitstatus):
    # solo una corrida completa del banco actualiza el informe (una corrida parcial con -k o un archivo no lo pisa)
    op = session.config.option
    completa = (all(Path(a).is_dir() for a in session.config.args) and not op.keyword and not op.markexpr
                and not getattr(op, "deselect", None) and not getattr(op, "maxfail", 0))
    if not completa:
        return
    casos = []
    for c in CASOS:
        meta = META.get(c.pop("_nodo"), {})
        casos.append({**meta, **c})
    json.dump({"fecha": time.strftime("%Y-%m-%d %H:%M"), "casos": casos},
              open(Path(__file__).parent / "informe.json", "w"), indent=1, ensure_ascii=False)
    ok = sum(c["estado"] == "ok" for c in casos)
    falla = sum(c["estado"] == "falla" for c in casos)
    (Path(__file__).parent.parent / "informe" / "pruebas.tex").write_text(
        "% generado por pruebas/conftest.py: no editar a mano\n"
        f"\\newcommand{{\\PruebasOk}}{{{ok}}}\n\\newcommand{{\\PruebasTotal}}{{{len(casos)}}}\n\\newcommand{{\\PruebasFallan}}{{{falla}}}\n")
