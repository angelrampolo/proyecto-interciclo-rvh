"""Sirve el dashboard en http://localhost:8097 (solo en esta maquina)."""
import http.server, functools, webbrowser, sys
from pathlib import Path
PUERTO = int(sys.argv[1]) if len(sys.argv) > 1 else 8097
h = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(Path(__file__).resolve().parent / "dashboard"))
print(f"Dashboard: http://localhost:{PUERTO}/  (Ctrl+C para cerrar)")
try:
    webbrowser.open(f"http://localhost:{PUERTO}/")
except Exception:
    pass
http.server.ThreadingHTTPServer(("127.0.0.1", PUERTO), h).serve_forever()
