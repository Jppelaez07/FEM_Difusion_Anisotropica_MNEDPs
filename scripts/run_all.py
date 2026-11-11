"""Corre todos los scripts del proyecto en orden y resume el resultado.

La salida de cada script se guarda en results/logs/<script>.txt.
Uso: python scripts/run_all.py
"""

import os
import subprocess
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
DIR_LOGS = RAIZ / "results" / "logs"

SCRIPTS = [
    "01_malla.py",            # malla y verificaciones de M y K
    "02_prueba_rapida.py",    # conservación de masa y elongación
    "03_mms.py",              # solución manufacturada: convergencia espacial y temporal
    "04_gauss.py",            # gaussiana analítica anisotrópica
    "05_estabilidad.py",      # estabilidad del explícito y oscilaciones de Crank-Nicolson
    "10_casos.py",            # escenarios de simulación
    "11_figuras_casos.py",    # figuras de los escenarios (usa 10_casos)
    "12_sensibilidad.py",     # sensibilidad a la anisotropía y difusión numérica
    "20_animaciones.py",      # animaciones (usa 10_casos)
]


def formato_tiempo(s):
    return f"{s:6.1f} s" if s < 120 else f"{s / 60:6.1f} min"


def main():
    DIR_LOGS.mkdir(parents=True, exist_ok=True)
    entorno = dict(os.environ, PYTHONIOENCODING="utf-8")
    resultados = []
    inicio = time.time()
    for script in SCRIPTS:
        print(f"Corriendo {script} ...", flush=True)
        t0 = time.time()
        log = DIR_LOGS / f"{Path(script).stem}.txt"
        with open(log, "w", encoding="utf-8") as fh:
            proc = subprocess.run([sys.executable, str(RAIZ / "scripts" / script)], cwd=RAIZ,
                                  env=entorno, stdout=fh, stderr=subprocess.STDOUT)
        dt = time.time() - t0
        ok = proc.returncode == 0
        resultados.append((script, ok, dt))
        print(f"  {'OK   ' if ok else 'FALLO'} {formato_tiempo(dt)}   (salida en {log.relative_to(RAIZ)})",
              flush=True)

    print("\nRESUMEN")
    for script, ok, dt in resultados:
        print(f"  {script:<22s} {'OK' if ok else 'FALLO':<6s} {formato_tiempo(dt)}")
    n_fallos = sum(not ok for _, ok, _ in resultados)
    print(f"  {'Total':<22s} {'':<6s} {formato_tiempo(time.time() - inicio)}")
    print(f"\n{len(resultados) - n_fallos} de {len(resultados)} scripts terminaron bien."
          + ("" if n_fallos == 0 else f" Revise los logs de los {n_fallos} con FALLO."))
    return 0 if n_fallos == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
