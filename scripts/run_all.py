"""Corre todos los scripts del proyecto en orden y resume el resultado.

Uso:
  python scripts/run_all.py            corrida completa (cerca de una hora), resultados en results/
  python scripts/run_all.py --rapido   prueba en pocos minutos, resultados en results_rapido/

El modo rápido no corre las animaciones, omite las mallas de h = 0.05 en
12_sensibilidad.py, la de N = 401 en 06_pakmor.py y 07_anillo.py, la de N = 257
en 08_sovinec.py, la de 801 x 201 en 13_halo.py y la de 561 x 241 en 14_snr.py, y usa
T = 100 en 10_casos.py. Sus resultados son reducidos y no deben usarse en el informe.
La salida de cada script se guarda en <carpeta>/logs/.
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]

SCRIPTS = [
    "01_malla.py",            # malla y verificaciones de M y K
    "02_prueba_rapida.py",    # conservación de masa y elongación
    "03_mms.py",              # solución manufacturada: convergencia espacial y temporal
    "04_gauss.py",            # gaussiana analítica anisotrópica
    "05_estabilidad.py",      # estabilidad del explícito y oscilaciones de Crank-Nicolson
    "06_pakmor.py",           # prueba 3.1 de Pakmor et al. (2016)
    "07_anillo.py",           # prueba 3.2 de Pakmor et al. (2016): cuña en campo circular
    "08_sovinec.py",          # prueba 3.4 de Pakmor et al. (2016): prueba de Sovinec
    "10_casos.py",            # escenarios de simulación
    "11_figuras_casos.py",    # figuras de los escenarios (usa 10_casos)
    "12_sensibilidad.py",     # sensibilidad a la anisotropía y difusión numérica
    "13_halo.py",             # leaky box en (x, z), Al-Zetoun y Achterberg (2018)
    "14_snr.py",              # rayos cósmicos de un remanente de supernova, Nava y Gabici (2013)
    "15_simulaciones.py",     # simulaciones adicionales del remanente (usa 14_snr)
    "20_animaciones.py",      # animaciones (usa 10_casos)
]
SIN_MODO_RAPIDO = {"20_animaciones.py"}


def formato_tiempo(s):
    return f"{s:6.1f} s" if s < 120 else f"{s / 60:6.1f} min"


def main():
    parser = argparse.ArgumentParser(description="Corre todos los scripts del proyecto.")
    parser.add_argument("--rapido", action="store_true",
                        help="prueba en pocos minutos; resultados reducidos en results_rapido/")
    args = parser.parse_args()

    entorno = dict(os.environ, PYTHONIOENCODING="utf-8")
    entorno.pop("MN_RAPIDO", None)
    scripts = SCRIPTS
    carpeta = RAIZ / "results"
    if args.rapido:
        entorno["MN_RAPIDO"] = "1"
        scripts = [s for s in SCRIPTS if s not in SIN_MODO_RAPIDO]
        carpeta = RAIZ / "results_rapido"
        print("MODO RÁPIDO: resultados reducidos en results_rapido/, no usar en el informe.")
        print("Se omiten: " + ", ".join(sorted(SIN_MODO_RAPIDO)) + "\n", flush=True)
    dir_logs = carpeta / "logs"
    dir_logs.mkdir(parents=True, exist_ok=True)

    resultados = []
    inicio = time.time()
    for script in scripts:
        print(f"Corriendo {script} ...", flush=True)
        t0 = time.time()
        log = dir_logs / f"{Path(script).stem}.txt"
        with open(log, "w", encoding="utf-8") as fh:
            proc = subprocess.run([sys.executable, str(RAIZ / "scripts" / script)], cwd=RAIZ,
                                  env=entorno, stdout=fh, stderr=subprocess.STDOUT)
        dt = time.time() - t0
        ok = proc.returncode == 0
        resultados.append((script, ok, dt))
        print(f"  {'OK   ' if ok else 'FALLO'} {formato_tiempo(dt)}   (salida en {log.relative_to(RAIZ)})",
              flush=True)

    print("\nRESUMEN" + (" (MODO RÁPIDO)" if args.rapido else ""))
    for script, ok, dt in resultados:
        print(f"  {script:<22s} {'OK' if ok else 'FALLO':<6s} {formato_tiempo(dt)}")
    n_fallos = sum(not ok for _, ok, _ in resultados)
    print(f"  {'Total':<22s} {'':<6s} {formato_tiempo(time.time() - inicio)}")
    print(f"\n{len(resultados) - n_fallos} de {len(resultados)} scripts terminaron bien."
          + ("" if n_fallos == 0 else f" Revise los logs de los {n_fallos} con FALLO."))
    return 0 if n_fallos == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
