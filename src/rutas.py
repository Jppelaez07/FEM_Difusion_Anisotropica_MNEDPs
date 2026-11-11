"""Carpeta de resultados y modo rápido.

El modo rápido se activa con la variable de entorno MN_RAPIDO=1
(scripts/run_all.py --rapido la define). En ese modo los resultados se guardan
en results_rapido/ para no sobrescribir los completos de results/.
"""

import os
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
RAPIDO = os.environ.get("MN_RAPIDO", "") == "1"
RESULTADOS = RAIZ / ("results_rapido" if RAPIDO else "results")
AVISO = "MODO RÁPIDO: resultados reducidos, no usar en el informe"


def avisar():
    """Imprime el aviso del modo rápido (si está activo)."""
    if RAPIDO:
        print(AVISO + "\n", flush=True)
