"""Corre los escenarios de src/casos.py e imprime una tabla resumen.

Guarda results/casos/<caso>.npz, results/casos/resumen.csv y results/casos/resumen.txt.
"""

import csv
import sys
import time
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from src import casos, postproc  # noqa: E402

CASOS_ISOLINEA = ["caso1", "caso2", "caso3"]
NIVEL_ISOLINEA = 0.1          # fracción de max n
TOL_NEUMANN = 1e-10


def main():
    casos.DIR_CASOS.mkdir(parents=True, exist_ok=True)
    lineas = []

    def reportar(texto=""):
        print(texto, flush=True)
        lineas.append(texto)

    resultados = {}
    for nombre in casos.CASOS:
        c = casos.config(nombre)
        t0 = time.time()
        resultados[nombre] = casos.correr_caso(c)
        print(f"  {nombre}: {len(resultados[nombre]['nodes'])} nodos, "
              f"{time.time() - t0:.1f} s", flush=True)
    print()

    # --- Tabla 1: métricas al final ---
    filas = []
    reportar("RESUMEN AL FINAL DE CADA CASO (t = T)")
    reportar(f"  {'caso':<16s} {'nodos':>6s} {'masa':>9s} {'flujo':>9s} {'<x^2>':>7s} {'<y^2>':>7s}"
             f" {'<xy>':>7s} {'aspecto':>7s} {'min n':>10s} {'max n':>7s} {'t cuasi-est.':>12s}"
             f" {'cambio rel.':>11s}")
    for nombre, r in resultados.items():
        tc = r["t_cuasi_estacionario"]
        fila = {
            "caso": nombre, "nodos": len(r["nodes"]),
            "masa": r["masa"][-1], "flujo": r["flujo_balance"][-1],
            "xx": r["mom_xx"][-1], "yy": r["mom_yy"][-1], "xy": r["mom_xy"][-1],
            "aspecto": r["razon_aspecto"][-1],
            "n_min": r["n_min"].min(), "n_max": r["n_max"][-1],
            "t_cuasi": tc, "cambio_final": r["cambio_rel"][-1],
        }
        filas.append(fila)
        reportar(f"  {nombre:<16s} {fila['nodos']:6d} {fila['masa']:9.3f} {fila['flujo']:9.4f}"
                 f" {fila['xx']:7.2f} {fila['yy']:7.2f} {fila['xy']:7.2f} {fila['aspecto']:7.2f}"
                 f" {fila['n_min']:10.2e} {fila['n_max']:7.4f}"
                 f" {('no alcanzado' if np.isnan(tc) else f'{tc:.2f}'):>12s} {fila['cambio_final']:11.2e}")
    reportar("  masa = 1^T M N; flujo = int Q - dM/dt (saliente); aspecto = sqrt(l1/l2) de los momentos;")
    reportar("  min n = mínimo en toda la simulación; t cuasi-est.: primer t con |dN|/|N| < 1e-4 por paso;")
    reportar("  cambio rel. = |dN|/|N| en el último paso.")
    reportar()

    # --- Tabla 2: isolínea frente a la teoría (casos 1-3) ---
    reportar(f"RAZÓN DE ASPECTO DE LA ISOLÍNEA n = {NIVEL_ISOLINEA:g} max n (t = T)")
    reportar(f"  {'caso':<8s} {'D_par/D_perp':>12s} {'radio a lo largo':>16s} {'radio a través':>14s}"
             f" {'razón':>7s} {'sqrt(D_par/D_perp)':>18s} {'error rel.':>10s}")
    for nombre in CASOS_ISOLINEA:
        r = resultados[nombre]
        c = casos.config(nombre)
        N = r["N_guardados"][-1]
        razon, r_par, r_perp = postproc.razon_aspecto_isolinea(
            r["nodes"], r["elements"], N, NIVEL_ISOLINEA * N.max(), c["fuente"]["xs"],
            c["fuente"]["ys"], np.radians(c["campo"]["alpha_grados"]), r_max=10.0)
        teo = np.sqrt(c["D_par"] / c["D_perp"])
        fila = next(f for f in filas if f["caso"] == nombre)
        fila.update({"isolinea_razon": razon, "isolinea_teorica": teo})
        reportar(f"  {nombre:<8s} {c['D_par'] / c['D_perp']:12g} {r_par:16.3f} {r_perp:14.3f}"
                 f" {razon:7.3f} {teo:18.3f} {abs(razon - teo) / teo:10.1%}")
    reportar()

    # --- Tabla 3: balance de masa del caso 5 ---
    reportar("BALANCE DE MASA, CASO 5 (fuente pulsada)")
    c5 = casos.config("caso5_neumann")
    f = c5["fuente"]
    entrada_exacta = f["Q0"] * 2.0 * np.pi * f["sigma"] ** 2 * f["t_apagado"]
    ok_balance = True
    for nombre in ("caso5_neumann", "caso5_dirichlet"):
        r = resultados[nombre]
        S = r["entrada_acum"]
        reportar(f"  {nombre}:")
        reportar(f"    int Q dt (discreta) = {S[-1]:.8f};  analítica Q0 2 pi sigma^2 t_off = "
                 f"{entrada_exacta:.8f}  (dif. rel. {abs(S[-1] / entrada_exacta - 1):.1e})")
        if nombre == "caso5_neumann":
            err = np.max(np.abs(r["masa"] - r["masa"][0] - S)) / S[-1]
            ok = err < TOL_NEUMANN
            ok_balance &= ok
            reportar(f"    max_t |masa - int Q dt| / int Q dt = {err:.2e}"
                     f"  (< {TOL_NEUMANN:.0e}: {'OK' if ok else 'FALLA'})")
            reportar(f"    flujo directo por la frontera al final: {r['flujo_directo'][-1]:.2e}"
                     f" (debe ser ~ 0)")
        else:
            sal_d, sal_b = r["salida_directa_acum"], r["salida_balance_acum"]
            err_d = np.max(np.abs(r["masa"] + sal_d - S)) / S[-1]
            err_b = np.max(np.abs(r["masa"] + sal_b - S)) / S[-1]
            reportar(f"    masa final = {r['masa'][-1]:.5f}, salida acumulada (directa) = {sal_d[-1]:.5f}")
            reportar(f"    max_t |masa + salida directa - int Q dt| / int Q dt = {err_d:.2e}"
                     f"  (salida calculada con -int D grad n . n_ext)")
            reportar(f"    max_t |masa + salida por balance - int Q dt| / int Q dt = {err_b:.2e}"
                     f"  (se cumple por definición)")
    reportar()

    with open(casos.DIR_CASOS / "resumen.csv", "w", newline="", encoding="utf-8") as fh:
        claves = list(filas[0].keys()) + ["isolinea_razon", "isolinea_teorica"]
        w = csv.DictWriter(fh, fieldnames=claves)
        w.writeheader()
        for fila in filas:
            w.writerow(fila)

    (casos.DIR_CASOS / "resumen.txt").write_text("\n".join(lineas) + "\n", encoding="utf-8")
    print("Guardado en results/casos/: <caso>.npz, resumen.csv, resumen.txt")
    return 0 if ok_balance else 1


if __name__ == "__main__":
    sys.exit(main())
