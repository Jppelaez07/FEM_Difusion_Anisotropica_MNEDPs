"""Corre los escenarios de src/casos.py e imprime las tablas resumen.

1. Métricas al final y t_99 (dM/dt < 1% de la entrada; el flujo directo como referencia).
2. Casos 1-3 frente a la solución exacta de la fuente continua en el plano infinito:
   error L2 relativo y razón de aspecto de la isolínea 0.1 max n en varios instantes.
3. Balance de masa del caso 5.
Guarda results/casos/<caso>.npz, resumen.csv, comparacion_exacta.csv y resumen.txt.
"""

import csv
import sys
import time
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from src import casos, fields, postproc, rutas  # noqa: E402

CASOS_EXACTA = ["caso1", "caso2", "caso3"]
NIVEL_ISOLINEA = 0.1          # fracción de max n
TOL_NEUMANN = 1e-10


def tensor_caso(c):
    """D constante de un caso con campo uniforme."""
    a = np.radians(c["campo"]["alpha_grados"])
    return fields.tensor_difusion(np.cos(a), np.sin(a), c["D_par"], c["D_perp"])[0]


def fmt_t(v):
    return "no alcanzado" if np.isnan(v) else f"{v:.2f}"


def main():
    rutas.avisar()
    casos.DIR_CASOS.mkdir(parents=True, exist_ok=True)
    lineas = []

    def reportar(texto=""):
        print(texto, flush=True)
        lineas.append(texto)

    resultados = {}
    for nombre in casos.CASOS:
        t0 = time.time()
        resultados[nombre] = casos.correr_caso(casos.config(nombre))
        print(f"  {nombre}: {len(resultados[nombre]['nodes'])} nodos, "
              f"{time.time() - t0:.1f} s", flush=True)
    print()

    # --- Tabla 1: métricas al final y t_99 ---
    filas = []
    reportar("1. RESUMEN AL FINAL DE CADA CASO")
    reportar(f"  {'caso':<16s} {'T':>4s} {'masa':>8s} {'salida/entrada':>14s} {'<x^2>':>7s} {'<y^2>':>7s}"
             f" {'<xy>':>6s} {'aspecto':>7s} {'min n':>10s} {'max n':>7s} {'t_99':>12s}"
             f" {'(t_99 directo)':>14s}")
    for nombre, r in resultados.items():
        c = casos.config(nombre)
        f = c["fuente"]
        entrada = f["Q0"] * 2.0 * np.pi * f["sigma"] ** 2
        continua = f["t_apagado"] is None
        fila = {
            "caso": nombre, "nodos": len(r["nodes"]), "T": c["T"], "masa": r["masa"][-1],
            "salida_sobre_entrada": r["flujo_directo"][-1] / entrada if continua else np.nan,
            "xx": r["mom_xx"][-1], "yy": r["mom_yy"][-1], "xy": r["mom_xy"][-1],
            "aspecto": r["razon_aspecto"][-1], "n_min": r["n_min"].min(), "n_max": r["n_max"][-1],
            "t_99_directo": r["t_99_directo"], "t_99_balance": r["t_99_balance"],
        }
        filas.append(fila)
        t99d = fmt_t(fila["t_99_directo"]) if continua else "no aplica"
        t99b = fmt_t(fila["t_99_balance"]) if continua else "no aplica"
        sal = f"{fila['salida_sobre_entrada']:14.4f}" if continua else f"{'-':>14s}"
        reportar(f"  {nombre:<16s} {c['T']:4.0f} {fila['masa']:8.3f} {sal} {fila['xx']:7.2f}"
                 f" {fila['yy']:7.2f} {fila['xy']:6.2f} {fila['aspecto']:7.2f} {fila['n_min']:10.2e}"
                 f" {fila['n_max']:7.4f} {t99b:>12s} {'(' + t99d + ')':>14s}")
    reportar("  salida/entrada = flujo directo -int D grad n . n_ext dividido por Q0 2 pi sigma^2 (en t = T);")
    reportar("  t_99 = primer t con dM/dt < 1% de la entrada (balance); entre paréntesis, como referencia,")
    reportar("  primer t con flujo directo >= 0.99 entrada;")
    reportar("  aspecto = sqrt(l1/l2) de los momentos; min n = mínimo en toda la simulación.")
    r4b = resultados["caso4b"]
    c4b = casos.config("caso4b")
    radio_anillo = np.hypot(c4b["fuente"]["xs"] - c4b["campo"]["xc"],
                            c4b["fuente"]["ys"] - c4b["campo"]["yc"])
    dist_frontera = c4b["dominio"]["Lx"] - c4b["campo"]["xc"] - radio_anillo
    reportar(f"  Caso 4b: las líneas de B son círculos cerrados, así que la masa solo sale por difusión")
    reportar(f"  perpendicular. Distancia del anillo a la frontera ~ {dist_frontera:g}; tiempo de escape"
             f" ~ d^2 / (2 D_perp) = {dist_frontera ** 2 / (2 * c4b['D_perp']):.0f} >> T = {c4b['T']:g}.")
    reportar(f"  En t = T la salida es {r4b['flujo_directo'][-1]:.2e} y la masa {r4b['masa'][-1]:.2f}"
             f" (entrada acumulada {r4b['entrada_acum'][-1]:.2f}).")
    reportar()

    # --- Tabla 2: comparación con la solución exacta en el plano infinito ---
    reportar("2. CASOS 1-3 FRENTE A LA SOLUCIÓN EXACTA (fuente continua, plano infinito)")
    reportar(f"  Isolínea n = {NIVEL_ISOLINEA:g} max n; razón = radio a lo largo de B / radio a través.")
    reportar(f"  {'caso':<6s} {'t':>6s} {'error L2 rel.':>13s} {'max n FEM':>10s} {'max n exacta':>12s}"
             f" {'razón FEM':>9s} {'razón exacta':>12s} {'sqrt(Dp/Dt)':>11s} {'dif. FEM-exacta':>15s}")
    comparacion = []
    for nombre in CASOS_EXACTA:
        r = resultados[nombre]
        c = casos.config(nombre)
        f = c["fuente"]
        n_ex = fields.solucion_fuente_continua(f["Q0"], f["xs"], f["ys"], f["sigma"], tensor_caso(c))
        ang = np.radians(c["campo"]["alpha_grados"])
        teo = np.sqrt(c["D_par"] / c["D_perp"])
        for k, tk in enumerate(r["t_comparacion"]):
            N = r["N_comparacion"][k]
            exacta = lambda x, y, tk=tk: n_ex(x, y, tk)
            err = postproc.error_L2_relativo(r["nodes"], r["elements"], N, exacta)
            max_ex = float(exacta(f["xs"], f["ys"]))
            razon_fem, _, _ = postproc.razon_aspecto_isolinea(
                r["nodes"], r["elements"], N, NIVEL_ISOLINEA * N.max(), f["xs"], f["ys"], ang,
                r_max=10.0)
            razon_ex, _, _ = postproc.razon_aspecto_curva(
                exacta, NIVEL_ISOLINEA * max_ex, f["xs"], f["ys"], ang, r_max=80.0, n_puntos=8001)
            comparacion.append({"caso": nombre, "t": tk, "error_L2_rel": err,
                                "max_n_fem": N.max(), "max_n_exacta": max_ex,
                                "razon_fem": razon_fem, "razon_exacta": razon_ex,
                                "razon_teorica": teo})
            reportar(f"  {nombre:<6s} {tk:6.0f} {err:13.3e} {N.max():10.4f} {max_ex:12.4f}"
                     f" {razon_fem:9.3f} {razon_ex:12.3f} {teo:11.3f}"
                     f" {abs(razon_fem - razon_ex) / razon_ex:15.1%}")
        reportar()

    # --- Tabla 3: balance de masa del caso 5 ---
    reportar("3. BALANCE DE MASA, CASO 5 (fuente pulsada)")
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
        else:
            sal_d = r["salida_directa_acum"]
            err_d = np.max(np.abs(r["masa"] + sal_d - S)) / S[-1]
            reportar(f"    masa final = {r['masa'][-1]:.5f}, salida acumulada (directa) = {sal_d[-1]:.5f}")
            reportar(f"    max_t |masa + salida directa - int Q dt| / int Q dt = {err_d:.2e}")
    reportar()

    with open(casos.DIR_CASOS / "resumen.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(filas[0].keys()))
        w.writeheader()
        w.writerows(filas)
    with open(casos.DIR_CASOS / "comparacion_exacta.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(comparacion[0].keys()))
        w.writeheader()
        w.writerows(comparacion)

    (casos.DIR_CASOS / "resumen.txt").write_text("\n".join(lineas) + "\n", encoding="utf-8")
    print("Guardado en results/casos/: <caso>.npz, resumen.csv, comparacion_exacta.csv, resumen.txt")
    return 0 if ok_balance else 1


if __name__ == "__main__":
    sys.exit(main())
