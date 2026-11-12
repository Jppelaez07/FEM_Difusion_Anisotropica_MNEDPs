"""Prueba 3.2 de Pakmor et al. (2016): difusión anisotrópica de una cuña en un campo circular
(problema de Parrish y Stone, 2005).

Dominio [-1, 1]^2, B circular centrado en el origen, D_par = 0.01, D_perp = 0, Q = 0,
Neumann, theta = 1. u0 = 12 en la cuña 0.5 < r < 0.7, |phi| < pi/12, y 10 en el resto.
Solución exacta en t = 10 (difusión a lo largo de cada círculo) y en t = 200 (10 + 1/6
dentro del anillo). Las mallas se generan en [0, 2]^2 y se desplazan a [-1, 1]^2.

Métricas: errores L1 y L2 con sus tasas, conservación de la energía 1^T M N y principio
del máximo. Guarda CSV, figuras PNG/PDF y results/07_anillo.txt.
"""

import csv
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import erfc

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from src import assembly, fields, mesh, postproc, rutas, timestep, viz  # noqa: E402

# Parámetros del artículo
D_par, D_perp = 0.01, 0.0
r_in, r_out = 0.5, 0.7
alfa = np.pi / 12                         # semiancho angular de la cuña
u_fondo, u_cuna = 10.0, 12.0
corridas = {10.0: 0.01, 200.0: 0.1}       # T: dt
N_lista = [51, 101, 201, 401]
if rutas.RAPIDO:
    N_lista = [N for N in N_lista if N < 401]
N_base = 101
tipos = ["estructurada", "no estructurada"]
ESCALA = (9.98, 10.64)                    # escala de colores de la Fig. 5 del artículo
UMBRAL_AREA = 9.98                        # área con u < 9.98: descenso mayor que el 1% de la amplitud

RESULTADOS = rutas.RESULTADOS
COLOR_N = {51: "#E69F00", 101: "#0072B2", 201: "#009E73", 401: "#D55E00"}
COLOR_TIPO = {"estructurada": "#0072B2", "no estructurada": "#D55E00"}

D_func = fields.funcion_tensor(fields.campo_circular(0.0, 0.0), D_par, D_perp)


def polares(x, y):
    return np.hypot(x, y), np.arctan2(y, x)


def u0(x, y):
    r, phi = polares(x, y)
    return np.where((r > r_in) & (r < r_out) & (np.abs(phi) < alfa), u_cuna, u_fondo)


def u_anillo(phi, r, t, imagenes=0):
    """Solución a lo largo del círculo de radio r; imagenes = k suma las copias en phi + 2 pi j, |j| <= k."""
    c = r / np.sqrt(4.0 * D_par * t)
    total = 0.0
    for j in range(-imagenes, imagenes + 1):
        p = phi + 2.0 * np.pi * j
        total = total + erfc((p - alfa) * c) - erfc((p + alfa) * c)
    return u_fondo + 0.5 * (u_cuna - u_fondo) * total


def u_exacta(t):
    """Solución exacta del artículo en t = 10 (erfc) o t = 200 (10 + 1/6 en el anillo)."""
    def u(x, y):
        r, phi = polares(x, y)
        dentro = (r > r_in) & (r < r_out)
        if t >= 200.0:
            valor = np.full_like(r, u_fondo + (u_cuna - u_fondo) * (2 * alfa) / (2 * np.pi))
        else:
            valor = u_anillo(phi, r, t)
        return np.where(dentro, valor, u_fondo)
    return u


def malla(tipo, N):
    if tipo == "estructurada":
        nodes, elements, bnd, _ = mesh.malla_estructurada(2.0, 2.0, N, N)
    else:
        nodes, elements, bnd, _ = mesh.malla_uniforme(2.0, 2.0, 2.0 / (N - 1))
    return nodes - 1.0, elements, bnd


def correr(nodes, elements, bnd, T, dt):
    """Corre hasta T registrando energía, mínimo y máximo en cada paso."""
    M, _ = assembly.ensamblar_MK(nodes, elements, D_func)
    fila_masa = np.asarray(M.sum(axis=0)).ravel()
    reg = {"t": [], "E": [], "min": [], "max": []}

    def al_paso(k, t, N):
        reg["t"].append(t)
        reg["E"].append(fila_masa @ N)
        reg["min"].append(N.min())
        reg["max"].append(N.max())

    res = timestep.simular(nodes, elements, bnd, D_func, u0, T, dt, theta=1.0, Q=None,
                           bc="neumann", t_guardar=[T], al_paso=al_paso)
    r = {k: np.array(v) for k, v in reg.items()}
    r["NT"] = res["N"][-1]
    return r


def tasas(h, e):
    h, e = np.asarray(h), np.asarray(e)
    return np.concatenate(([np.nan], np.log(e[:-1] / e[1:]) / np.log(h[:-1] / h[1:])))


def guardar_csv(nombre, filas):
    with open(RESULTADOS / nombre, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(filas[0].keys()))
        w.writeheader()
        w.writerows(filas)


def fracion_area_bajo(nodes, elements, N, umbral):
    """Fracción del área donde u_h < umbral (cuadratura de 6 puntos)."""
    detJ, _, _ = assembly.geometria(nodes, elements)
    uq = N[elements] @ postproc.CUAD_BARI.T
    pesos = (detJ / 2.0)[:, None] * postproc.CUAD_PESOS
    return np.sum(pesos * (uq < umbral)) / np.sum(pesos)


def main():
    rutas.avisar()
    RESULTADOS.mkdir(exist_ok=True)
    viz.estilo()
    lineas = []

    def reportar(texto=""):
        print(texto, flush=True)
        lineas.append(texto)

    # --- Comprobaciones de la solución exacta ---
    phi = np.linspace(-np.pi, np.pi, 20001)
    lejos = np.abs(np.abs(phi) - alfa) > 1e-3
    dif_t0 = max(np.max(np.abs(u_anillo(phi[lejos], r, 1e-8)
                               - np.where(np.abs(phi[lejos]) < alfa, u_cuna, u_fondo)))
                 for r in (0.51, 0.6, 0.69))
    u_centro = u_anillo(0.0, 0.6, 10.0)
    cola = max(np.max(np.abs(u_anillo(phi, r, 10.0, imagenes=3) - u_anillo(phi, r, 10.0)))
               for r in (0.5, 0.6, 0.7))
    desvio_200 = max(np.max(np.abs(u_anillo(phi, r, 200.0, imagenes=10) - (u_fondo + 1.0 / 6.0)))
                     for r in (0.5, 0.6, 0.7))

    reportar("PRUEBA 3.2 DE PAKMOR ET AL. (2016): CUÑA EN UN CAMPO MAGNÉTICO CIRCULAR")
    reportar(f"  Dominio [-1, 1]^2, B circular en el origen, D_par = {D_par:g}, D_perp = {D_perp:g},"
             " Q = 0, Neumann, theta = 1")
    reportar(f"  u0 = {u_cuna:g} en {r_in:g} < r < {r_out:g}, |phi| < pi/12; {u_fondo:g} en el resto"
             " (interpolada en los nodos)")
    reportar("  Corridas: " + ", ".join(f"T = {T:g} con dt = {dt:g}" for T, dt in corridas.items()))
    reportar("  Solución exacta en t = 10: u = 10 + erfc((phi - pi/12) r / sqrt(4 D_par t))"
             " - erfc((phi + pi/12) r / sqrt(4 D_par t)) en el anillo.")
    reportar(f"    En t -> 0 reproduce u0: max |u - u0| lejos de los bordes de la cuña = {dif_t0:.1e}")
    reportar(f"    En phi = 0, r = 0.6, t = 10: u = {u_centro:.6f} (> 10)")
    reportar(f"    Aporte de la cola que da la vuelta al anillo (imágenes periódicas) en t = 10: {cola:.1e}")
    reportar("  Solución exacta en t = 200: u = 10 + 1/6 en el anillo (mezcla angular completa).")
    reportar(f"    La solución periódica exacta en t = 200 se aparta de 10 + 1/6 como máximo en {desvio_200:.1e}")
    reportar()

    # --- Corridas ---
    res = {}
    mallas = {}
    for tipo in tipos:
        for N in N_lista:
            t1 = time.time()
            nodes, elements, bnd = malla(tipo, N)
            mallas[(tipo, N)] = (nodes, elements, mesh.calidad_malla(nodes, elements))
            for T, dt in corridas.items():
                res[(tipo, N, T)] = correr(nodes, elements, bnd, T, dt)
            print(f"  {tipo}, N = {N}: {len(nodes)} nodos ({time.time() - t1:.1f} s)", flush=True)

    # --- Tablas de convergencia ---
    filas = []
    for T in corridas:
        ue = u_exacta(T)
        reportar(f"CONVERGENCIA EN t = {T:g} (tasas con h = 2/(N-1))")
        reportar(f"  {'malla':<16s} {'N':>4s} {'nodos':>7s} {'error L1':>11s} {'tasa':>5s}"
                 f" {'error L2':>11s} {'tasa':>5s} {'min u':>9s} {'max u':>9s} {'var. E max':>11s}")
        for tipo in tipos:
            sel = []
            for N in N_lista:
                nodes, elements, q = mallas[(tipo, N)]
                r = res[(tipo, N, T)]
                sel.append({
                    "t": T, "malla": tipo, "N": N, "h": 2.0 / (N - 1), "h_medio": q["h_medio"],
                    "nodos": q["n_nodos"],
                    "error_L1": postproc.error_L1(nodes, elements, r["NT"], ue),
                    "error_L2": postproc.errores_L2_H1(nodes, elements, r["NT"], ue,
                                                       lambda x, y: (0 * x, 0 * y))[0],
                    "min_u": r["min"].min(), "max_u": r["max"].max(),
                    "variacion_E_max": np.max(np.abs(r["E"] / r["E"][0] - 1.0)),
                    "descenso_bajo_10_pct": 100.0 * max(u_fondo - r["min"].min(), 0.0) / (u_cuna - u_fondo),
                    "area_bajo_998_pct": 100.0 * fracion_area_bajo(nodes, elements, r["NT"], UMBRAL_AREA),
                })
            t1 = tasas([f["h"] for f in sel], [f["error_L1"] for f in sel])
            t2 = tasas([f["h"] for f in sel], [f["error_L2"] for f in sel])
            for f, a, b in zip(sel, t1, t2):
                f["tasa_L1"], f["tasa_L2"] = a, b
                fa = "    -" if np.isnan(a) else f"{a:5.2f}"
                fb = "    -" if np.isnan(b) else f"{b:5.2f}"
                reportar(f"  {tipo:<16s} {f['N']:4d} {f['nodos']:7d} {f['error_L1']:11.4e} {fa}"
                         f" {f['error_L2']:11.4e} {fb} {f['min_u']:9.5f} {f['max_u']:9.5f}"
                         f" {f['variacion_E_max']:11.2e}")
            filas.extend(sel)
        reportar()
    guardar_csv("07_anillo_convergencia.csv", filas)

    # --- Principio del máximo y energía ---
    reportar("PRINCIPIO DEL MÁXIMO (en toda la simulación) Y CONSERVACIÓN DE LA ENERGÍA")
    reportar("  descenso = (10 - min u) / (12 - 10) en %; área = fracción del dominio con u < 9.98 en t = T")
    reportar("  (9.98 = un descenso de más del 1% de la amplitud; contar u < 10 mediría oscilaciones de redondeo)")
    reportar(f"  {'malla':<16s} {'N':>4s} {'T':>4s} {'min u':>9s} {'max u':>9s} {'descenso %':>10s}"
             f" {'área %':>8s} {'var. E max':>11s}")
    for f in filas:
        reportar(f"  {f['malla']:<16s} {f['N']:4d} {f['t']:4.0f} {f['min_u']:9.5f} {f['max_u']:9.5f}"
                 f" {f['descenso_bajo_10_pct']:10.3f} {f['area_bajo_998_pct']:8.3f} {f['variacion_E_max']:11.2e}")
    reportar()

    # --- Comparación con el artículo ---
    reportar("COMPARACIÓN CON EL ARTÍCULO")
    for T in corridas:
        for tipo in tipos:
            sel = [f for f in filas if f["t"] == T and f["malla"] == tipo]
            pend = -np.polyfit(np.log([f["N"] for f in sel]), np.log([f["error_L1"] for f in sel]), 1)[0]
            reportar(f"  t = {T:g}, {tipo}: pendiente del error L1 contra N = {pend:.2f}")
    reportar("  Pakmor et al. reportan tasas de convergencia del error L1 de 0.55 y 0.7 para este")
    reportar("  problema: la solución exacta es discontinua en r = 0.5 y r = 0.7 (D_perp = 0), así que")
    reportar("  ningún método de bajo orden converge con orden 1 o 2.")
    peor = min(f["min_u"] for f in filas)
    reportar(f"  Mínimo de u en todas las corridas: {peor:.5f}. Con limitador, el artículo mantiene u >= 10;")
    reportar("  sin limitador, reporta valores por debajo de 10 de unos pocos por ciento. Este código no")
    reportar("  tiene limitador (Galerkin P1 estándar), así que se espera lo segundo.")
    reportar("  La energía 1^T M N se conserva hasta el redondeo con Neumann (1^T K = 0).")
    reportar()

    # --- Figuras: mapas en t = 10 y t = 200 ---
    niveles = np.linspace(*ESCALA, 34)
    for T in corridas:
        paneles = [(f"N = {N}", ("estructurada", N)) for N in N_lista] + [("Exacta", None)]
        n_col = 3
        n_fil = int(np.ceil(len(paneles) / n_col))
        fig, axes = plt.subplots(n_fil, n_col, figsize=(13.5, 4.5 * n_fil), constrained_layout=True)
        axes = np.atleast_2d(axes).ravel()
        xg = np.linspace(-0.8, 0.8, 801)
        Xg, Yg = np.meshgrid(xg, xg)
        for ax, (titulo, clave) in zip(axes, paneles):
            if clave is None:
                # La exacta es discontinua: se dibuja en una rejilla fina, no en una malla
                cf = ax.contourf(Xg, Yg, u_exacta(T)(Xg, Yg), levels=niveles, cmap="viridis",
                                 extend="both")
            else:
                nodes, elements, _ = mallas[clave]
                cf = ax.tricontourf(nodes[:, 0], nodes[:, 1], elements, res[(*clave, T)]["NT"],
                                    levels=niveles, cmap="viridis", extend="both")
            ax.set_title(titulo)
            ax.set_aspect("equal")
            ax.set_xlim(-0.8, 0.8)
            ax.set_ylim(-0.8, 0.8)
            ax.set_xlabel("x (adimensional)")
            ax.set_ylabel("y (adimensional)")
            ax.grid(False)
        for ax in axes[len(paneles):]:
            ax.set_visible(False)
        fig.colorbar(cf, ax=axes[:len(paneles)].tolist(), shrink=0.85,
                     label=f"u (adimensional; escala fija {ESCALA[0]:g} a {ESCALA[1]:g})")
        fig.suptitle(f"Pakmor 3.2: cuña en campo circular, t = {T:g} (mallas estructuradas)")
        viz.guardar(fig, RESULTADOS / f"07_anillo_mapas_t{T:g}")

    # --- Perfil angular en r = 0.6, t = 10 ---
    phi = np.linspace(-np.pi, np.pi, 1441)
    fig, ax = plt.subplots(figsize=(8, 4.6), constrained_layout=True)
    for N in N_lista:
        nodes, elements, _ = mallas[("estructurada", N)]
        ax.plot(phi, postproc.evaluar_en_puntos(nodes, elements, res[("estructurada", N, 10.0)]["NT"],
                                                0.6 * np.cos(phi), 0.6 * np.sin(phi)),
                color=COLOR_N[N], label=f"N = {N}")
    ax.plot(phi, u_anillo(phi, 0.6, 10.0), color="#111827", linestyle="--", linewidth=1.3, label="Exacta")
    ax.set_xlabel(r"$\varphi$ (rad)")
    ax.set_ylabel("u (adimensional)")
    ax.set_title("Perfil angular en r = 0.6, t = 10 (mallas estructuradas)")
    ax.legend()
    viz.guardar(fig, RESULTADOS / "07_anillo_perfil_angular")

    # --- Perfil radial en phi = 0, t = 200 ---
    rr = np.linspace(0.3, 0.9, 1201)
    fig, ax = plt.subplots(figsize=(8, 4.6), constrained_layout=True)
    for N in N_lista:
        nodes, elements, _ = mallas[("estructurada", N)]
        ax.plot(rr, postproc.evaluar_en_puntos(nodes, elements, res[("estructurada", N, 200.0)]["NT"],
                                               rr, 0 * rr), color=COLOR_N[N], label=f"N = {N}")
    ax.plot(rr, u_exacta(200.0)(rr, 0 * rr), color="#111827", linestyle="--", linewidth=1.3,
            label="Exacta")
    ax.axhline(u_fondo, color="#9ca3af", linewidth=0.8)
    ax.set_xlabel("r (adimensional)")
    ax.set_ylabel("u (adimensional)")
    ax.set_title(r"Perfil radial en $\varphi$ = 0, t = 200: ensanchamiento por difusión numérica")
    ax.legend()
    viz.guardar(fig, RESULTADOS / "07_anillo_perfil_radial")

    # --- Convergencia ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    for ax, T in zip(axes, corridas):
        for tipo in tipos:
            sel = [f for f in filas if f["t"] == T and f["malla"] == tipo]
            ax.loglog([f["N"] for f in sel], [f["error_L1"] for f in sel], "o-",
                      color=COLOR_TIPO[tipo], markersize=7, label=tipo)
        e0 = [f["error_L1"] for f in filas if f["t"] == T and f["malla"] == "estructurada"][0]
        Nr = np.array([N_lista[0], N_lista[-1]], dtype=float)
        for p, estilo in ((0.55, ":"), (0.7, "--"), (1.0, "-.")):
            ax.loglog(Nr, 1.15 * e0 * (Nr / Nr[0]) ** (-p), color="#6b7280", linestyle=estilo,
                      linewidth=1.1, label=rf"$\propto N^{{-{p:g}}}$")
        ax.set_xlabel("N (nodos por lado)")
        ax.set_ylabel(r"Error $L^1$")
        ax.set_title(f"t = {T:g}")
        ax.legend(fontsize=9)
    fig.suptitle("Pakmor 3.2: convergencia del error L1")
    viz.guardar(fig, RESULTADOS / "07_anillo_convergencia")

    # --- Energía ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), constrained_layout=True)
    for ax, T in zip(axes, corridas):
        for N in N_lista:
            r = res[("estructurada", N, T)]
            ax.semilogy(r["t"][1:], np.abs(r["E"][1:] / r["E"][0] - 1.0) + 1e-17, color=COLOR_N[N],
                        label=f"N = {N}")
        ax.set_xlabel("t (adimensional)")
        ax.set_ylabel(r"$|E(t)/E(0) - 1|$")
        ax.set_title(f"Conservación de la energía, t hasta {T:g}")
        ax.legend(fontsize=9)
    fig.suptitle(r"Energía $E = \mathbf{1}^T M N$ (mallas estructuradas, Neumann)")
    viz.guardar(fig, RESULTADOS / "07_anillo_energia")

    (RESULTADOS / "07_anillo.txt").write_text("\n".join(lineas) + "\n", encoding="utf-8")
    print("Guardado en " + RESULTADOS.name + "/: 07_anillo.txt, 07_anillo_convergencia.csv y figuras"
          " 07_anillo_mapas_t10, _mapas_t200, _perfil_angular, _perfil_radial, _convergencia, _energia")
    return 0


if __name__ == "__main__":
    sys.exit(main())
