"""Escenario físico: difusión anisotrópica de rayos cósmicos que escapan de un remanente
de supernova, basado en Nava y Gabici (2013, MNRAS 429, 1643).

Dominio [-350, 350] x [-150, 150] pc (x paralelo a B), B uniforme horizontal,
D_par = 1e29 cm^2/s (en pc^2/kyr), D_perp = eps D_par, n0 gaussiana de 5 pc, Q = 0,
Dirichlet, theta = 1, dt = 0.02 kyr, T = 10 kyr. Se trabaja en pc y kyr.

Verificación contra la gaussiana anisotrópica exacta y análisis de dos nubes de prueba,
A en (50, 0) pc (a lo largo de B) y B en (0, 50) pc (perpendicular): densidad en el tiempo
y coeficiente isotrópico D_iso que reproduciría la densidad observada.
Guarda CSV, figuras PNG/PDF y results/14_snr.txt.
"""

import csv
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from scipy.optimize import brentq

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from src import fields, mesh, postproc, rutas, timestep, viz  # noqa: E402

# --- Unidades ---
PC_CM = 3.0857e18
KYR_S = 3.156e10
D_PAR_CGS = 1e29                                   # cm^2/s
D_par = D_PAR_CGS * KYR_S / PC_CM ** 2             # pc^2/kyr (~331)

# --- Problema ---
LX, LY = 350.0, 150.0                              # dominio [-LX, LX] x [-LY, LY] pc
sigma0, n_max = 5.0, 1.0
eps_base = 0.01
dt, T = 0.02, 10.0                                 # kyr
t_guardar = [0.0, 1.0, 2.0, 5.0, 10.0]
t_error = [1.0, 5.0, 10.0]
eps_barrido = [1.0, 0.1, 0.01, 0.001]
nube_A, nube_B = (50.0, 0.0), (0.0, 50.0)

MALLAS = [("estructurada 281x121", "estructurada", (281, 121)),
          ("estructurada 561x241", "estructurada", (561, 241)),
          ("gmsh 5/1 (r = 30)", "gmsh", (5.0, 1.0, 30.0))]
if rutas.RAPIDO:
    MALLAS = [m for m in MALLAS if m[0] != "estructurada 561x241"]

RESULTADOS = rutas.RESULTADOS
COLOR_EPS = {1.0: "#D55E00", 0.1: "#E69F00", 0.01: "#0072B2", 0.001: "#009E73"}
COLOR_MALLA = ["#0072B2", "#009E73", "#D55E00"]


def construir_malla(tipo, par):
    if tipo == "estructurada":
        nodes, elements, bnd, _ = mesh.malla_estructurada(2 * LX, 2 * LY, *par)
    else:
        h_global, h_fuente, r_ref = par
        nodes, elements, bnd, _ = mesh.make_mesh(2 * LX, 2 * LY, h_global, h_fuente, LX, LY, r_ref)
    return nodes - np.array([LX, LY]), elements, bnd


def tensor(eps):
    return np.diag([D_par, eps * D_par])


def n_isotropa(D, r, t):
    """Gaussiana isotrópica en 2D: n_max sigma0^2 / S exp(-r^2 / (2 S)), S = sigma0^2 + 2 D t."""
    S = sigma0 ** 2 + 2.0 * D * t
    return n_max * sigma0 ** 2 / S * np.exp(-r ** 2 / (2.0 * S))


def raices_D_iso(n_obs, r, t):
    """Coeficientes isotrópicos que dan n_obs en (r, t). Retorna (raíz pequeña, raíz grande, D*, n*)."""
    S_star = r ** 2 / 2.0                         # máximo de n_isotropa como función de D
    D_star = (S_star - sigma0 ** 2) / (2.0 * t)
    n_star = n_isotropa(D_star, r, t)
    if n_obs > n_star:
        return np.nan, np.nan, D_star, n_star
    f = lambda logD: n_isotropa(10.0 ** logD, r, t) - n_obs
    lo, hi = np.log10(D_star) - 12, np.log10(D_star) + 12
    pequena = 10.0 ** brentq(f, lo, np.log10(D_star)) if f(lo) < 0 else np.nan
    grande = 10.0 ** brentq(f, np.log10(D_star), hi) if f(hi) < 0 else np.nan
    return pequena, grande, D_star, n_star


def correr(nodes, elements, bnd, eps):
    D_func = fields.funcion_tensor(fields.campo_uniforme(0.0), D_par, eps * D_par)
    n0 = fields.condicion_inicial_gaussiana(n_max, 0.0, 0.0, sigma0)
    P_A = viz.matriz_rejilla(nodes, elements, np.array([nube_A[0]]), np.array([nube_A[1]]))
    P_B = viz.matriz_rejilla(nodes, elements, np.array([nube_B[0]]), np.array([nube_B[1]]))
    reg = {"t": [], "nA": [], "nB": [], "pico": []}

    def al_paso(k, t, N):
        reg["t"].append(t)
        reg["nA"].append((P_A @ N)[0])
        reg["nB"].append((P_B @ N)[0])
        reg["pico"].append(N.max())

    res = timestep.simular(nodes, elements, bnd, D_func, n0, T, dt, theta=1.0, Q=None,
                           bc="dirichlet", t_guardar=t_guardar, al_paso=al_paso)
    r = {k: np.array(v) for k, v in reg.items()}
    r["estados"] = dict(zip(np.round(res["t"], 6), res["N"]))
    r["M"] = res["M"]
    return r


def guardar_csv(nombre, filas):
    with open(RESULTADOS / nombre, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(filas[0].keys()))
        w.writeheader()
        w.writerows(filas)


def analisis_nube(n_obs, dndt, r, t):
    """D_iso para la densidad observada; la raíz física se elige por el signo de dn/dt."""
    pequena, grande, D_star, n_star = raices_D_iso(n_obs, r, t)
    fisica = np.nan
    if not np.isnan(pequena):
        fisica = grande if dndt < 0 else pequena    # si ya baja, el frente ya pasó: raíz grande
    return {"pequena": pequena, "grande": grande, "fisica": fisica, "D_star": D_star, "n_star": n_star}


def main():
    rutas.avisar()
    RESULTADOS.mkdir(exist_ok=True)
    viz.estilo()
    lineas = []

    def reportar(texto=""):
        print(texto, flush=True)
        lineas.append(texto)

    reportar("ESCENARIO FÍSICO: RAYOS CÓSMICOS QUE ESCAPAN DE UN REMANENTE DE SUPERNOVA")
    reportar("(basado en Nava y Gabici, 2013)")
    reportar("UNIDADES (se trabaja en pc y kyr)")
    reportar(f"  1 pc = {PC_CM:.4e} cm;  1 kyr = {KYR_S:.4e} s")
    reportar(f"  D_par = {D_PAR_CGS:.0e} cm^2/s = {D_par:.2f} pc^2/kyr")
    reportar(f"  sigma0 = {sigma0:g} pc;  dt = {dt:g} kyr;  T = {T:g} kyr;  tiempo de difusión de la fuente"
             f" sigma0^2 / D_par = {sigma0 ** 2 / D_par:.3f} kyr")
    reportar("  B0 = 3 microG no entra en el modelo: solo importa la dirección de B a través de D.")
    reportar()

    # --- Mallas y verificación del caso base ---
    D0 = tensor(eps_base)
    n_ex = fields.solucion_gaussiana(n_max, 0.0, 0.0, sigma0, D0)
    corridas_malla = {}
    mallas = {}
    reportar(f"VERIFICACIÓN (caso base eps = {eps_base:g}, D_perp = {eps_base * D_par:.3f} pc^2/kyr)")
    for nombre, tipo, par in MALLAS:
        t1 = time.time()
        nodes, elements, bnd = construir_malla(tipo, par)
        mallas[nombre] = (nodes, elements, bnd)
        corridas_malla[nombre] = correr(nodes, elements, bnd, eps_base)
        print(f"  {nombre}: {len(nodes)} nodos ({time.time() - t1:.0f} s)", flush=True)
    nb, _, bb = mallas[MALLAS[0][0]]
    borde = {eps: max(fields.solucion_gaussiana(n_max, 0, 0, sigma0, tensor(eps))(nb[bb, 0], nb[bb, 1], t).max()
                      for t in t_guardar[1:]) for eps in eps_barrido}
    reportar("  Valor exacto máximo en la frontera (t <= 10 kyr), relativo a n_max = 1:")
    reportar("    " + ";  ".join(f"eps = {e:g}: {v:.1e}" for e, v in borde.items()))
    reportar("    Con eps = 1 la frontera y = +-150 pc no es despreciable (sigma_y ~ 82 pc en t = 10 kyr):")
    reportar("    esa corrida del barrido pierde algo de masa por la frontera.")
    reportar()

    filas_ver = []
    reportar(f"  {'malla':<22s} {'nodos':>7s} {'t (kyr)':>7s} {'L1 rel.':>9s} {'L2 rel.':>9s} {'<x^2>':>9s} {'teo':>9s}"
             f" {'<y^2>':>8s} {'teo':>8s} {'aspecto':>8s} {'teo':>7s}")
    for nombre, _, _ in MALLAS:
        nodes, elements, _ = mallas[nombre]
        r = corridas_malla[nombre]
        for t in t_error:
            N = r["estados"][round(t, 6)]
            exacta = lambda x, y, t=t: n_ex(x, y, t)
            mom = postproc.momentos_cuadratura(nodes, elements, N)
            Sx, Sy = sigma0 ** 2 + 2 * D_par * t, sigma0 ** 2 + 2 * eps_base * D_par * t
            fila = {"malla": nombre, "nodos": len(nodes), "t_kyr": t,
                    "error_L1_rel": postproc.error_L1_relativo(nodes, elements, N, exacta),
                    "error_L2_rel": postproc.error_L2_relativo(nodes, elements, N, exacta),
                    "x2": mom["xx"][0], "x2_teo": Sx, "y2": mom["yy"][0], "y2_teo": Sy,
                    "aspecto": float(postproc.razon_aspecto(mom)[0]), "aspecto_teo": np.sqrt(Sx / Sy)}
            filas_ver.append(fila)
            reportar(f"  {nombre:<22s} {len(nodes):7d} {t:7g} {fila['error_L1_rel']:9.2e} {fila['error_L2_rel']:9.2e}"
                     f" {fila['x2']:9.1f} {Sx:9.1f} {fila['y2']:8.2f} {Sy:8.2f} {fila['aspecto']:8.3f}"
                     f" {fila['aspecto_teo']:7.3f}")
    reportar("  <x^2>, <y^2> en pc^2; teórico: sigma0^2 + 2 D t; aspecto = sqrt(<x^2>/<y^2>).")
    reportar()
    reportar(f"  Densidad en las nubes en t = {T:g} kyr según la malla (exacta: n_A = {float(n_ex(*nube_A, T)):.4e},"
             f" n_B = {float(n_ex(*nube_B, T)):.4e})")
    for nombre, _, _ in MALLAS:
        r = corridas_malla[nombre]
        reportar(f"    {nombre:<22s} n_A = {r['nA'][-1]:.4e} ({r['nA'][-1] / float(n_ex(*nube_A, T)) - 1:+.1%})"
                 f"   n_B = {r['nB'][-1]:.4e} ({r['nB'][-1] / float(n_ex(*nube_B, T)) - 1:+.0%})")
    reportar("    La nube B está en la cola profunda de la gaussiana perpendicular (n_B / pico ~ 1e-6):")
    reportar("    un error absoluto pequeño es un error relativo grande allí.")
    reportar()
    guardar_csv("14_snr_verificacion.csv", filas_ver)

    # --- Barrido de eps (malla base) ---
    malla_base = MALLAS[0][0]
    nodes, elements, bnd = mallas[malla_base]
    corridas = {eps_base: corridas_malla[malla_base]}
    for eps in eps_barrido:
        if eps != eps_base:
            t1 = time.time()
            corridas[eps] = correr(nodes, elements, bnd, eps)
            print(f"  eps = {eps:g}: {time.time() - t1:.0f} s", flush=True)

    r_nube = np.hypot(*nube_A)
    filas = []
    for eps in eps_barrido:
        r = corridas[eps]
        N_T = r["estados"][round(T, 6)]
        mom = postproc.momentos_cuadratura(nodes, elements, N_T)
        ex = fields.solucion_gaussiana(n_max, 0, 0, sigma0, tensor(eps))
        dA, dB = (r["nA"][-1] - r["nA"][-2]) / dt, (r["nB"][-1] - r["nB"][-2]) / dt
        aA = analisis_nube(r["nA"][-1], dA, r_nube, T)
        aB = analisis_nube(r["nB"][-1], dB, r_nube, T)
        dB_ex = (float(ex(*nube_B, T)) - float(ex(*nube_B, T - dt))) / dt
        aB_ex = analisis_nube(float(ex(*nube_B, T)), dB_ex, r_nube, T)
        filas.append({
            "eps": eps, "aspecto": float(postproc.razon_aspecto(mom)[0]),
            "aspecto_teo": np.sqrt((sigma0 ** 2 + 2 * D_par * T) / (sigma0 ** 2 + 2 * eps * D_par * T)),
            "pico": r["pico"][-1], "pico_exacto": float(ex(0.0, 0.0, T)),
            "nA": r["nA"][-1], "nA_exacta": float(ex(*nube_A, T)), "nB": r["nB"][-1],
            "nB_exacta": float(ex(*nube_B, T)), "nA_sobre_nB": r["nA"][-1] / r["nB"][-1],
            "dnA_dt": dA, "dnB_dt": dB,
            "Diso_A_pequena": aA["pequena"] / D_par, "Diso_A_grande": aA["grande"] / D_par,
            "Diso_A_fisica": aA["fisica"] / D_par,
            "Diso_B_pequena": aB["pequena"] / D_par, "Diso_B_grande": aB["grande"] / D_par,
            "Diso_B_fisica": aB["fisica"] / D_par, "Diso_B_fisica_exacta": aB_ex["fisica"] / D_par,
            "D_estrella_sobre_Dpar": aA["D_star"] / D_par, "n_estrella": aA["n_star"],
        })
    guardar_csv("14_snr_barrido.csv", filas)

    reportar(f"BARRIDO DE eps = D_perp / D_par (malla {malla_base}, t = {T:g} kyr)")
    reportar(f"  {'eps':>6s} {'aspecto':>8s} {'teo':>8s} {'pico':>9s} {'exacto':>9s} {'n_A':>10s} {'exacta':>10s}"
             f" {'n_B':>10s} {'exacta':>10s} {'n_A/n_B':>9s}")
    for f in filas:
        reportar(f"  {f['eps']:6g} {f['aspecto']:8.3f} {f['aspecto_teo']:8.3f} {f['pico']:9.3e} {f['pico_exacto']:9.3e}"
                 f" {f['nA']:10.3e} {f['nA_exacta']:10.3e} {f['nB']:10.3e} {f['nB_exacta']:10.3e}"
                 f" {f['nA_sobre_nB']:9.2e}")
    reportar(f"  Nube A en ({nube_A[0]:g}, {nube_A[1]:g}) pc (a lo largo de B); nube B en ({nube_B[0]:g}, {nube_B[1]:g}) pc.")
    reportar()

    f_star = filas[0]
    reportar("COEFICIENTE ISOTRÓPICO INFERIDO (D_iso / D_par) A PARTIR DE LA DENSIDAD EN t = 10 kyr")
    reportar(f"  Con difusión isotrópica, n(r = {r_nube:g} pc, t = {T:g} kyr) tiene un máximo como función de D:")
    reportar(f"  n* = {f_star['n_estrella']:.3e} en D* = {f_star['D_estrella_sobre_Dpar']:.4f} D_par"
             f" = {f_star['D_estrella_sobre_Dpar'] * D_PAR_CGS:.2e} cm^2/s.")
    reportar("  Si la densidad observada supera n*, ningún D isotrópico la reproduce; si es menor, hay dos")
    reportar("  raíces: la pequeña (el frente todavía no llega a la nube, n sube) y la grande (el frente ya")
    reportar("  pasó, n baja). La física es la que tiene el mismo signo de dn/dt que la simulación.")
    reportar(f"  {'eps':>6s} {'nube':>4s} {'n':>10s} {'dn/dt':>10s} {'raíz pequeña':>13s} {'raíz grande':>12s}"
             f" {'física':>9s} {'D_iso (cm^2/s)':>15s}")
    fmt = lambda v: "     -" if np.isnan(v) else f"{v:.3e}"
    for f in filas:
        for nube in ("A", "B"):
            fisica = f[f"Diso_{nube}_fisica"]
            reportar(f"  {f['eps']:6g} {nube:>4s} {f['n' + nube]:10.3e} {f['dn' + nube + '_dt']:10.2e}"
                     f" {fmt(f[f'Diso_{nube}_pequena']):>13s} {fmt(f[f'Diso_{nube}_grande']):>12s}"
                     f" {fmt(fisica):>9s} {('sin solución' if np.isnan(fisica) else f'{fisica * D_PAR_CGS:.2e}'):>15s}")
    reportar("  Para la nube B, D_iso físico usando la densidad exacta en vez de la numérica:")
    reportar("    " + ";  ".join(f"eps = {f['eps']:g}: {fmt(f['Diso_B_fisica_exacta'])}" for f in filas))
    reportar("    Con eps pequeño la densidad numérica en B tiene un error relativo grande; D_iso depende de")
    reportar("    ella de forma logarítmica, así que el valor inferido cambia menos (hasta ~45% con")
    reportar("    eps = 0.001), sin alterar la conclusión: desde B se infiere un D_iso del orden de D_perp.")
    reportar()

    # --- Discusión ---
    base = next(f for f in filas if f["eps"] == eps_base)
    reportar("DISCUSIÓN")
    reportar(f"  Con eps = {eps_base:g}, en t = {T:g} kyr la nube A (a lo largo de B) recibe n_A / n_B ="
             f" {base['nA_sobre_nB']:.1e} veces más rayos")
    reportar("  cósmicos que la B (perpendicular), aunque están a la misma distancia de la fuente.")
    if np.isnan(base["Diso_A_fisica"]):
        reportar(f"  La densidad en A ({base['nA']:.2e}) supera el máximo que permite cualquier difusión isotrópica")
        reportar(f"  ({base['n_estrella']:.2e}): un observador que suponga isotropía no puede explicarla con ningún D;")
        reportar(f"  lo más cercano es D* = {base['D_estrella_sobre_Dpar']:.3f} D_par, y aun así se queda"
                 f" {base['nA'] / base['n_estrella']:.1f} veces corto.")
    reportar(f"  Para la nube B el D isotrópico físico es {fmt(base['Diso_B_fisica'])} D_par: interpretar la")
    reportar("  densidad perpendicular con difusión isotrópica da un coeficiente muy distinto del real.")
    reportar("  Esto concuerda con la conclusión del artículo: con más anisotropía, la interpretación")
    reportar("  isotrópica da coeficientes muy distintos del real, porque los rayos cósmicos se concentran")
    reportar("  en el tubo de flujo que pasa por la fuente.")
    reportar("  Diferencias de modelo con el artículo (la comparación es cualitativa):")
    reportar("  - Ellos usan D_perp = 0 con deambulación de las líneas de campo (difusión compuesta,")
    reportar("    <dx_perp^2> ~ sqrt(t), subdifusiva); aquí D_perp es efectiva y <dy^2> ~ t.")
    reportar("  - Ellos trabajan en 3D, con D(E) e inyección que depende del tiempo; aquí, 2D, una sola")
    reportar("    energía y un pulso instantáneo. En 2D la densidad decae como 1/t y no como t^(-3/2),")
    reportar("    así que los valores numéricos de D_iso no son trasladables al caso 3D.")
    reportar(f"  Resolución: con eps = 0.001, sigma_y = {np.sqrt(sigma0 ** 2 + 2e-3 * D_par * T):.1f} pc en t = 10 kyr,"
             " solo unos 2 nodos por sigma en la malla base (h = 2.5 pc).")
    reportar("  Con eps = 1 la frontera absorbente en y = +-150 pc recorta la nube (sigma ~ 82 pc), y por eso")
    reportar("  la razón de aspecto numérica es mayor que 1.")
    reportar("  La malla de gmsh está refinada solo alrededor de la fuente (r = 30 pc); la nube sale de esa")
    reportar("  zona a lo largo de B y en los elementos de h = 5 pc su ancho perpendicular (~9.5 pc) queda")
    reportar("  mal resuelto, por eso su error crece con el tiempo. Como en el halo (13_halo.py), lo que")
    reportar("  importa es resolver la dirección perpendicular en todo el tubo de flujo.")
    reportar()

    # --- Figuras ---
    nodes, elements, _ = mallas[malla_base]
    x, y = nodes[:, 0], nodes[:, 1]
    norma = LogNorm(vmin=1e-5, vmax=1.0)
    niveles = np.logspace(-5, 0, 31)

    def mapa(ax, N, titulo):
        cf = ax.tricontourf(x, y, elements, np.clip(N, 1e-5, None), levels=niveles, norm=norma,
                            cmap="viridis", extend="min")
        for (px, py), nombre in ((nube_A, "A"), (nube_B, "B")):
            ax.plot(px, py, "o", markersize=7, markerfacecolor="none", markeredgecolor="#f97316", mew=2)
            ax.annotate(nombre, (px, py), textcoords="offset points", xytext=(6, 6), color="#f97316",
                        fontsize=11, fontweight="bold")
        ax.set_aspect("equal")
        ax.set_xlabel("x (pc), paralelo a B")
        ax.set_ylabel("y (pc)")
        ax.set_title(titulo, fontsize=11)
        ax.grid(False)
        return cf

    r0 = corridas[eps_base]
    fig, axes = plt.subplots(2, 2, figsize=(14, 7.4), constrained_layout=True)
    for ax, t in zip(axes.ravel(), [0.0, 2.0, 5.0, 10.0]):
        cf = mapa(ax, r0["estados"][round(t, 6)], f"t = {t:g} kyr")
    cb = fig.colorbar(cf, ax=axes.ravel().tolist(), shrink=0.85, label="n (escala logarítmica)")
    cb.set_ticks([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1])
    fig.suptitle(rf"Rayos cósmicos de un remanente: $\varepsilon$ = {eps_base:g}, malla {malla_base}")
    viz.guardar(fig, RESULTADOS / "14_snr_mapas")

    fig, axes = plt.subplots(2, 2, figsize=(14, 7.4), constrained_layout=True)
    for ax, eps in zip(axes.ravel(), eps_barrido):
        cf = mapa(ax, corridas[eps]["estados"][round(T, 6)], rf"$\varepsilon$ = {eps:g}, t = {T:g} kyr")
    cb = fig.colorbar(cf, ax=axes.ravel().tolist(), shrink=0.85, label="n (escala logarítmica)")
    cb.set_ticks([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1])
    fig.suptitle("Efecto de la anisotropía en t = 10 kyr (misma escala de colores)")
    viz.guardar(fig, RESULTADOS / "14_snr_barrido")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), constrained_layout=True)
    s_x, s_y = np.linspace(-LX, LX, 1401), np.linspace(-LY, LY, 1201)
    for i, (nombre, _, _) in enumerate(MALLAS):
        nd, el, _ = mallas[nombre]
        N = corridas_malla[nombre]["estados"][round(T, 6)]
        axes[0].plot(s_x, postproc.evaluar_en_puntos(nd, el, N, s_x, 0 * s_x), color=COLOR_MALLA[i], label=nombre)
        axes[1].plot(s_y, postproc.evaluar_en_puntos(nd, el, N, 0 * s_y, s_y), color=COLOR_MALLA[i], label=nombre)
    axes[0].plot(s_x, n_ex(s_x, 0 * s_x, T), color="#111827", linestyle="--", linewidth=1.2, label="exacta")
    axes[1].plot(s_y, n_ex(0 * s_y, s_y, T), color="#111827", linestyle="--", linewidth=1.2, label="exacta")
    axes[0].set_xlabel("x (pc), a lo largo de B (y = 0)")
    axes[1].set_xlabel("y (pc), a través de B (x = 0)")
    axes[1].set_xlim(-60, 60)
    for ax in axes:
        ax.set_ylabel("n")
    axes[0].legend(fontsize=9)
    fig.suptitle(rf"Cortes en t = {T:g} kyr, $\varepsilon$ = {eps_base:g}: numérica frente a exacta")
    viz.guardar(fig, RESULTADOS / "14_snr_cortes")

    fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
    tt = np.linspace(dt, T, 500)
    for eps in (1.0, 0.01):
        r = corridas[eps]
        ex = fields.solucion_gaussiana(n_max, 0, 0, sigma0, tensor(eps))
        ax.semilogy(r["t"], np.clip(r["nA"], 1e-12, None), color=COLOR_EPS[eps], label=rf"nube A, $\varepsilon$ = {eps:g}")
        ax.semilogy(r["t"], np.clip(r["nB"], 1e-12, None), color=COLOR_EPS[eps], linestyle="--",
                    label=rf"nube B, $\varepsilon$ = {eps:g}")
        ax.semilogy(tt, [ex(*nube_A, t) for t in tt], color="#111827", linewidth=0.8, linestyle=":")
        ax.semilogy(tt, [max(ex(*nube_B, t), 1e-12) for t in tt], color="#111827", linewidth=0.8, linestyle=":")
    ax.plot([], [], color="#111827", linestyle=":", linewidth=0.8, label="exacta")
    ax.set_ylim(1e-10, 0.1)
    ax.set_xlabel("t (kyr)")
    ax.set_ylabel("n en la nube")
    ax.set_title("Densidad en las nubes A (a lo largo de B) y B (perpendicular), a 50 pc")
    ax.legend(fontsize=9)
    viz.guardar(fig, RESULTADOS / "14_snr_nubes")

    fig, ax = plt.subplots(figsize=(7.4, 4.8), constrained_layout=True)
    e = np.array([f["eps"] for f in filas])
    for nube, marca, color, z in (("B", "s", "#D55E00", 2), ("A", "o", "#0072B2", 3)):
        ax.loglog(e, [f[f"Diso_{nube}_fisica"] for f in filas], marca + "-", color=color, markersize=8,
                  zorder=z, label=f"nube {nube}, raíz física")
        otra = [f[f"Diso_{nube}_grande"] if f[f"Diso_{nube}_fisica"] == f[f"Diso_{nube}_pequena"]
                else f[f"Diso_{nube}_pequena"] for f in filas]
        ax.loglog(e, otra, marca, color=color, markersize=7, markerfacecolor="white", zorder=z,
                  label=f"nube {nube}, otra raíz")
    sin_sol = [f["eps"] for f in filas if np.isnan(f["Diso_A_fisica"])]
    if sin_sol:
        ax.loglog(sin_sol, [filas[0]["D_estrella_sobre_Dpar"]] * len(sin_sol), "x", color="#0072B2",
                  markersize=10, mew=2, zorder=4, label=r"nube A: sin solución (n_A > n$^*$)")
    ax.axhline(1.0, color="#6b7280", linestyle=":", label=r"$D_{iso} = D_\parallel$")
    ax.axhline(filas[0]["D_estrella_sobre_Dpar"], color="#9ca3af", linestyle="--",
               label=r"$D^*$ (máximo de n isotrópica)")
    ax.set_xlabel(r"$\varepsilon = D_\perp / D_\parallel$")
    ax.set_ylabel(r"$D_{iso} / D_\parallel$")
    ax.set_title("Coeficiente isotrópico inferido de la densidad en t = 10 kyr\n"
                 "(nube A sin solución cuando su densidad supera el máximo isotrópico)", fontsize=10)
    ax.legend(fontsize=8)
    viz.guardar(fig, RESULTADOS / "14_snr_D_iso")

    (RESULTADOS / "14_snr.txt").write_text("\n".join(lineas) + "\n", encoding="utf-8")
    print("Guardado en " + RESULTADOS.name + "/: 14_snr.txt, 14_snr_verificacion.csv, 14_snr_barrido.csv"
          " y figuras 14_snr_mapas/barrido/cortes/nubes/D_iso")
    return 0


if __name__ == "__main__":
    sys.exit(main())
