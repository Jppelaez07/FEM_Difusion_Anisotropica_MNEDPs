"""Prueba 3.1 de Pakmor et al. (2016): difusión isotrópica de un perfil gaussiano.

u(x, t) = 1 + 1e-2 / (2 pi Dc) exp(-r^2 / (2 Dc)),  Dc = 2 kappa t,  kappa = 0.01,
en [-0.5, 0.5]^2, desde t0 = 0.1 hasta t = 0.2, con u = 1 en la frontera.

Como la EDP es lineal y K 1 = 0, se escribe u = 1 + v y se resuelve v con Dirichlet
homogénea, con tau = t - t0 en [0, 0.1]. La solución exacta de v es
fields.solucion_gaussiana con sigma0^2 = 2 kappa t0 y D = kappa I.
Las mallas se generan en [0, 1]^2 y se desplazan a [-0.5, 0.5]^2.

1. Caso base: malla estructurada de 51 x 51 nodos, theta = 1 y theta = 0.5.
2. Convergencia: mallas estructuradas y no estructuradas (gmsh) con N nodos por lado.
Guarda CSV, figuras PNG/PDF y results/06_pakmor.txt.
"""

import csv
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
from matplotlib.colors import TwoSlopeNorm

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from src import fields, mesh, postproc, rutas, timestep, viz  # noqa: E402

# Parámetros del artículo
kappa = 0.01
t0, tf = 0.1, 0.2
amplitud = 1e-2                          # u = 1 + amplitud / (2 pi Dc) exp(...)
dt = 1e-4                                # 1000 pasos
tau_f = tf - t0
sigma0 = np.sqrt(2.0 * kappa * t0)
n_max = amplitud / (2.0 * np.pi * sigma0 ** 2)
D0 = kappa * np.eye(2)

N_base = 51
N_conv = [26, 51, 101, 201, 401]
if rutas.RAPIDO:
    N_conv = [N for N in N_conv if N < 401]
N_cortes = [26, 51, 101]
thetas = [0.5, 1.0]
tipos = ["estructurada", "no estructurada"]

RESULTADOS = rutas.RESULTADOS
COLOR = {("estructurada", 0.5): "#0072B2", ("estructurada", 1.0): "#56B4E9",
         ("no estructurada", 0.5): "#D55E00", ("no estructurada", 1.0): "#E69F00"}
MARCA = {0.5: "o", 1.0: "s"}

D_func = fields.funcion_tensor(fields.campo_uniforme(0.0), kappa, kappa)
v_exacta = fields.solucion_gaussiana(n_max, 0.0, 0.0, sigma0, D0)     # v(x, y, tau)


def u_articulo(x, y, t):
    """Fórmula del artículo."""
    Dc = 2.0 * kappa * t
    return 1.0 + amplitud / (2.0 * np.pi * Dc) * np.exp(-(x ** 2 + y ** 2) / (2.0 * Dc))


def malla(tipo, N):
    """Malla de [-0.5, 0.5]^2 con N nodos por lado (o h = 1/(N-1) si es no estructurada)."""
    if tipo == "estructurada":
        nodes, elements, bnd, _ = mesh.malla_estructurada(1.0, 1.0, N, N)
    else:
        nodes, elements, bnd, _ = mesh.malla_uniforme(1.0, 1.0, 1.0 / (N - 1))
    return nodes - 0.5, elements, bnd


def correr(nodes, elements, bnd, theta):
    """Retorna v en tau = 0 y tau = tau_f y la matriz de masa global."""
    res = timestep.simular(nodes, elements, bnd, D_func, lambda x, y: v_exacta(x, y, 0.0),
                           tau_f, dt, theta=theta, Q=None, bc="dirichlet", t_guardar=[0.0, tau_f])
    return res["N"][0], res["N"][-1], res["M"]


def metricas(nodes, elements, N0, NT, M):
    vT = lambda x, y: v_exacta(x, y, tau_f)
    sin_grad = lambda x, y: (0 * x, 0 * y)
    return {
        "L1": postproc.error_L1(nodes, elements, NT, vT),
        "L1_rel": postproc.error_L1_relativo(nodes, elements, NT, vT),
        "L2": postproc.errores_L2_H1(nodes, elements, NT, vT, sin_grad)[0],
        "L2_rel": postproc.error_L2_relativo(nodes, elements, NT, vT),
        "min_u_t0": 1.0 + N0.min(), "min_u_tf": 1.0 + NT.min(),
        "masa_v_t0": postproc.masa(M, N0), "masa_v_tf": postproc.masa(M, NT),
    }


def tasas(h, e):
    h, e = np.asarray(h), np.asarray(e)
    return np.concatenate(([np.nan], np.log(e[:-1] / e[1:]) / np.log(h[:-1] / h[1:])))


def guardar_csv(nombre, filas):
    with open(RESULTADOS / nombre, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(filas[0].keys()))
        w.writeheader()
        w.writerows(filas)


def main():
    rutas.avisar()
    RESULTADOS.mkdir(exist_ok=True)
    viz.estilo()
    lineas = []

    def reportar(texto=""):
        print(texto, flush=True)
        lineas.append(texto)

    # --- Comprobaciones previas ---
    g = np.linspace(-0.5, 0.5, 401)
    X, Y = np.meshgrid(g, g)
    dif = max(np.max(np.abs(1.0 + v_exacta(X, Y, t - t0) - u_articulo(X, Y, t))) for t in (t0, tf))
    nb, eb, bb = malla("estructurada", 401)
    v_borde = max(v_exacta(nb[bb, 0], nb[bb, 1], tau).max() for tau in (0.0, tau_f))

    reportar("PRUEBA 3.1 DE PAKMOR ET AL. (2016): DIFUSIÓN ISOTRÓPICA DE UN PERFIL GAUSSIANO")
    reportar(f"  kappa = {kappa:g}, t0 = {t0:g}, t = {tf:g}, dominio [-0.5, 0.5]^2, u = 1 en la frontera")
    reportar(f"  u = 1 + v; v con Dirichlet homogénea; sigma0^2 = 2 kappa t0 = {sigma0 ** 2:g},"
             f" n_max = {n_max:.6f}, D = kappa I")
    reportar(f"  dt = {dt:g} ({int(round(tau_f / dt))} pasos), Q = 0")
    reportar(f"  max |1 + v_exacta - u_artículo| en t0 y en t = {tf:g}: {dif:.1e}")
    reportar(f"  max de v exacta en la frontera (t0 a t = {tf:g}): {v_borde:.1e}  (despreciable)")
    reportar(f"  masa exacta de v: {amplitud:g} (constante en el plano infinito)")
    reportar()

    # --- Corridas ---
    resultados = {}
    datos_malla = {}
    for tipo in tipos:
        for N in N_conv:
            t1 = time.time()
            nodes, elements, bnd = malla(tipo, N)
            q = mesh.calidad_malla(nodes, elements)
            datos_malla[(tipo, N)] = (nodes, elements, q)
            for th in thetas:
                N0, NT, M = correr(nodes, elements, bnd, th)
                resultados[(tipo, N, th)] = {"N0": N0, "NT": NT,
                                             **metricas(nodes, elements, N0, NT, M)}
            print(f"  {tipo}, N = {N}: {len(nodes)} nodos ({time.time() - t1:.1f} s)", flush=True)

    # --- Caso base ---
    reportar(f"1. CASO BASE: malla estructurada de {N_base} x {N_base} nodos (h = {1 / (N_base - 1):g})")
    reportar(f"  {'theta':>5s} {'error L1':>11s} {'L1 rel.':>9s} {'error L2':>11s} {'L2 rel.':>9s}"
             f" {'min u t0':>9s} {'min u tf':>9s} {'masa v t0':>11s} {'masa v tf':>11s}")
    filas_base = []
    for th in thetas:
        r = resultados[("estructurada", N_base, th)]
        filas_base.append({"theta": th, **{k: v for k, v in r.items() if k not in ("N0", "NT")}})
        reportar(f"  {th:5g} {r['L1']:11.4e} {r['L1_rel']:9.2e} {r['L2']:11.4e} {r['L2_rel']:9.2e}"
                 f" {r['min_u_t0']:9.6f} {r['min_u_tf']:9.6f} {r['masa_v_t0']:11.6e} {r['masa_v_tf']:11.6e}")
    reportar("  Errores relativos respecto a v (la perturbación); como int |u| ~ 1, el error")
    reportar("  relativo respecto a u es prácticamente igual al absoluto.")
    reportar()
    guardar_csv("06_pakmor_caso_base.csv", filas_base)

    # --- Convergencia ---
    reportar("2. CONVERGENCIA EN t = 0.2 (tasas entre mallas consecutivas, con h = 1/(N-1))")
    reportar(f"  {'malla':<16s} {'theta':>5s} {'N':>4s} {'h medio':>8s} {'nodos':>7s} {'error L1':>11s}"
             f" {'L1 rel.':>9s} {'tasa':>5s} {'error L2':>11s} {'L2 rel.':>9s} {'tasa':>5s} {'min u':>9s}")
    filas = []
    for tipo in tipos:
        for th in thetas:
            sel = [(N, resultados[(tipo, N, th)]) for N in N_conv]
            h = [1.0 / (N - 1) for N, _ in sel]
            t1 = tasas(h, [r["L1"] for _, r in sel])
            t2 = tasas(h, [r["L2"] for _, r in sel])
            for (N, r), a, b in zip(sel, t1, t2):
                _, _, q = datos_malla[(tipo, N)]
                filas.append({"malla": tipo, "theta": th, "N": N, "h": 1.0 / (N - 1),
                              "h_medio": q["h_medio"], "nodos": q["n_nodos"],
                              "error_L1": r["L1"], "error_L1_rel": r["L1_rel"], "tasa_L1": a,
                              "error_L2": r["L2"], "error_L2_rel": r["L2_rel"], "tasa_L2": b,
                              "min_u_t0": r["min_u_t0"], "min_u_tf": r["min_u_tf"],
                              "masa_v_t0": r["masa_v_t0"], "masa_v_tf": r["masa_v_tf"]})
                fa = "    -" if np.isnan(a) else f"{a:5.2f}"
                fb = "    -" if np.isnan(b) else f"{b:5.2f}"
                reportar(f"  {tipo:<16s} {th:5g} {N:4d} {q['h_medio']:8.5f} {q['n_nodos']:7d}"
                         f" {r['L1']:11.4e} {r['L1_rel']:9.2e} {fa} {r['L2']:11.4e} {r['L2_rel']:9.2e}"
                         f" {fb} {r['min_u_tf']:9.6f}")
            reportar()
    guardar_csv("06_pakmor_convergencia.csv", filas)

    # --- Resumen y comparación con el artículo ---
    N_fin = N_conv[-1]
    reportar("3. RESUMEN")
    for tipo in tipos:
        for th in thetas:
            sel = [f for f in filas if f["malla"] == tipo and f["theta"] == th]
            pend = np.polyfit(np.log([f["N"] for f in sel]), np.log([f["error_L1"] for f in sel]), 1)[0]
            reportar(f"  {tipo:<16s} theta = {th:g}: pendiente de log(L1) vs log(N) = {pend:5.2f};"
                     f" última tasa L1 = {sel[-1]['tasa_L1']:.2f}")
    masas = [f["masa_v_tf"] / f["masa_v_t0"] - 1 for f in filas]
    minimos = [f["min_u_tf"] for f in filas]
    reportar(f"  Masa de v: variación relativa entre t0 y t = {tf:g} entre {min(masas):.1e} y {max(masas):.1e}.")
    reportar(f"  Mínimo de u en t = {tf:g}: entre {min(minimos):.8f} y {max(minimos):.8f}"
             f" ({'u >= 1 en todas las corridas' if min(minimos) >= 1 - 1e-12 else 'hay valores por debajo de 1'}).")
    reportar("  Comparación con el artículo: Pakmor et al. obtienen un error L1 que decrece como N^-2")
    reportar("  (segundo orden). Aquí la comparación es cualitativa: se compara la pendiente, no los")
    reportar("  valores, porque el método numérico es distinto.")
    reportar("  Error temporal de theta = 1: como el de theta = 0.5 es despreciable con este dt,")
    reportar("  se estima con ||u(theta = 1) - u(theta = 0.5)||_L1 en la misma malla, y se compara con")
    reportar("  el error espacial (error L1 con theta = 0.5).")
    reportar(f"  {'malla':<16s} {'N':>4s} {'error espacial':>15s} {'error temporal':>15s} {'temporal/espacial':>18s}")
    cero = lambda x, y: 0 * x
    for tipo in tipos:
        for N in N_conv:
            nodes, elements, _ = datos_malla[(tipo, N)]
            dif = resultados[(tipo, N, 1.0)]["NT"] - resultados[(tipo, N, 0.5)]["NT"]
            e_t = postproc.error_L1(nodes, elements, dif, cero)
            e_s = resultados[(tipo, N, 0.5)]["L1"]
            reportar(f"  {tipo:<16s} {N:4d} {e_s:15.4e} {e_t:15.4e} {e_t / e_s:18.2f}")
    reportar("  Con theta = 1, en las mallas finas el error temporal O(dt) es del mismo orden que el")
    reportar("  espacial y lo cancela en parte (el error total queda por debajo del de theta = 0.5).")
    reportar("  Por eso las tasas mayores que 2 con theta = 1 no son un orden mayor: son el efecto de")
    reportar("  esa cancelación. El orden espacial limpio (2.00) se ve con theta = 0.5.")
    reportar()

    # --- Figura 1: mapas del caso base (theta = 1) ---
    nodes, elements, _ = datos_malla[("estructurada", N_base)]
    r = resultados[("estructurada", N_base, 1.0)]
    x, y = nodes[:, 0], nodes[:, 1]
    u_ex = 1.0 + v_exacta(x, y, tau_f)
    vmax_u = 1.0 + max(r["N0"].max(), r["NT"].max())
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 9.2), constrained_layout=True)
    paneles = [(axes[0, 0], 1.0 + r["N0"], f"u numérica, t = {t0:g}"),
               (axes[0, 1], 1.0 + r["NT"], f"u numérica, t = {tf:g}"),
               (axes[1, 0], u_ex, f"u exacta, t = {tf:g}")]
    for ax, valores, titulo in paneles:
        # extend="min": los valores apenas por debajo de 1 (v_h ~ -1e-6) toman el color mínimo
        cf = ax.tricontourf(x, y, elements, valores, levels=np.linspace(1.0, vmax_u, 31), cmap="viridis",
                            extend="min")
        ax.set_title(titulo)
    fig.colorbar(cf, ax=[axes[0, 0], axes[0, 1]], shrink=0.85, label="u (adimensional)", format="%.2f")
    fig.colorbar(cf, ax=axes[1, 0], shrink=0.85, label="u (adimensional)", format="%.2f")
    err = 1.0 + r["NT"] - u_ex
    lim = np.abs(err).max()
    ce = axes[1, 1].tricontourf(x, y, elements, err, levels=np.linspace(-lim, lim, 31), cmap="RdBu_r",
                                norm=TwoSlopeNorm(0.0, -lim, lim))
    fig.colorbar(ce, ax=axes[1, 1], shrink=0.85, label=r"$u_h - u$ (adimensional)", format="%.1e")
    axes[1, 1].set_title(f"Error $u_h - u$, t = {tf:g}")
    for ax in axes.ravel():
        ax.set_xlim(-0.25, 0.25)
        ax.set_ylim(-0.25, 0.25)
        ax.set_aspect("equal")
        ax.set_xlabel("x (adimensional)")
        ax.set_ylabel("y (adimensional)")
        ax.grid(False)
    fig.suptitle(f"Pakmor 3.1, malla estructurada {N_base} x {N_base}, theta = 1"
                 " (ventana [-0.25, 0.25]^2 del dominio [-0.5, 0.5]^2)")
    viz.guardar(fig, RESULTADOS / "06_pakmor_mapas")

    # --- Figura 2: cortes por y = 0 (theta = 1, malla estructurada) ---
    xs = np.linspace(-0.25, 0.25, 801)
    colores_N = {26: "#E69F00", 51: "#0072B2", 101: "#009E73"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True, sharey=True)
    for ax, (clave, tau, t) in zip(axes, (("N0", 0.0, t0), ("NT", tau_f, tf))):
        for N in N_cortes:
            nd, el, _ = datos_malla[("estructurada", N)]
            ax.plot(xs, 1.0 + postproc.evaluar_en_puntos(nd, el, resultados[("estructurada", N, 1.0)][clave],
                                                         xs, 0 * xs),
                    color=colores_N[N], label=f"N = {N}")
        ax.plot(xs, u_articulo(xs, 0 * xs, t), color="#111827", linestyle="--", linewidth=1.3,
                label="Exacta")
        ax.set_title(f"Corte por y = 0, t = {t:g}")
        ax.set_xlabel("x (adimensional)")
    axes[0].set_ylabel("u (adimensional)")
    axes[0].legend()
    fig.suptitle("Pakmor 3.1: numérica (malla estructurada, theta = 1) frente a exacta\n"
                 "con N par, y = 0 cae entre dos filas de nodos y el corte interpola entre ellas",
                 fontsize=11)
    viz.guardar(fig, RESULTADOS / "06_pakmor_cortes")

    # --- Figura 3: convergencia (formato de la Fig. 3 del artículo) ---
    fig, ax = plt.subplots(figsize=(6.6, 5.0), constrained_layout=True)
    for tipo in tipos:
        for th in thetas:
            sel = [f for f in filas if f["malla"] == tipo and f["theta"] == th]
            ax.loglog([f["N"] for f in sel], [f["error_L1"] for f in sel], marker=MARCA[th],
                      linestyle="-" if tipo == "estructurada" else "--", color=COLOR[(tipo, th)],
                      markersize=7, label=rf"{tipo}, $\theta$ = {th:g}")
    Nr = np.array([10.0, 1000.0])
    ref = [f["error_L1"] for f in filas if f["malla"] == "estructurada" and f["theta"] == 0.5][0]
    ax.loglog(Nr, ref * (Nr / N_conv[0]) ** -2 * 0.5, color="#6b7280", linestyle=":", linewidth=1.3,
              label=r"$\propto N^{-2}$")
    ax.set_xlim(10, 1000)
    ax.set_xlabel("N (nodos por lado)")
    ax.set_ylabel(r"Error $L^1$ en t = 0.2")
    ax.set_title("Pakmor 3.1: convergencia del error L1")
    ax.legend(fontsize=9)
    viz.guardar(fig, RESULTADOS / "06_pakmor_convergencia")

    (RESULTADOS / "06_pakmor.txt").write_text("\n".join(lineas) + "\n", encoding="utf-8")
    print("Guardado en " + RESULTADOS.name + "/: 06_pakmor.txt, 06_pakmor_caso_base.csv,"
          " 06_pakmor_convergencia.csv y figuras 06_pakmor_mapas/cortes/convergencia (png y pdf)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
