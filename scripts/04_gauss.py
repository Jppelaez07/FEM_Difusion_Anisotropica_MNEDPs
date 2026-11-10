"""Comparación con la solución gaussiana anisotrópica analítica.

Dominio [0,20]^2, Dirichlet, Q = 0, B uniforme a 30 grados, theta = 1, T = 2.
La nube no llega a la frontera, así que la solución en el plano infinito sirve
de referencia. Calcula el error L2 relativo en el tiempo para 3 mallas y
compara cortes a lo largo y a través de B.
Guarda figuras PNG/PDF, CSV y un resumen en results/.
"""

import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from src import fields, mesh, postproc, timestep  # noqa: E402

# Parámetros (unidades adimensionales)
Lx, Ly = 20.0, 20.0
xs, ys = Lx / 2.0, Ly / 2.0
D_par, D_perp = 1.0, 0.1
alpha = np.radians(30.0)
n_max, sigma0 = 1.0, 1.0
T, dt, theta = 2.0, 0.002, 1.0
t_guardar = np.round(np.arange(0.0, T + 1e-9, 0.05), 10)

# Mallas: refinadas alrededor de la fuente
h_global, r_ref = 1.0, 7.0
h_fuente = [0.4, 0.2, 0.1]

t_cortes = [0.5, 2.0]
RESULTADOS = RAIZ / "results"
COLORES = ["#93c5fd", "#3b82f6", "#1e3a8a"]          # de malla gruesa a fina
COLOR_T = {0.5: "#c2410c", 2.0: "#1d4ed8"}


def main():
    RESULTADOS.mkdir(exist_ok=True)
    lineas = []

    def reportar(texto=""):
        print(texto, flush=True)
        lineas.append(texto)

    B = fields.campo_uniforme(alpha)
    D_func = fields.funcion_tensor(B, D_par, D_perp)
    D0 = D_func(np.array([xs]), np.array([ys]))[0]
    n_ana = fields.solucion_gaussiana(n_max, xs, ys, sigma0, D0)
    n0 = lambda x, y: n_ana(x, y, 0.0)

    reportar("SOLUCIÓN GAUSSIANA ANISOTRÓPICA")
    reportar(f"  D_par = {D_par:g}, D_perp = {D_perp:g}, B a 30 grados, sigma0 = {sigma0:g}")
    reportar(f"  Dirichlet, theta = {theta:g}, dt = {dt:g}, T = {T:g}")
    reportar(f"  Mallas refinadas: h_global = {h_global:g}, r_ref = {r_ref:g}")
    reportar()

    resultados = []
    for hf in h_fuente:
        nodes, elements, bnd, _ = mesh.make_mesh(Lx, Ly, h_global, hf, xs, ys, r_ref)
        res = timestep.simular(nodes, elements, bnd, D_func, n0, T, dt, theta=theta,
                               Q=None, bc="dirichlet", t_guardar=t_guardar)
        err_rel = []
        for k, tk in enumerate(res["t"]):
            exacta = lambda x, y, tk=tk: n_ana(x, y, tk)
            e, _ = postproc.errores_L2_H1(nodes, elements, res["N"][k], exacta,
                                          lambda x, y: (0 * x, 0 * y))
            norma, _ = postproc.errores_L2_H1(nodes, elements, np.zeros(len(nodes)), exacta,
                                              lambda x, y: (0 * x, 0 * y))
            err_rel.append(e / norma)
        resultados.append({"h": hf, "nodes": nodes, "elements": elements, "res": res,
                           "err": np.array(err_rel)})

    t = resultados[0]["res"]["t"]
    idx = [int(np.argmin(np.abs(t - tc))) for tc in (0.0, 0.5, 1.0, 2.0)]
    reportar("ERROR L2 RELATIVO")
    reportar(f"  {'h fuente':>8s} {'nodos':>6s} {'t=0':>10s} {'t=0.5':>10s} {'t=1':>10s}"
             f" {'t=2':>10s} {'máx':>10s}")
    for r in resultados:
        e = r["err"]
        reportar(f"  {r['h']:8.3f} {len(r['nodes']):6d}"
                 + "".join(f" {e[k]:10.3e}" for k in idx) + f" {e.max():10.3e}")
    reportar()
    tasas = [np.log(resultados[i]["err"][-1] / resultados[i + 1]["err"][-1])
             / np.log(resultados[i]["h"] / resultados[i + 1]["h"]) for i in range(len(resultados) - 1)]
    reportar("  Tasa del error en t = 2 al refinar h fuente: "
             + ", ".join(f"{v:.2f}" for v in tasas) + "  (esperado cerca de 2)")
    fina = resultados[-1]
    n_min = postproc.minimo(fina["res"]["N"])
    masa = postproc.masa(fina["res"]["M"], fina["res"]["N"])
    reportar(f"  Malla fina: mínimo de n = {n_min.min():.3e},"
             f" variación relativa de la masa hasta t = 2: {abs(masa[-1] / masa[0] - 1):.2e}")
    reportar("  (la masa puede bajar algo por la frontera absorbente)")

    # CSV: error relativo en el tiempo
    with open(RESULTADOS / "04_gauss_error.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["t"] + [f"error_rel_h_{r['h']:g}" for r in resultados])
        for k, tk in enumerate(t):
            w.writerow([tk] + [r["err"][k] for r in resultados])

    # Figura 1: error relativo en el tiempo
    fig, ax = plt.subplots(figsize=(6.4, 4.6), constrained_layout=True)
    for r, c in zip(resultados, COLORES):
        ax.semilogy(t[1:], r["err"][1:], color=c, linewidth=2,
                    label=f"h fuente = {r['h']:g} ({len(r['nodes'])} nodos)")
    ax.set_xlabel("t (adimensional)")
    ax.set_ylabel("Error L2 relativo")
    ax.set_title("Error frente a la solución gaussiana analítica")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="center right")
    for ext in ("png", "pdf"):
        fig.savefig(RESULTADOS / f"04_gauss_error.{ext}", dpi=200)
    plt.close(fig)

    # Figura 2: cortes a lo largo y a través de B (malla fina)
    s = np.linspace(-8.0, 8.0, 401)
    direcciones = {"A lo largo de B (30°)": alpha, "A través de B (120°)": alpha + np.pi / 2}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True, sharey=True)
    for ax, (titulo, ang) in zip(axes, direcciones.items()):
        xp, yp = xs + s * np.cos(ang), ys + s * np.sin(ang)
        for tc in t_cortes:
            k = int(np.argmin(np.abs(t - tc)))
            num = postproc.evaluar_en_puntos(fina["nodes"], fina["elements"],
                                             fina["res"]["N"][k], xp, yp)
            ax.plot(s, num, color=COLOR_T[tc], linewidth=2.2, label=f"Numérica, t = {t[k]:g}")
            ax.plot(s, n_ana(xp, yp, t[k]), color="#111827", linestyle="--", linewidth=1.2,
                    label=f"Analítica, t = {t[k]:g}")
        ax.set_title(titulo)
        ax.set_xlabel("Distancia al centro sobre el corte (adimensional)")
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("n (adimensional)")
    axes[0].legend()
    fig.suptitle(f"Cortes de la solución: numérica (h fuente = {fina['h']:g}) vs analítica")
    for ext in ("png", "pdf"):
        fig.savefig(RESULTADOS / f"04_gauss_cortes.{ext}", dpi=200)
    plt.close(fig)

    (RESULTADOS / "04_gauss.txt").write_text("\n".join(lineas) + "\n", encoding="utf-8")
    print("\nGuardado en results/: 04_gauss_error.csv/png/pdf, 04_gauss_cortes.png/pdf, 04_gauss.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
