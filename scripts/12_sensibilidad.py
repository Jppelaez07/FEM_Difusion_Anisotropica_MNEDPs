"""Sensibilidad a la anisotropía y difusión perpendicular numérica.

Gaussiana inicial (sigma0 = 1), Q = 0, Neumann, B uniforme, D_par = 1, theta = 1, T = 4.
Los coeficientes efectivos salen de la pendiente de los momentos de segundo orden
en los ejes paralelo y perpendicular a B: d<sigma^2>/dt = 2 D. Los momentos se
integran con la cuadratura de 6 puntos y las mallas son uniformes, para que toda
la nube (incluidas sus colas) quede en elementos de tamaño h.

A. Barrido de D_par/D_perp en {1, 3, 10, 30, 100, 1000} con B a 30 grados:
   D_eff / D y razón de aspecto final frente a la teórica.
B. Difusión perpendicular numérica: error D_perp_eff / D_perp - 1 en 4 mallas,
   con B a 0 y 30 grados, para D_par/D_perp = 1000 y 100.
Guarda CSV, figuras PNG/PDF y un resumen en results/.
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

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from src import fields, mesh, postproc, timestep, viz  # noqa: E402

# Parámetros comunes (unidades adimensionales)
Lx, Ly = 20.0, 20.0
xs, ys = Lx / 2.0, Ly / 2.0
D_par = 1.0
sigma0, n_max = 1.0, 1.0
T, dt, theta = 4.0, 0.04, 1.0
t_guardar = np.round(np.arange(0.0, T + 1e-9, 0.2), 10)
t_ajuste_min = 0.4                       # el ajuste lineal usa t >= t_ajuste_min

# A. Barrido
razones = [1, 3, 10, 30, 100, 1000]
alpha_barrido = 30.0
h_barrido = 0.1

# B. Difusión numérica
h_numerica = [0.4, 0.2, 0.1, 0.05]
razones_numerica = [1000, 100]
angulos_numerica = [0.0, 30.0]

RESULTADOS = RAIZ / "results"
COLOR_RAZON = {1000: "#D55E00", 100: "#0072B2"}
LINEA_ANGULO = {0.0: "-", 30.0: "--"}


def malla(h):
    return mesh.malla_uniforme(Lx, Ly, h)


def h_medio_malla(nodes, elements):
    """h medio (lado más largo) de los elementos."""
    return mesh.calidad_malla(nodes, elements)["h_medio"]


def coeficientes_efectivos(nodes, elements, bnd, razon, alpha_grados):
    """Corre el caso y retorna D_par_eff, D_perp_eff y la razón de aspecto final."""
    a = np.radians(alpha_grados)
    D_func = fields.funcion_tensor(fields.campo_uniforme(a), D_par, D_par / razon)
    n0 = fields.condicion_inicial_gaussiana(n_max, xs, ys, sigma0)
    res = timestep.simular(nodes, elements, bnd, D_func, n0, T, dt, theta=theta, Q=None,
                           bc="neumann", t_guardar=t_guardar)
    mom = postproc.momentos_cuadratura(nodes, elements, res["N"])
    c, s = np.cos(a), np.sin(a)
    var_par = c * c * mom["xx"] + 2 * s * c * mom["xy"] + s * s * mom["yy"]
    var_perp = s * s * mom["xx"] - 2 * s * c * mom["xy"] + c * c * mom["yy"]
    sel = res["t"] >= t_ajuste_min
    D_par_eff = np.polyfit(res["t"][sel], var_par[sel], 1)[0] / 2.0
    D_perp_eff = np.polyfit(res["t"][sel], var_perp[sel], 1)[0] / 2.0
    aspecto = float(postproc.razon_aspecto(mom)[-1])
    return D_par_eff, D_perp_eff, aspecto


def guardar_csv(nombre, filas):
    with open(RESULTADOS / nombre, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(filas[0].keys()))
        w.writeheader()
        w.writerows(filas)


def parte_A(reportar):
    nodes, elements, bnd, _ = malla(h_barrido)
    hm = h_medio_malla(nodes, elements)
    filas = []
    for razon in razones:
        Dp_eff, Dt_eff, asp = coeficientes_efectivos(nodes, elements, bnd, razon, alpha_barrido)
        D_perp = D_par / razon
        teo = np.sqrt((sigma0 ** 2 + 2 * D_par * T) / (sigma0 ** 2 + 2 * D_perp * T))
        filas.append({"razon": razon, "D_perp": D_perp, "D_par_eff": Dp_eff,
                      "D_par_eff_sobre_D_par": Dp_eff / D_par, "D_perp_eff": Dt_eff,
                      "D_perp_eff_sobre_D_perp": Dt_eff / D_perp,
                      "aspecto_fem": asp, "aspecto_teorico": teo})

    reportar(f"A. BARRIDO DE ANISOTROPÍA (B a {alpha_barrido:g} grados, Neumann, T = {T:g})")
    reportar(f"   Malla uniforme: {len(nodes)} nodos, h medio = {hm:.3f}; ajuste lineal en t >= {t_ajuste_min:g}")
    reportar(f"   {'Dp/Dt':>6s} {'D_perp':>8s} {'D_par_eff':>10s} {'/D_par':>8s} {'D_perp_eff':>11s}"
             f" {'/D_perp':>8s} {'aspecto FEM':>11s} {'teórico':>8s} {'error':>7s}")
    for f in filas:
        reportar(f"   {f['razon']:6g} {f['D_perp']:8.4f} {f['D_par_eff']:10.5f} {f['D_par_eff_sobre_D_par']:8.4f}"
                 f" {f['D_perp_eff']:11.6f} {f['D_perp_eff_sobre_D_perp']:8.4f} {f['aspecto_fem']:11.4f}"
                 f" {f['aspecto_teorico']:8.4f} {abs(f['aspecto_fem'] / f['aspecto_teorico'] - 1):7.2%}")
    reportar("   aspecto teórico = sqrt((sigma0^2 + 2 D_par T) / (sigma0^2 + 2 D_perp T))")
    reportar()
    guardar_csv("12_barrido_anisotropia.csv", filas)

    r = np.array([f["razon"] for f in filas], dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    ax = axes[0]
    ax.semilogx(r, [f["D_par_eff_sobre_D_par"] for f in filas], "o-", color="#0072B2",
                markersize=7, label=r"$D_{\parallel,ef} / D_\parallel$")
    ax.semilogx(r, [f["D_perp_eff_sobre_D_perp"] for f in filas], "s-", color="#D55E00",
                markersize=7, label=r"$D_{\perp,ef} / D_\perp$")
    ax.axhline(1.0, color="#6b7280", linestyle=":", linewidth=1.2)
    ax.set_xlabel(r"$D_\parallel / D_\perp$")
    ax.set_ylabel("Coeficiente efectivo / nominal")
    ax.set_title(f"Coeficientes efectivos (h medio = {hm:.2f})")
    ax.legend()
    ax = axes[1]
    ax.loglog(r, [f["aspecto_teorico"] for f in filas], "-", color="#6b7280", linewidth=1.5,
              label="Teórica")
    ax.loglog(r, [f["aspecto_fem"] for f in filas], "o", color="#0072B2", markersize=8,
              label="FEM (momentos)")
    ax.set_xlabel(r"$D_\parallel / D_\perp$")
    ax.set_ylabel(f"Razón de aspecto en t = {T:g}")
    ax.set_title("Razón de aspecto final de la nube")
    ax.legend()
    viz.guardar(fig, RESULTADOS / "12_barrido_anisotropia")


def parte_B(reportar):
    mallas = []
    for h in h_numerica:
        t0 = time.time()
        nodes, elements, bnd, _ = malla(h)
        mallas.append((h, nodes, elements, bnd, h_medio_malla(nodes, elements)))
        print(f"   malla h = {h:g}: {len(nodes)} nodos ({time.time() - t0:.1f} s)", flush=True)

    filas = []
    for razon in razones_numerica:
        for ang in angulos_numerica:
            for h, nodes, elements, bnd, hm in mallas:
                t0 = time.time()
                Dp_eff, Dt_eff, _ = coeficientes_efectivos(nodes, elements, bnd, razon, ang)
                filas.append({"razon": razon, "angulo": ang, "h": h, "h_medio": hm,
                              "nodos": len(nodes), "error_D_perp": Dt_eff / (D_par / razon) - 1.0,
                              "error_D_par": Dp_eff / D_par - 1.0})
                print(f"   razón {razon}, {ang:g} grados, h = {h:g}: {time.time() - t0:.1f} s", flush=True)

    reportar("B. DIFUSIÓN PERPENDICULAR NUMÉRICA (error relativo D_eff / D - 1)")
    reportar(f"   {'Dp/Dt':>6s} {'ángulo':>6s} {'h':>5s} {'h medio':>8s} {'nodos':>7s}"
             f" {'error D_perp':>13s} {'tasa':>6s} {'error D_par':>12s}")
    for razon in razones_numerica:
        for ang in angulos_numerica:
            sel = [f for f in filas if f["razon"] == razon and f["angulo"] == ang]
            e = np.abs([f["error_D_perp"] for f in sel])
            hm = np.array([f["h_medio"] for f in sel])
            tasas = np.concatenate(([np.nan], np.log(e[:-1] / e[1:]) / np.log(hm[:-1] / hm[1:])))
            for f, tasa in zip(sel, tasas):
                f["tasa"] = tasa
                reportar(f"   {razon:6g} {ang:6g} {f['h']:5.2f} {f['h_medio']:8.4f} {f['nodos']:7d}"
                         f" {f['error_D_perp']:13.4e} {('     -' if np.isnan(tasa) else f'{tasa:6.2f}')}"
                         f" {f['error_D_par']:12.4e}")
            reportar()
    guardar_csv("12_difusion_numerica.csv", filas)

    fig, ax = plt.subplots(figsize=(6.8, 5.0), constrained_layout=True)
    for razon in razones_numerica:
        for ang in angulos_numerica:
            sel = [f for f in filas if f["razon"] == razon and f["angulo"] == ang]
            ax.loglog([f["h_medio"] for f in sel], np.abs([f["error_D_perp"] for f in sel]),
                      marker="o", linestyle=LINEA_ANGULO[ang], color=COLOR_RAZON[razon],
                      markersize=7, label=rf"$D_\parallel/D_\perp$ = {razon}, B a {ang:g}°")
    hm = np.array([m[4] for m in mallas])
    ref = max(abs(f["error_D_perp"]) for f in filas if f["h"] == h_numerica[-1])
    ax.loglog(hm, 0.5 * ref * (hm / hm[-1]) ** 2, color="#6b7280", linestyle=":", linewidth=1.2,
              label="pendiente 2")
    ax.set_xticks(hm)
    ax.set_xticklabels([f"{v:.3f}" for v in hm])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("h medio de la malla uniforme (adimensional)")
    ax.set_ylabel(r"$|D_{\perp,ef} / D_\perp - 1|$")
    ax.set_title("Difusión perpendicular numérica")
    ax.legend(fontsize=8)
    viz.guardar(fig, RESULTADOS / "12_difusion_numerica")


def main():
    RESULTADOS.mkdir(exist_ok=True)
    viz.estilo()
    lineas = []

    def reportar(texto=""):
        print(texto, flush=True)
        lineas.append(texto)

    parte_A(reportar)
    parte_B(reportar)
    (RESULTADOS / "12_sensibilidad.txt").write_text("\n".join(lineas) + "\n", encoding="utf-8")
    print("Guardado en results/: 12_barrido_anisotropia y 12_difusion_numerica (csv, png, pdf),"
          " 12_sensibilidad.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
