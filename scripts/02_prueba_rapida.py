"""Prueba rápida del solver: difusión anisotrópica de una gaussiana con Neumann.

Gaussiana inicial, Q = 0, flujo nulo en la frontera, D∥ = 1, D⊥ = 0.1 y
campo uniforme a 30°. Se verifica:
  - conservación de la masa 1ᵀ M N (variación relativa < 1e-10)
  - elongación de la nube en la dirección de B (ángulo del eje mayor ≈ 30°)
  - momentos frente a la solución en dominio infinito: Σ(t) = σ0² I + 2 D t
Guarda results/02_prueba_rapida.png y results/02_prueba_rapida.txt.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from src import fields, mesh, postproc, timestep  # noqa: E402

# Parámetros (unidades adimensionales)
Lx, Ly = 20.0, 20.0
xs, ys = Lx / 2.0, Ly / 2.0
h_global, h_source, r_ref = 1.0, 0.1, 3.0
D_par, D_perp = 1.0, 0.1
alpha_grados = 30.0
n_max, sigma0 = 1.0, 1.0
T, dt, theta = 4.0, 0.01, 1.0
t_figura = [0.0, 0.5, 1.5, 4.0]
t_guardar = np.round(np.arange(0.0, T + dt / 2, 0.1), 10)

TOL_MASA = 1e-10
RESULTADOS = RAIZ / "results"


def main():
    RESULTADOS.mkdir(exist_ok=True)
    lineas = []

    def reportar(texto=""):
        print(texto)
        lineas.append(texto)

    alpha = np.radians(alpha_grados)
    nodes, elements, bnd, _ = mesh.make_mesh(Lx, Ly, h_global, h_source, xs, ys, r_ref)
    B = fields.campo_uniforme(alpha)
    D = fields.funcion_tensor(B, D_par, D_perp)
    n0 = fields.condicion_inicial_gaussiana(n_max, xs, ys, sigma0)

    res = timestep.simular(nodes, elements, bnd, D, n0, T, dt, theta=theta,
                           Q=None, bc="neumann", t_guardar=t_guardar)
    t, N, M = res["t"], res["N"], res["M"]
    mom = postproc.momentos(nodes, M, N)
    masa = mom["masa"]
    var_masa = np.abs(masa - masa[0]) / masa[0]
    angulo = postproc.angulo_principal(mom)
    n_min = postproc.minimo(N)

    # Dominio infinito con D constante: Σ(t) = σ0² I + 2 D t
    D0 = D(np.array([xs]), np.array([ys]))[0]
    teo = {k: sigma0 ** 2 * (i == j) + 2.0 * D0[i, j] * t
           for k, (i, j) in {"xx": (0, 0), "yy": (1, 1), "xy": (0, 1)}.items()}

    reportar("PRUEBA RÁPIDA: gaussiana, Q = 0, Neumann, campo uniforme")
    reportar(f"  Malla: {len(nodes)} nodos, {len(elements)} triángulos")
    reportar(f"  D∥ = {D_par:g}, D⊥ = {D_perp:g}, α = {alpha_grados:g}°, σ0 = {sigma0:g}")
    reportar(f"  θ = {theta:g}, Δt = {dt:g}, T = {T:g}")
    reportar()
    reportar(f"  {'t':>5s} {'masa':>14s} {'⟨x²⟩':>8s} {'teo':>7s} {'⟨y²⟩':>8s} {'teo':>7s}"
             f" {'⟨xy⟩':>8s} {'teo':>7s} {'ángulo':>8s} {'min n':>11s}")
    for k in np.flatnonzero(np.isin(np.round(t, 10), [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0])):
        reportar(f"  {t[k]:5.2f} {masa[k]:14.10f} {mom['xx'][k]:8.4f} {teo['xx'][k]:7.4f}"
                 f" {mom['yy'][k]:8.4f} {teo['yy'][k]:7.4f} {mom['xy'][k]:8.4f} {teo['xy'][k]:7.4f}"
                 f" {(f'{angulo[k]:7.2f}°' if k > 0 else '       —')} {n_min[k]:11.3e}")
    reportar()

    ok_masa = var_masa.max() < TOL_MASA
    ok_ang = abs(angulo[-1] - alpha_grados) < 1.0
    reportar("VERIFICACIONES")
    reportar(f"  Variación relativa máx. de la masa: {var_masa.max():.3e}"
             f"  (< {TOL_MASA:.0e}: {'OK' if ok_masa else 'FALLA'})")
    reportar(f"  Ángulo del eje mayor en t = {T:g}:  {angulo[-1]:.2f}°"
             f"  (esperado {alpha_grados:g}°: {'OK' if ok_ang else 'FALLA'})")
    err_mom = max(abs(mom[c][-1] - teo[c][-1]) / teo["xx"][-1] for c in ("xx", "yy", "xy"))
    reportar(f"  Error relativo de los momentos en t = {T:g} frente a dominio infinito: {err_mom:.2e}")
    reportar(f"  Mínimo de n en toda la simulación: {n_min.min():.3e}")
    reportar()
    reportar("RESULTADO: " + ("OK" if ok_masa and ok_ang else "HAY FALLAS"))

    # --- Figura: 4 instantes ---
    fig, axes = plt.subplots(2, 2, figsize=(10, 9.4), constrained_layout=True)
    largo = 9.0
    dx, dy = largo * np.cos(alpha), largo * np.sin(alpha)
    for ax, tf in zip(axes.ravel(), t_figura):
        k = int(np.argmin(np.abs(t - tf)))
        niveles = np.linspace(0.0, N[k].max(), 31)     # valores < 0 (≈ -5e-5) van al color mínimo
        cf = ax.tricontourf(nodes[:, 0], nodes[:, 1], elements, N[k], levels=niveles,
                            cmap="Blues", extend="min")
        fig.colorbar(cf, ax=ax, shrink=0.85, label="n (adimensional)", format="%.2f")
        ax.plot([xs - dx, xs + dx], [ys - dy, ys + dy], color="#c2410c", linestyle="--",
                linewidth=1.2, label=f"Dirección de B ({alpha_grados:g}°)")
        # Elipse de 2 desviaciones estándar a partir de los momentos numéricos
        cov = np.array([[mom["xx"][k], mom["xy"][k]], [mom["xy"][k], mom["yy"][k]]])
        val, vec = np.linalg.eigh(cov)
        ax.add_patch(Ellipse((mom["xc"][k], mom["yc"][k]), 4 * np.sqrt(val[1]),
                             4 * np.sqrt(val[0]),
                             angle=np.degrees(np.arctan2(vec[1, 1], vec[0, 1])),
                             fill=False, color="#111827", linewidth=1.2,
                             label="Elipse 2σ (momentos)"))
        ax.set_title(f"t = {t[k]:.1f}   (eje mayor a {angulo[k]:.1f}°)" if k > 0
                     else "t = 0.0   (condición inicial)")
        ax.set_xlim(1, 19)
        ax.set_ylim(1, 19)
        ax.set_aspect("equal")
        ax.set_xlabel("x (adimensional)")
        ax.set_ylabel("y (adimensional)")
    axes[0, 0].legend(loc="upper left", fontsize=8, framealpha=0.9)
    fig.suptitle(f"Difusión anisotrópica con Neumann: D∥ = {D_par:g}, D⊥ = {D_perp:g}, "
                 f"B uniforme a {alpha_grados:g}°")
    fig.savefig(RESULTADOS / "02_prueba_rapida.png", dpi=200)
    plt.close(fig)

    (RESULTADOS / "02_prueba_rapida.txt").write_text("\n".join(lineas) + "\n", encoding="utf-8")
    print("\nGuardado: results/02_prueba_rapida.png, results/02_prueba_rapida.txt")
    return 0 if ok_masa and ok_ang else 1


if __name__ == "__main__":
    sys.exit(main())
