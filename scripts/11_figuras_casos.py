"""Figuras finales de los escenarios (requiere correr antes scripts/10_casos.py).

1. Panel 2x3 con el estado final de cada caso, misma escala de colores e isolíneas.
2. Masa total y flujo de salida / entrada contra el tiempo, con t_99.
3. Balance de masa del caso 5 (Dirichlet y Neumann).
4. Casos 1-3 frente a la solución exacta en el plano infinito.
Guarda PNG y PDF en results/casos/.
"""

import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from src import casos, fields, postproc, viz  # noqa: E402

PANELES = ["caso1", "caso2", "caso3", "caso4a", "caso4b", "caso5_dirichlet"]
ETIQUETAS = {
    "caso1": "Caso 1", "caso2": "Caso 2", "caso3": "Caso 3", "caso4a": "Caso 4a",
    "caso4b": "Caso 4b", "caso5_dirichlet": "Caso 5 (Dirichlet)",
    "caso5_neumann": "Caso 5 (Neumann)",
}
DECADAS = 3                                   # rango de la escala logarítmica
ISOLINEAS = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]


def titulo_figura(texto):
    return texto.replace("D_par/D_perp", r"$D_\parallel/D_\perp$")


def figura_estados_finales(datos):
    vmax = max(datos[n]["N_guardados"][-1].max() for n in PANELES)
    vmin = vmax * 10.0 ** (-DECADAS)
    niveles = np.logspace(np.log10(vmin), np.log10(vmax), 31)
    norma = LogNorm(vmin=vmin, vmax=vmax)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10), constrained_layout=True)
    for ax, nombre in zip(axes.ravel(), PANELES):
        d = datos[nombre]
        c = d["config"]
        x, y, tri = d["nodes"][:, 0], d["nodes"][:, 1], d["elements"]
        N = np.clip(d["N_guardados"][-1], vmin, None)          # valores <= vmin al color mínimo
        cf = ax.tricontourf(x, y, tri, N, levels=niveles, norm=norma, cmap=viz.MAPA_COLOR,
                            extend="min")
        cs = ax.tricontour(x, y, tri, N, levels=ISOLINEAS, colors="#6b7280", linewidths=0.7)
        ax.clabel(cs, fmt="%g", fontsize=8)
        f = c["fuente"]
        ax.plot(f["xs"], f["ys"], marker="+", color="#22d3ee", markersize=10, mew=2)
        if c["campo"]["tipo"] == "circular":
            r = np.hypot(f["xs"] - c["campo"]["xc"], f["ys"] - c["campo"]["yc"])
            ax.add_patch(plt.Circle((c["campo"]["xc"], c["campo"]["yc"]), r, fill=False,
                                    color="#22d3ee", linestyle="--", linewidth=1))
        ax.set_title(titulo_figura(c["titulo"]) + f"\nt = {d['t_guardados'][-1]:g}")
        ax.set_aspect("equal")
        ax.set_xlim(0, c["dominio"]["Lx"])
        ax.set_ylim(0, c["dominio"]["Ly"])
        ax.set_xlabel("x (adimensional)")
        ax.set_ylabel("y (adimensional)")
        ax.grid(False)
    cbar = fig.colorbar(cf, ax=axes.ravel().tolist(), shrink=0.8)
    cbar.set_label("n (adimensional, escala logarítmica)")
    cbar.set_ticks([v for v in ISOLINEAS if vmin <= v <= vmax])
    cbar.set_ticklabels([f"{v:g}" for v in ISOLINEAS if vmin <= v <= vmax])
    fig.suptitle("Estado final de cada caso. Isolíneas en n = "
                 + ", ".join(f"{v:g}" for v in ISOLINEAS) + ";  + = fuente", fontsize=13)
    viz.guardar(fig, casos.DIR_CASOS / "estados_finales")


def figura_masa_flujo(datos):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    for nombre, d in datos.items():
        estilo = dict(color=viz.PALETA_CASOS[nombre], linestyle=viz.LINEA_CASOS.get(nombre, "-"),
                      label=ETIQUETAS[nombre])
        axes[0].plot(d["t"], d["masa"], **estilo)
        f = d["config"]["fuente"]
        if f["t_apagado"] is None:
            entrada = f["Q0"] * 2 * np.pi * f["sigma"] ** 2
            axes[1].plot(d["t"], d["flujo_directo"] / entrada, **estilo)
            if not np.isnan(d["t_99_directo"]):
                axes[1].plot(d["t_99_directo"], 0.99, "o", color=estilo["color"], markersize=7,
                             markeredgecolor="white", zorder=4)
    c = datos["caso1"]["config"]["fuente"]
    t = datos["caso1"]["t"]
    axes[0].plot(t, c["Q0"] * 2 * np.pi * c["sigma"] ** 2 * t, color="#9ca3af", linestyle=":",
                 linewidth=1.5, label="Entrada acumulada (fuente continua)")
    axes[0].set_title("Masa total")
    axes[0].set_ylabel(r"$\mathbf{1}^T M N$ (adimensional)")
    axes[1].axhline(0.99, color="#9ca3af", linestyle=":", linewidth=1.5)
    axes[1].annotate("0.99 (puntos: t_99)", (2, 0.99), xytext=(0, 4), textcoords="offset points",
                     fontsize=9, color="#6b7280")
    axes[1].set_title(r"Flujo directo de salida / entrada ($Q_0\,2\pi\sigma_Q^2$), fuente continua")
    axes[1].set_ylabel("Salida / entrada")
    for ax in axes:
        ax.set_xlabel("t (adimensional)")
    axes[0].legend(ncol=2)
    viz.guardar(fig, casos.DIR_CASOS / "masa_flujo")


def figura_comparacion_exacta(datos):
    with open(casos.DIR_CASOS / "comparacion_exacta.csv", encoding="utf-8") as fh:
        filas = list(csv.DictReader(fh))
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)

    ax_err, ax_raz = axes[0]
    for nombre in ("caso1", "caso2", "caso3"):
        sel = [f for f in filas if f["caso"] == nombre]
        t = np.array([float(f["t"]) for f in sel])
        color = viz.PALETA_CASOS[nombre]
        ax_err.loglog(t, [float(f["error_L2_rel"]) for f in sel], "o-", color=color,
                      markersize=6, label=ETIQUETAS[nombre])
        ax_raz.semilogx(t, [float(f["razon_fem"]) for f in sel], "o-", color=color, markersize=6,
                        label=f"{ETIQUETAS[nombre]}: FEM (Dirichlet)")
        ax_raz.semilogx(t, [float(f["razon_exacta"]) for f in sel], "s--", color=color,
                        markersize=6, markerfacecolor="white", label=f"{ETIQUETAS[nombre]}: exacta (plano)")
        ax_raz.axhline(float(sel[0]["razon_teorica"]), color=color, linestyle=":", linewidth=1.2)
    ax_err.set_title("Error L2 relativo de FEM frente a la solución en el plano infinito")
    ax_err.set_xlabel("t (adimensional)")
    ax_err.set_ylabel("Error L2 relativo")
    ax_err.legend()
    ax_raz.set_title(r"Razón de aspecto de la isolínea 0.1 max n (punteada: $\sqrt{D_\parallel/D_\perp}$)")
    ax_raz.set_xlabel("t (adimensional)")
    ax_raz.set_ylabel("Radio a lo largo / radio a través")
    ax_raz.legend(fontsize=8, ncol=1)

    # Cortes del caso 3 a lo largo y a través de B
    d = datos["caso3"]
    c = d["config"]
    f = c["fuente"]
    a = np.radians(c["campo"]["alpha_grados"])
    D0 = fields.tensor_difusion(np.cos(a), np.sin(a), c["D_par"], c["D_perp"])[0]
    n_ex = fields.solucion_fuente_continua(f["Q0"], f["xs"], f["ys"], f["sigma"], D0)
    s = np.linspace(-10.0, 10.0, 801)
    colores_t = {5.0: "#0072B2", 40.0: "#D55E00"}
    for ax, ang, titulo in ((axes[1, 0], a, "a lo largo de B"), (axes[1, 1], a + np.pi / 2, "a través de B")):
        xp, yp = f["xs"] + s * np.cos(ang), f["ys"] + s * np.sin(ang)
        for tc, col in colores_t.items():
            k = int(np.argmin(np.abs(d["t_comparacion"] - tc)))
            ax.plot(s, postproc.evaluar_en_puntos(d["nodes"], d["elements"], d["N_comparacion"][k],
                                                  xp, yp), color=col, label=f"FEM, t = {tc:g}")
            ax.plot(s, n_ex(xp, yp, tc), color="#111827", linestyle="--", linewidth=1.2,
                    label=f"Exacta (plano), t = {tc:g}")
        ax.set_title(f"Caso 3: corte {titulo}")
        ax.set_xlabel("Distancia a la fuente (adimensional)")
        ax.set_ylabel("n (adimensional)")
        if ang != a:
            ax.set_xlim(-4, 4)
    axes[1, 0].legend()
    viz.guardar(fig, casos.DIR_CASOS / "comparacion_exacta")


def figura_balance_caso5(datos):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), constrained_layout=True, sharey=True)
    color = viz.PALETA_CASOS["caso5_dirichlet"]
    d = datos["caso5_dirichlet"]
    ax = axes[0]
    ax.plot(d["t"], d["entrada_acum"], color="#111827", linewidth=3, alpha=0.35,
            label=r"Entrada acumulada $\int\int Q\,d\Omega\,dt$")
    ax.plot(d["t"], d["masa"], color=color, label="Masa en el dominio")
    ax.plot(d["t"], d["salida_directa_acum"], color="#0072B2",
            label="Salida acumulada por la frontera")
    ax.plot(d["t"], d["masa"] + d["salida_directa_acum"], color="#111827", linestyle="--",
            linewidth=1.5, label="Masa + salida")
    ax.set_title("Caso 5, Dirichlet")
    ax.set_ylabel("Masa (adimensional)")
    ax.legend()

    d = datos["caso5_neumann"]
    ax = axes[1]
    ax.plot(d["t"], d["entrada_acum"], color="#111827", linewidth=3, alpha=0.35,
            label=r"Entrada acumulada $\int\int Q\,d\Omega\,dt$")
    ax.plot(d["t"], d["masa"], color=color, linestyle="--", label="Masa en el dominio")
    ax.set_title("Caso 5, Neumann (sin pérdidas)")
    ax.legend()
    for ax in axes:
        ax.set_xlabel("t (adimensional)")
        ax.axvline(d["config"]["fuente"]["t_apagado"], color="#9ca3af", linewidth=1)
    fig.suptitle("Balance de masa con fuente pulsada (encendida hasta t = "
                 f"{d['config']['fuente']['t_apagado']:g})")
    viz.guardar(fig, casos.DIR_CASOS / "balance_caso5")


def main():
    viz.estilo()
    faltan = [n for n in casos.CASOS if not (casos.DIR_CASOS / f"{n}.npz").exists()]
    if faltan:
        print("Faltan resultados; corra primero scripts/10_casos.py:", ", ".join(faltan))
        return 1
    datos = {n: casos.cargar_caso(n) for n in casos.CASOS}
    figura_estados_finales(datos)
    figura_masa_flujo(datos)
    figura_balance_caso5(datos)
    figura_comparacion_exacta(datos)
    print("Guardado en results/casos/: estados_finales, masa_flujo, balance_caso5,"
          " comparacion_exacta (png y pdf)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
