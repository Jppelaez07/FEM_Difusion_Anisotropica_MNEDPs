"""Malla no estructurada con refinamiento en la fuente y verificaciones básicas.

Genera la malla con gmsh, reporta su calidad, ensambla M y K y verifica:
  - área total: Σ|J|/2 = 1ᵀ M 1 = Lx·Ly
  - simetría de M y K
  - K·1 ≈ 0 (el gradiente de una constante es cero)
Guarda results/malla.png y results/01_malla.txt.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from src import assembly, fields, mesh  # noqa: E402

# Parámetros (unidades adimensionales)
Lx, Ly = 20.0, 20.0
h_global, h_source, r_ref = 1.0, 0.1, 3.0
xs, ys = Lx / 2.0, Ly / 2.0
D_par, D_perp = 1.0, 0.1
alpha = 0.0                     # dirección de B (rad)

TOL = 1e-10
RESULTADOS = RAIZ / "results"


def main():
    RESULTADOS.mkdir(exist_ok=True)
    lineas = []

    def reportar(texto=""):
        print(texto)
        lineas.append(texto)

    nodes, elements, bnd, inte = mesh.make_mesh(Lx, Ly, h_global, h_source, xs, ys, r_ref)
    q = mesh.calidad_malla(nodes, elements)

    reportar("MALLA")
    reportar(f"  Dominio: [0, {Lx:g}] x [0, {Ly:g}],  fuente en ({xs:g}, {ys:g})")
    reportar(f"  h_global = {h_global:g},  h_source = {h_source:g},  r_ref = {r_ref:g}")
    reportar(f"  Nodos:              {q['n_nodos']}")
    reportar(f"    de frontera:      {len(bnd)}")
    reportar(f"    interiores:       {len(inte)}")
    reportar(f"  Elementos:          {q['n_elementos']}")
    reportar(f"  h mínimo:           {q['h_min']:.4f}")
    reportar(f"  h máximo:           {q['h_max']:.4f}")
    reportar(f"  Ángulo mínimo:      {q['angulo_min']:.2f}°")
    reportar()

    D = fields.funcion_tensor(fields.campo_uniforme(alpha), D_par, D_perp)
    M, K = assembly.ensamblar_MK(nodes, elements, D)
    detJ = mesh.determinantes(nodes, elements)
    unos = np.ones(len(nodes))

    area_J = detJ.sum() / 2.0
    area_M = unos @ (M @ unos)
    sim_M = abs(M - M.T).max()
    sim_K = abs(K - K.T).max()
    K1 = np.abs(K @ unos).max()

    # Nodos de frontera: deben estar sobre el borde del rectángulo
    x_b, y_b = nodes[bnd, 0], nodes[bnd, 1]
    en_borde = np.minimum.reduce([x_b, Lx - x_b, y_b, Ly - y_b])
    x_i, y_i = nodes[inte, 0], nodes[inte, 1]
    dist_borde_int = np.minimum.reduce([x_i, Lx - x_i, y_i, Ly - y_i])

    verificaciones = [
        ("min |J| > 0", detJ.min(), detJ.min() > 0),
        ("|Σ|J|/2 - Lx·Ly|", abs(area_J - Lx * Ly), abs(area_J - Lx * Ly) <= TOL * Lx * Ly),
        ("|1ᵀM1 - Lx·Ly|", abs(area_M - Lx * Ly), abs(area_M - Lx * Ly) <= TOL * Lx * Ly),
        ("max|M - Mᵀ|", sim_M, sim_M <= TOL),
        ("max|K - Kᵀ|", sim_K, sim_K <= TOL),
        ("max|K·1|", K1, K1 <= TOL),
        ("max dist. borde (frontera)", en_borde.max(), en_borde.max() <= 1e-12),
        ("min dist. borde (interiores)", dist_borde_int.min(), dist_borde_int.min() > 1e-12),
    ]
    reportar(f"VERIFICACIONES (tolerancia {TOL:.0e})")
    todo_ok = True
    for nombre, valor, ok in verificaciones:
        todo_ok &= bool(ok)
        reportar(f"  {nombre:<30s} {valor:12.3e}  {'OK' if ok else 'FALLA'}")
    reportar()
    reportar("RESULTADO: " + ("TODAS LAS VERIFICACIONES PASAN" if todo_ok else "HAY FALLAS"))

    # Figura de la malla
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.triplot(nodes[:, 0], nodes[:, 1], elements, color="navy", linewidth=0.3)
    ax.scatter(nodes[bnd, 0], nodes[bnd, 1], color="red", s=10, zorder=3,
               label="Nodos de frontera")
    ax.add_patch(plt.Circle((xs, ys), r_ref, fill=False, color="darkorange",
                            linestyle="--", linewidth=1.5,
                            label=f"Zona refinada (r = {r_ref:g})"))
    ax.set_xlabel("x (adimensional)")
    ax.set_ylabel("y (adimensional)")
    ax.set_title(f"Malla no estructurada: {q['n_nodos']} nodos, {q['n_elementos']} triángulos")
    ax.set_aspect("equal")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(RESULTADOS / "malla.png", dpi=200)
    plt.close(fig)

    (RESULTADOS / "01_malla.txt").write_text("\n".join(lineas) + "\n", encoding="utf-8")
    print("\nGuardado: results/malla.png, results/01_malla.txt")
    return 0 if todo_ok else 1


if __name__ == "__main__":
    sys.exit(main())
