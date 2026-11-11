"""Estilo común de las figuras e interpolación a rejillas regulares."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.tri import Triangulation
from scipy.sparse import csr_matrix

# Paleta Okabe-Ito (apta para daltonismo), en orden fijo por caso
PALETA_CASOS = {
    "caso1": "#000000",
    "caso2": "#E69F00",
    "caso3": "#56B4E9",
    "caso4a": "#009E73",
    "caso4b": "#0072B2",
    "caso5_dirichlet": "#D55E00",
    "caso5_neumann": "#D55E00",
}
LINEA_CASOS = {"caso3": (0, (4, 3)), "caso5_neumann": "--"}
MAPA_COLOR = "magma_r"


def estilo():
    """Tamaños de fuente y detalles comunes."""
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "lines.linewidth": 2.0,
        "savefig.dpi": 200,
    })


def matriz_rejilla(nodes, elements, xg, yg):
    """Matriz dispersa P tal que (P @ N).reshape(len(yg), len(xg)) interpola N en la rejilla.

    Usa las coordenadas baricéntricas de cada punto en su triángulo; fuera de la malla vale 0.
    """
    X, Y = np.meshgrid(xg, yg)
    px, py = X.ravel(), Y.ravel()
    e = Triangulation(nodes[:, 0], nodes[:, 1], elements).get_trifinder()(px, py)
    dentro = np.flatnonzero(e >= 0)
    tri = elements[e[dentro]]
    p0, p1, p2 = (nodes[tri[:, m]] for m in range(3))
    v0, v1 = p1 - p0, p2 - p0
    vx, vy = px[dentro] - p0[:, 0], py[dentro] - p0[:, 1]
    det = v0[:, 0] * v1[:, 1] - v1[:, 0] * v0[:, 1]
    l1 = (vx * v1[:, 1] - v1[:, 0] * vy) / det
    l2 = (v0[:, 0] * vy - vx * v0[:, 1]) / det
    pesos = np.column_stack((1.0 - l1 - l2, l1, l2))
    filas = np.repeat(dentro, 3)
    return csr_matrix((pesos.ravel(), (filas, tri.ravel())), shape=(px.size, len(nodes)))


def guardar(fig, ruta_sin_extension):
    """Guarda la figura en PNG y PDF."""
    for ext in ("png", "pdf"):
        fig.savefig(f"{ruta_sin_extension}.{ext}")
    plt.close(fig)
