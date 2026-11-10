"""Estilo común de las figuras."""

import matplotlib.pyplot as plt

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


def guardar(fig, ruta_sin_extension):
    """Guarda la figura en PNG y PDF."""
    for ext in ("png", "pdf"):
        fig.savefig(f"{ruta_sin_extension}.{ext}")
    plt.close(fig)
