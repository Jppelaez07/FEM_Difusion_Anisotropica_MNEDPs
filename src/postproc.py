"""Postproceso: masa, momentos, mínimo de n y errores en normas L2 y H1.

Masa y momentos se calculan con la matriz de masa global M.
N puede ser un vector (Mtot,) o varios instantes (S, Mtot).
"""

import numpy as np
from matplotlib.tri import LinearTriInterpolator, Triangulation

from . import assembly

# Cuadratura de Dunavant de 6 puntos (exacta para polinomios de grado 4).
# Coordenadas baricéntricas (l0, l1, l2) y pesos que suman 1.
_A1, _W1 = 0.445948490915964886, 0.223381589678011466
_A2, _W2 = 0.091576213509770743, 0.109951743655321868
_B1, _B2 = 1.0 - 2.0 * _A1, 1.0 - 2.0 * _A2
CUAD_BARI = np.array([[_A1, _A1, _B1], [_A1, _B1, _A1], [_B1, _A1, _A1],
                      [_A2, _A2, _B2], [_A2, _B2, _A2], [_B2, _A2, _A2]])
CUAD_PESOS = np.array([_W1, _W1, _W1, _W2, _W2, _W2])


def errores_L2_H1(nodes, elements, N, n_exacta, grad_exacta):
    """Error en norma L2 y seminorma H1 entre la solución FEM N y la exacta.

    n_exacta(x, y) -> valores; grad_exacta(x, y) -> (dn/dx, dn/dy).
    Retorna (error_L2, error_H1).
    """
    detJ, H, _ = assembly.geometria(nodes, elements)
    area = detJ / 2.0
    p = nodes[elements]                                   # (E, 3, 2)
    xq = np.einsum("qm,emi->eqi", CUAD_BARI, p)            # puntos de cuadratura (E, 6, 2)
    Ne = np.asarray(N, dtype=float)[elements]              # (E, 3)

    # L2: n_h en los puntos de cuadratura
    nh = Ne @ CUAD_BARI.T                                  # (E, 6)
    e2 = (nh - n_exacta(xq[..., 0], xq[..., 1])) ** 2
    err_L2 = np.sqrt(np.sum(area * (e2 @ CUAD_PESOS)))

    # H1: el gradiente de n_h es constante en cada elemento
    grad_h = np.einsum("eim,em->ei", H, Ne)                # (E, 2)
    gx, gy = grad_exacta(xq[..., 0], xq[..., 1])
    g2 = (grad_h[:, None, 0] - gx) ** 2 + (grad_h[:, None, 1] - gy) ** 2
    err_H1 = np.sqrt(np.sum(area * (g2 @ CUAD_PESOS)))
    return err_L2, err_H1


def evaluar_en_puntos(nodes, elements, N, xp, yp):
    """Valor de la solución FEM en puntos arbitrarios (NaN fuera de la malla)."""
    tri = Triangulation(nodes[:, 0], nodes[:, 1], elements)
    valores = LinearTriInterpolator(tri, np.asarray(N, dtype=float))(xp, yp)
    return np.ma.filled(valores.astype(float), np.nan)


def norma_L2_discreta(M, v):
    """Norma L2 de una función FEM con valores nodales v: sqrt(v^T M v)."""
    v = np.asarray(v, dtype=float)
    return np.sqrt(v @ (M @ v))


def _MN(M, N):
    N = np.asarray(N, dtype=float)
    return (M @ N.T).T


def masa(M, N):
    """Masa total 1^T M N."""
    return _MN(M, N).sum(axis=-1)


def centro_de_masa(nodes, M, N):
    """Centro de masa (xc, yc)."""
    MN = _MN(M, N)
    m = MN.sum(axis=-1)
    return MN @ nodes[:, 0] / m, MN @ nodes[:, 1] / m


def momentos(nodes, M, N):
    """Masa, centro de masa y momentos de segundo orden respecto a él.

    Retorna un dict con masa, xc, yc, xx = <(x-xc)^2>, yy = <(y-yc)^2>,
    xy = <(x-xc)(y-yc)>.
    """
    MN = _MN(M, N)
    m = MN.sum(axis=-1)
    x, y = nodes[:, 0], nodes[:, 1]
    xc = MN @ x / m
    yc = MN @ y / m
    dx = x - np.expand_dims(xc, -1)
    dy = y - np.expand_dims(yc, -1)
    return {
        "masa": m,
        "xc": xc,
        "yc": yc,
        "xx": (MN * dx * dx).sum(axis=-1) / m,
        "yy": (MN * dy * dy).sum(axis=-1) / m,
        "xy": (MN * dx * dy).sum(axis=-1) / m,
    }


def angulo_principal(mom):
    """Ángulo del eje mayor de la nube, en grados."""
    return 0.5 * np.degrees(np.arctan2(2.0 * mom["xy"], mom["xx"] - mom["yy"]))


def minimo(N):
    """Mínimo de n en cada instante."""
    return np.asarray(N).min(axis=-1)
