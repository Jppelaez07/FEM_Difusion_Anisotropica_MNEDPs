"""Postproceso: masa total, momentos de segundo orden y mínimo de n.

Las integrales se calculan con la matriz de masa global M (completa):
  ∫ g n_h dΩ ≈ gᵀ M N,
donde g es el vector de valores nodales de g. Es exacto si g es lineal
(p. ej. g = 1, x, y); para g cuadrática el error es O(h²).

Todas las funciones aceptan N de forma (Mtot,) o (S, Mtot) (varios instantes).
"""

import numpy as np


def _MN(M, N):
    N = np.asarray(N, dtype=float)
    return (M @ N.T).T          # (Mtot,) o (S, Mtot)


def masa(M, N):
    """Masa total 1ᵀ M N = ∫ n_h dΩ."""
    return _MN(M, N).sum(axis=-1)


def centro_de_masa(nodes, M, N):
    """(x̄, ȳ) = (∫ x n dΩ, ∫ y n dΩ) / ∫ n dΩ."""
    MN = _MN(M, N)
    m = MN.sum(axis=-1)
    return MN @ nodes[:, 0] / m, MN @ nodes[:, 1] / m


def momentos(nodes, M, N):
    """Momentos de segundo orden respecto al centro de masa.

    Retorna un dict con masa, xc, yc, xx = ⟨(x−x̄)²⟩, yy = ⟨(y−ȳ)²⟩,
    xy = ⟨(x−x̄)(y−ȳ)⟩, donde ⟨g⟩ = ∫ g n dΩ / ∫ n dΩ.
    """
    MN = _MN(M, N)
    m = MN.sum(axis=-1)
    x, y = nodes[:, 0], nodes[:, 1]
    xc = MN @ x / m
    yc = MN @ y / m
    dx = x - np.expand_dims(xc, -1)          # (Mtot,) o (S, Mtot)
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
    """Ángulo (grados) del eje mayor de la nube, a partir de los momentos."""
    return 0.5 * np.degrees(np.arctan2(2.0 * mom["xy"], mom["xx"] - mom["yy"]))


def minimo(N):
    """Mínimo de n en cada instante (para vigilar la positividad)."""
    return np.asarray(N).min(axis=-1)
