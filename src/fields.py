"""Campos del modelo: B, tensor D, fuente Q y condición inicial n0.

Todas las funciones están vectorizadas: reciben arreglos x, y de igual forma.
"""

import numpy as np


def campo_uniforme(Bx, By):
    """Devuelve B(x, y) = (Bx, By) constante como función vectorizada."""
    def B(x, y):
        x = np.asarray(x, dtype=float)
        return np.full_like(x, Bx), np.full_like(x, By)
    return B


def tensor_difusion(Bx, By, D_par, D_perp):
    """D = D⊥ I + (D∥ − D⊥) b̂ b̂ᵀ, con b̂ = B/‖B‖, para cada punto.

    Bx, By: arreglos de forma (P,). Retorna un arreglo (P, 2, 2).
    Donde ‖B‖ = 0 se usa el tensor isotrópico D⊥ I.
    """
    if D_par <= 0 or D_perp <= 0:
        raise ValueError("Los coeficientes de difusión deben ser positivos")
    Bx = np.atleast_1d(np.asarray(Bx, dtype=float))
    By = np.atleast_1d(np.asarray(By, dtype=float))
    norma = np.hypot(Bx, By)
    nulo = norma == 0
    norma_segura = np.where(nulo, 1.0, norma)
    b = np.stack((Bx / norma_segura, By / norma_segura), axis=-1)   # (P, 2)
    b[nulo] = 0.0                                                    # => D = D⊥ I

    D = D_perp * np.broadcast_to(np.eye(2), (b.shape[0], 2, 2)).copy()
    D += (D_par - D_perp) * np.einsum("pi,pj->pij", b, b)
    return D


def funcion_tensor(B, D_par, D_perp):
    """Crea D(x, y) -> (P, 2, 2) a partir de un campo B(x, y)."""
    def D(x, y):
        Bx, By = B(x, y)
        return tensor_difusion(Bx, By, D_par, D_perp)
    return D


def gaussiana(x, y, amplitud, xc, yc, sigma):
    return amplitud * np.exp(-((x - xc) ** 2 + (y - yc) ** 2) / (2.0 * sigma ** 2))


def fuente_gaussiana(Q0, xs, ys, sigma_s, factor_temporal=None):
    """Q(x, y, t) = Q0 exp(-r²/(2σ_s²)) · g(t); g(t) = 1 si no se da."""
    def Q(x, y, t):
        g = 1.0 if factor_temporal is None else factor_temporal(t)
        return gaussiana(x, y, Q0, xs, ys, sigma_s) * g
    return Q


def condicion_inicial_gaussiana(n_max, xs, ys, sigma):
    """n0(x, y) = n_max exp(-((x-xs)² + (y-ys)²)/(2σ²))."""
    def n0(x, y):
        return gaussiana(x, y, n_max, xs, ys, sigma)
    return n0
