"""Campos del modelo: campo magnético B, tensor D, fuente Q y condición inicial n0.

Todas las funciones son vectorizadas (reciben arreglos x, y).
Un campo magnético es una función B(x, y) -> (Bx, By).
"""

import numpy as np

TOL_B = 1e-14   # si |B| < TOL_B se usa D isotrópico


# --- Campos magnéticos ---

def campo_uniforme(alpha, B0=1.0):
    """Campo uniforme de magnitud B0 y ángulo alpha (radianes)."""
    Bx, By = B0 * np.cos(alpha), B0 * np.sin(alpha)

    def B(x, y):
        x = np.asarray(x, dtype=float)
        return np.full_like(x, Bx), np.full_like(x, By)
    return B


def campo_sinusoidal(A, lam):
    """B = (1, A cos(2 pi x / lam))."""
    def B(x, y):
        x = np.asarray(x, dtype=float)
        return np.ones_like(x), A * np.cos(2.0 * np.pi * x / lam)
    return B


def campo_circular(xc, yc):
    """B = (-(y - yc), x - xc): líneas circulares alrededor de (xc, yc)."""
    def B(x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        return -(y - yc), x - xc
    return B


# --- Tensor de difusión ---

def tensor_difusion(Bx, By, D_par, D_perp):
    """D = D_perp I + (D_par - D_perp) b b^T, con b = B/|B|. Retorna (P, 2, 2)."""
    if D_par <= 0 or D_perp <= 0:
        raise ValueError("Los coeficientes de difusión deben ser positivos")
    Bx = np.atleast_1d(np.asarray(Bx, dtype=float))
    By = np.atleast_1d(np.asarray(By, dtype=float))
    norma = np.hypot(Bx, By)
    nulo = norma < TOL_B
    norma_segura = np.where(nulo, 1.0, norma)
    b = np.stack((Bx / norma_segura, By / norma_segura), axis=-1)
    b[nulo] = 0.0                                   # B nulo: D = D_perp I

    D = D_perp * np.broadcast_to(np.eye(2), (b.shape[0], 2, 2)).copy()
    D += (D_par - D_perp) * np.einsum("pi,pj->pij", b, b)
    return D


def funcion_tensor(B, D_par, D_perp):
    """Crea la función D(x, y) a partir de un campo B(x, y)."""
    def D(x, y):
        Bx, By = B(x, y)
        return tensor_difusion(Bx, By, D_par, D_perp)
    return D


# --- Fuente y condición inicial ---

def gaussiana(x, y, amplitud, xc, yc, sigma):
    return amplitud * np.exp(-((x - xc) ** 2 + (y - yc) ** 2) / (2.0 * sigma ** 2))


def fuente_gaussiana(Q0, xs, ys, sigma_s, factor_temporal=None):
    """Fuente gaussiana Q(x, y, t), multiplicada por factor_temporal(t) si se da."""
    def Q(x, y, t):
        g = 1.0 if factor_temporal is None else factor_temporal(t)
        return gaussiana(x, y, Q0, xs, ys, sigma_s) * g
    return Q


def condicion_inicial_gaussiana(n_max, xs, ys, sigma):
    """Condición inicial gaussiana n0(x, y)."""
    def n0(x, y):
        return gaussiana(x, y, n_max, xs, ys, sigma)
    return n0


# --- Solución analítica ---

def solucion_gaussiana(n_max, xs, ys, sigma0, D):
    """Solución en el plano infinito con D constante (2x2) y Q = 0.

    n(x, t) = n_max sigma0^2 / sqrt(det S) exp(-r^T S^-1 r / 2), con S(t) = sigma0^2 I + 2 D t.
    """
    D = np.asarray(D, dtype=float)

    def n(x, y, t):
        S = sigma0 ** 2 * np.eye(2) + 2.0 * D * t
        Si = np.linalg.inv(S)
        rx, ry = np.asarray(x) - xs, np.asarray(y) - ys
        cuad = Si[0, 0] * rx ** 2 + 2.0 * Si[0, 1] * rx * ry + Si[1, 1] * ry ** 2
        return n_max * sigma0 ** 2 / np.sqrt(np.linalg.det(S)) * np.exp(-0.5 * cuad)
    return n
