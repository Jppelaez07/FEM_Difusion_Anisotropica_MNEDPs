"""Campos del modelo: B, tensor D, fuente Q y condición inicial n0.

Todas las funciones están vectorizadas: reciben arreglos x, y de igual forma.
Un campo magnético es una función B(x, y) -> (Bx, By).
"""

import numpy as np

TOL_B = 1e-14   # por debajo de esta norma se usa D isotrópico


# ---------------------------------------------------------------------------
# Campos magnéticos de ejemplo
# ---------------------------------------------------------------------------

def campo_uniforme(alpha, B0=1.0):
    """B = B0 (cos α, sin α), con α en radianes."""
    Bx, By = B0 * np.cos(alpha), B0 * np.sin(alpha)

    def B(x, y):
        x = np.asarray(x, dtype=float)
        return np.full_like(x, Bx), np.full_like(x, By)
    return B


def campo_sinusoidal(A, lam):
    """B = (1, A cos(2πx/λ)): líneas de campo onduladas en la dirección x."""
    def B(x, y):
        x = np.asarray(x, dtype=float)
        return np.ones_like(x), A * np.cos(2.0 * np.pi * x / lam)
    return B


def campo_circular(xc, yc):
    """B = (−(y − yc), x − xc): líneas de campo circulares alrededor de (xc, yc).

    En el centro ‖B‖ = 0, y ahí se usa D isotrópico.
    """
    def B(x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        return -(y - yc), x - xc
    return B


# ---------------------------------------------------------------------------
# Tensor de difusión
# ---------------------------------------------------------------------------

def tensor_difusion(Bx, By, D_par, D_perp):
    """D = D⊥ I + (D∥ − D⊥) b̂ b̂ᵀ, con b̂ = B/‖B‖, para cada punto.

    Bx, By: arreglos de forma (P,). Retorna un arreglo (P, 2, 2).
    Donde ‖B‖ < TOL_B se usa el tensor isotrópico D⊥ I.
    """
    if D_par <= 0 or D_perp <= 0:
        raise ValueError("Los coeficientes de difusión deben ser positivos")
    Bx = np.atleast_1d(np.asarray(Bx, dtype=float))
    By = np.atleast_1d(np.asarray(By, dtype=float))
    norma = np.hypot(Bx, By)
    nulo = norma < TOL_B
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


# ---------------------------------------------------------------------------
# Fuente y condición inicial
# ---------------------------------------------------------------------------

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
