"""Ensamblaje vectorizado de M, K y F para elementos P1 (sección III de la formulación).

Notación de la formulación:
  J = [[x1-x0, y1-y0], [x2-x0, y2-y0]],   H = J⁻¹ [[-1, 1, 0], [-1, 0, 1]]
  M(e) = |J|/24 [[2,1,1],[1,2,1],[1,1,2]]
  K(e) = |J|/2 Hᵀ D H        (D evaluado en el baricentro)
  F(e)_m ≈ |J|/6 Q(baricentro, t)

Todas las matrices locales se calculan a la vez (einsum) y se ensamblan
con scipy.sparse (COO -> CSR), sin bucles de Python sobre los elementos.
"""

import numpy as np
from scipy.sparse import coo_matrix

GRAD_REF = np.array([[-1.0, 1.0, 0.0],
                     [-1.0, 0.0, 1.0]])

MASA_REF = np.array([[2.0, 1.0, 1.0],
                     [1.0, 2.0, 1.0],
                     [1.0, 1.0, 2.0]])


def geometria(nodes, elements):
    """Calcula |J| (E,), H (E, 2, 3) y baricentros (E, 2) de todos los elementos."""
    p = nodes[elements]                         # (E, 3, 2)
    J = np.stack((p[:, 1] - p[:, 0],
                  p[:, 2] - p[:, 0]), axis=1)   # (E, 2, 2), filas = ∂(x,y)/∂r, ∂(x,y)/∂s
    detJ = J[:, 0, 0] * J[:, 1, 1] - J[:, 0, 1] * J[:, 1, 0]
    if np.any(detJ <= 0):
        raise ValueError("Hay elementos con |J| <= 0; use mesh.orientar_antihorario")
    J_inv = np.empty_like(J)
    J_inv[:, 0, 0] = J[:, 1, 1]
    J_inv[:, 0, 1] = -J[:, 0, 1]
    J_inv[:, 1, 0] = -J[:, 1, 0]
    J_inv[:, 1, 1] = J[:, 0, 0]
    J_inv /= detJ[:, None, None]
    H = J_inv @ GRAD_REF                        # (E, 2, 3): columnas = ∇φ_g(e,m)
    baricentros = p.mean(axis=1)
    return detJ, H, baricentros


def _indices(elements):
    filas = np.repeat(elements, 3, axis=1).ravel()   # g(e,m) para m,n = 0..2
    cols = np.tile(elements, (1, 3)).ravel()
    return filas, cols


def ensamblar_MK(nodes, elements, D_func, geom=None):
    """Ensambla M y K globales sobre todos los nodos (matrices de Neumann).

    D_func(x, y) -> (E, 2, 2) se evalúa en los baricentros.
    geom: resultado de geometria(), para no recalcularlo.
    """
    Mtot = len(nodes)
    detJ, H, bc = geometria(nodes, elements) if geom is None else geom
    D = D_func(bc[:, 0], bc[:, 1])

    Me = (detJ / 24.0)[:, None, None] * MASA_REF            # (E, 3, 3)
    Ke = (detJ / 2.0)[:, None, None] * np.einsum("eim,eij,ejn->emn", H, D, H)

    filas, cols = _indices(elements)
    M = coo_matrix((Me.ravel(), (filas, cols)), shape=(Mtot, Mtot)).tocsr()
    K = coo_matrix((Ke.ravel(), (filas, cols)), shape=(Mtot, Mtot)).tocsr()
    for A in (M, K):
        A.sum_duplicates()
        A.eliminate_zeros()
    return M, K


def ensamblar_F(nodes, elements, Q, t, geom=None):
    """Vector de cargas global F(t) con Q evaluada en el baricentro."""
    detJ, _, bc = geometria(nodes, elements) if geom is None else geom
    Fe = detJ / 6.0 * Q(bc[:, 0], bc[:, 1], t)               # igual para m = 0, 1, 2
    return np.bincount(elements.ravel(), weights=np.repeat(Fe, 3),
                       minlength=len(nodes))


def restringir(A, idx):
    """Bloque de una matriz (o vector) asociado a los índices idx (Dirichlet)."""
    if A.ndim == 1:
        return A[idx]
    return A[idx, :][:, idx].tocsr()
