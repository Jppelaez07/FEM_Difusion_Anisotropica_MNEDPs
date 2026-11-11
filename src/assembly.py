"""Ensamblaje vectorizado de M, K y F para elementos P1.

Matrices locales:
  M_e = |J|/24 [[2,1,1],[1,2,1],[1,1,2]]
  K_e = |J|/2 H^T D H     (D en el baricentro)
  F_e = |J|/6 Q           (Q en el baricentro)
"""

import numpy as np
from scipy.sparse import coo_matrix

GRAD_REF = np.array([[-1.0, 1.0, 0.0],
                     [-1.0, 0.0, 1.0]])

MASA_REF = np.array([[2.0, 1.0, 1.0],
                     [1.0, 2.0, 1.0],
                     [1.0, 1.0, 2.0]])


def geometria(nodes, elements):
    """Retorna |J| (E,), H (E, 2, 3) y baricentros (E, 2) de todos los elementos."""
    p = nodes[elements]                         # (E, 3, 2)
    J = np.stack((p[:, 1] - p[:, 0],
                  p[:, 2] - p[:, 0]), axis=1)   # (E, 2, 2)
    detJ = J[:, 0, 0] * J[:, 1, 1] - J[:, 0, 1] * J[:, 1, 0]
    if np.any(detJ <= 0):
        raise ValueError("Hay elementos con |J| <= 0; use mesh.orientar_antihorario")
    J_inv = np.empty_like(J)
    J_inv[:, 0, 0] = J[:, 1, 1]
    J_inv[:, 0, 1] = -J[:, 0, 1]
    J_inv[:, 1, 0] = -J[:, 1, 0]
    J_inv[:, 1, 1] = J[:, 0, 0]
    J_inv /= detJ[:, None, None]
    H = J_inv @ GRAD_REF                        # gradientes de las funciones de forma
    baricentros = p.mean(axis=1)
    return detJ, H, baricentros


def _indices(elements):
    """Índices globales (fila, columna) de las 9 entradas de cada matriz local."""
    filas = np.repeat(elements, 3, axis=1).ravel()
    cols = np.tile(elements, (1, 3)).ravel()
    return filas, cols


def ensamblar_MK(nodes, elements, D_func, geom=None):
    """M y K globales sobre todos los nodos.

    D_func(x, y) -> (E, 2, 2). geom: resultado de geometria(), si ya se calculó.
    """
    Mtot = len(nodes)
    detJ, H, baric = geometria(nodes, elements) if geom is None else geom
    D = D_func(baric[:, 0], baric[:, 1])

    Me = (detJ / 24.0)[:, None, None] * MASA_REF
    Ke = (detJ / 2.0)[:, None, None] * np.einsum("eim,eij,ejn->emn", H, D, H)

    filas, cols = _indices(elements)
    M = coo_matrix((Me.ravel(), (filas, cols)), shape=(Mtot, Mtot)).tocsr()
    K = coo_matrix((Ke.ravel(), (filas, cols)), shape=(Mtot, Mtot)).tocsr()
    for A in (M, K):
        A.sum_duplicates()
        A.eliminate_zeros()
    return M, K


def ensamblar_F(nodes, elements, Q, t, geom=None):
    """Vector de cargas global F en el instante t."""
    detJ, _, baric = geometria(nodes, elements) if geom is None else geom
    Fe = detJ / 6.0 * Q(baric[:, 0], baric[:, 1], t)
    return np.bincount(elements.ravel(), weights=np.repeat(Fe, 3),
                       minlength=len(nodes))


def restringir(A, idx):
    """Submatriz (o subvector) con las filas y columnas idx."""
    if A.ndim == 1:
        return A[idx]
    return A[idx, :][:, idx].tocsr()
