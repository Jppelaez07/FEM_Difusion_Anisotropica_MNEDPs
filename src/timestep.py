"""Integración temporal con el método theta.

(M + theta dt K) N_{k+1} = (M - (1-theta) dt K) N_k + dt (theta F_{k+1} + (1-theta) F_k)
"""

import numpy as np
from scipy.sparse.linalg import splu

from . import assembly


def _pasos_guardados(t_guardar, T, dt, Nt):
    """Pasos k más cercanos a los instantes pedidos."""
    if t_guardar is None:
        return {0, Nt}
    t_guardar = np.atleast_1d(np.asarray(t_guardar, dtype=float))
    if np.any(t_guardar < 0) or np.any(t_guardar > T * (1 + 1e-12)):
        raise ValueError("Los instantes a guardar deben estar en [0, T]")
    return set(np.rint(t_guardar / dt).astype(int).tolist())


def metodo_theta(M, K, N0, T, dt, theta=1.0, F=None, t_guardar=None):
    """Resuelve M dN/dt + K N = F desde N0 hasta T.

    theta: 1 = Euler implícito, 0.5 = Crank-Nicolson, 0 = Euler explícito.
    dt se ajusta para que T sea un múltiplo exacto.
    F: None, vector constante o función F(t).
    t_guardar: instantes a guardar (None = solo 0 y T).

    Retorna (tiempos, soluciones), con soluciones de forma (S, len(N0)).
    """
    if not 0.0 <= theta <= 1.0:
        raise ValueError("theta debe estar en [0, 1]")
    Nt = max(1, int(round(T / dt)))
    dt = T / Nt
    guardar = _pasos_guardados(t_guardar, T, dt, Nt)

    A = (M + theta * dt * K).tocsc()
    B = (M - (1.0 - theta) * dt * K).tocsr()
    lu = splu(A)                        # se factoriza una sola vez

    variable = callable(F)
    if variable:
        F_k = F(0.0)
    else:
        F_cte = np.zeros(len(N0)) if F is None else np.asarray(F, dtype=float)

    N = np.array(N0, dtype=float, copy=True)
    tiempos, soluciones = [], []
    if 0 in guardar:
        tiempos.append(0.0)
        soluciones.append(N.copy())

    for k in range(Nt):
        t_sig = (k + 1) * dt
        if variable:
            F_sig = F(t_sig)
            rhs = B @ N + dt * (theta * F_sig + (1.0 - theta) * F_k)
            F_k = F_sig
        else:
            rhs = B @ N + dt * F_cte
        N = lu.solve(rhs)

        if k + 1 in guardar:
            tiempos.append(t_sig)
            soluciones.append(N.copy())

    return np.array(tiempos), np.array(soluciones)


def simular(nodes, elements, boundary_nodes, D_func, n0, T, dt, theta=1.0,
            Q=None, bc="dirichlet", t_guardar=None):
    """Ensambla y resuelve el problema completo en la malla dada.

    D_func: D(x, y), p. ej. fields.funcion_tensor(B, D_par, D_perp).
    n0: condición inicial n0(x, y).  Q: fuente Q(x, y, t) o None.
    bc: "dirichlet" (n = 0 en la frontera) o "neumann" (flujo nulo).

    Retorna un dict con:
      t       instantes guardados
      N       solución en todos los nodos (S, Mtot); 0 en la frontera si es Dirichlet
      M, K    matrices globales completas
      libres  nodos donde se resolvió
    """
    bc = bc.lower()
    if bc not in ("dirichlet", "neumann"):
        raise ValueError('bc debe ser "dirichlet" o "neumann"')

    Mtot = len(nodes)
    geom = assembly.geometria(nodes, elements)
    M, K = assembly.ensamblar_MK(nodes, elements, D_func, geom=geom)

    if bc == "dirichlet":
        libres = np.setdiff1d(np.arange(Mtot), boundary_nodes)
        M_l = assembly.restringir(M, libres)
        K_l = assembly.restringir(K, libres)
    else:
        libres = np.arange(Mtot)
        M_l, K_l = M, K

    N0 = n0(nodes[libres, 0], nodes[libres, 1])

    F = None
    if Q is not None:
        def F(t):
            return assembly.ensamblar_F(nodes, elements, Q, t, geom=geom)[libres]

    t, N_libres = metodo_theta(M_l, K_l, N0, T, dt, theta=theta, F=F, t_guardar=t_guardar)

    N = np.zeros((len(t), Mtot))
    N[:, libres] = N_libres
    return {"t": t, "N": N, "M": M, "K": K, "libres": libres}
