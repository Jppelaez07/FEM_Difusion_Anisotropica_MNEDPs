"""Integración temporal con el método θ (sección IV de la formulación).

(M + θΔt K) N^{k+1} = (M − (1−θ)Δt K) N^k + Δt (θ F^{k+1} + (1−θ) F^k)

La matriz A = M + θΔt K se factoriza una sola vez (LU dispersa, splu).
"""

import numpy as np
from scipy.sparse.linalg import splu

from . import assembly


def _pasos_guardados(t_guardar, T, dt, Nt):
    """Índices de paso k (t_k = kΔt) más cercanos a los instantes pedidos."""
    if t_guardar is None:
        return {0, Nt}
    t_guardar = np.atleast_1d(np.asarray(t_guardar, dtype=float))
    if np.any(t_guardar < 0) or np.any(t_guardar > T * (1 + 1e-12)):
        raise ValueError("Los instantes a guardar deben estar en [0, T]")
    return set(np.rint(t_guardar / dt).astype(int).tolist())


def metodo_theta(M, K, N0, T, dt, theta=1.0, F=None, t_guardar=None):
    """Integra M dN/dt + K N = F(t) con N(0) = N0 en [0, T].

    Parámetros
    ----------
    M, K : matrices dispersas del sistema que se resuelve.
    N0 : vector inicial.
    T, dt : tiempo final y paso deseado; dt se ajusta a T/Nt para llegar a T.
    theta : 1 = Euler implícito (caso base), 0.5 = Crank-Nicolson, 0 = explícito.
    F : None (sin fuente), vector constante, o función F(t) -> vector.
    t_guardar : instantes a guardar (se redondean al paso más cercano);
        None = solo t = 0 y t = T.

    Retorna
    -------
    tiempos : arreglo (S,) de instantes guardados.
    soluciones : arreglo (S, len(N0)) con N en esos instantes.
    """
    if not 0.0 <= theta <= 1.0:
        raise ValueError("theta debe estar en [0, 1]")
    Nt = max(1, int(round(T / dt)))
    dt = T / Nt
    guardar = _pasos_guardados(t_guardar, T, dt, Nt)

    A = (M + theta * dt * K).tocsc()
    B = (M - (1.0 - theta) * dt * K).tocsr()
    lu = splu(A)

    variable = callable(F)
    if variable:
        F_k = F(0.0)
    else:
        # Fuente constante: θF + (1−θ)F = F
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
    """Resuelve el problema completo en la malla dada.

    Parámetros
    ----------
    nodes, elements, boundary_nodes : malla (ver mesh.py).
    D_func : D(x, y) -> (P, 2, 2), p. ej. fields.funcion_tensor(B, D∥, D⊥).
    n0 : condición inicial n0(x, y).
    Q : fuente Q(x, y, t), o None si no hay fuente.
    bc : "dirichlet" (n = 0 en ∂Ω; se resuelve el bloque de nodos interiores)
         o "neumann" (flujo nulo; se usan las matrices completas).
    T, dt, theta, t_guardar : ver metodo_theta.

    Retorna
    -------
    dict con
      t : instantes guardados (S,)
      N : solución nodal global (S, Mtot); con Dirichlet vale 0 en la frontera
      M, K : matrices globales completas (útiles para el postproceso)
      libres : índices de los nodos donde se resolvió
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
