"""Integración temporal con el método θ (sección IV de la formulación).

(M + θΔt K) N^{k+1} = (M − (1−θ)Δt K) N^k + Δt (θ F^{k+1} + (1−θ) F^k)

La matriz A = M + θΔt K se factoriza una sola vez al inicio.
"""

import numpy as np
from scipy.sparse.linalg import factorized


def metodo_theta(M, K, N0, T, dt, theta=1.0, F=None, guardar_cada=None):
    """Integra M dN/dt + K N = F(t) con N(0) = N0 en [0, T].

    Parámetros
    ----------
    M, K : matrices dispersas (restringidas a nodos interiores si es Dirichlet).
    N0 : vector inicial.
    T, dt : tiempo final y paso deseado; dt se ajusta a T/Nt para llegar a T.
    theta : 1 = Euler implícito (caso base), 0.5 = Crank-Nicolson, 0 = explícito.
    F : None (sin fuente), vector constante, o función F(t) -> vector.
    guardar_cada : guardar la solución cada tantos pasos; None = solo t=0 y t=T.

    Retorna
    -------
    tiempos, soluciones : listas con los instantes guardados y sus vectores N.
    """
    Nt = int(round(T / dt))
    dt = T / Nt

    A = (M + theta * dt * K).tocsc()
    B = (M - (1.0 - theta) * dt * K).tocsr()
    resolver = factorized(A)

    variable = callable(F)
    if variable:
        F_k = F(0.0)
    else:
        # Fuente constante: θF + (1−θ)F = F
        F_cte = np.zeros(len(N0)) if F is None else np.asarray(F, dtype=float)

    N = np.array(N0, dtype=float, copy=True)
    tiempos, soluciones = [0.0], [N.copy()]

    for k in range(Nt):
        t_sig = (k + 1) * dt
        if variable:
            F_sig = F(t_sig)
            rhs = B @ N + dt * (theta * F_sig + (1.0 - theta) * F_k)
            F_k = F_sig
        else:
            rhs = B @ N + dt * F_cte
        N = resolver(rhs)

        if (guardar_cada and (k + 1) % guardar_cada == 0) or k == Nt - 1:
            tiempos.append(t_sig)
            soluciones.append(N.copy())

    return tiempos, soluciones
