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


def error_L2_relativo(nodes, elements, N, n_exacta):
    """||n_h - n|| / ||n|| en norma L2, con la cuadratura de 6 puntos."""
    detJ, _, _ = assembly.geometria(nodes, elements)
    xq = np.einsum("qm,emi->eqi", CUAD_BARI, nodes[elements])
    nh = np.asarray(N, dtype=float)[elements] @ CUAD_BARI.T
    ne = n_exacta(xq[..., 0], xq[..., 1])
    pesos = (detJ / 2.0)[:, None] * CUAD_PESOS
    return np.sqrt(np.sum(pesos * (nh - ne) ** 2) / np.sum(pesos * ne ** 2))


def norma_L2_discreta(M, v):
    """Norma L2 de una función FEM con valores nodales v: sqrt(v^T M v)."""
    v = np.asarray(v, dtype=float)
    return np.sqrt(v @ (M @ v))


def _MN(M, N):
    """Producto M N para uno o varios instantes."""
    N = np.asarray(N, dtype=float)
    return (M @ N.T).T


def masa(M, N):
    """Masa total 1^T M N."""
    return _MN(M, N).sum(axis=-1)


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


def razon_aspecto(mom):
    """sqrt(l1 / l2), con l1 >= l2 los valores propios de la matriz de momentos."""
    xx, yy, xy = (np.asarray(mom[k], dtype=float) for k in ("xx", "yy", "xy"))
    media = 0.5 * (xx + yy)
    radio = np.sqrt((0.5 * (xx - yy)) ** 2 + xy ** 2)
    return np.sqrt((media + radio) / (media - radio))


def razon_aspecto_isolinea(nodes, elements, N, nivel, xc, yc, angulo, r_max, n_puntos=2001):
    """Razón de aspecto de la isolínea n = nivel de la solución FEM (ver razon_aspecto_curva)."""
    tri = Triangulation(nodes[:, 0], nodes[:, 1], elements)
    interp = LinearTriInterpolator(tri, np.asarray(N, dtype=float))
    f = lambda xp, yp: np.ma.filled(interp(xp, yp).astype(float), np.nan)
    return razon_aspecto_curva(f, nivel, xc, yc, angulo, r_max, n_puntos)


def razon_aspecto_curva(f, nivel, xc, yc, angulo, r_max, n_puntos=2001):
    """Razón de aspecto de la isolínea f = nivel alrededor de (xc, yc).

    f(x, y) -> valores. Mide la distancia a la que f baja de nivel en las
    direcciones +-angulo (a lo largo) y +-(angulo + 90 grados) (a través), y
    retorna (razón, radio a lo largo, radio a través). Cada radio es el
    promedio de los dos sentidos.
    """
    s = np.linspace(0.0, r_max, n_puntos)

    def radio(ang):
        xp, yp = xc + s * np.cos(ang), yc + s * np.sin(ang)
        v = f(xp, yp)
        abajo = np.flatnonzero(~(v >= nivel))
        if abajo.size == 0 or abajo[0] == 0:
            return np.nan
        k = abajo[0]
        return s[k - 1] + (v[k - 1] - nivel) / (v[k - 1] - v[k]) * (s[k] - s[k - 1])

    r_par = 0.5 * (radio(angulo) + radio(angulo + np.pi))
    r_perp = 0.5 * (radio(angulo + np.pi / 2) + radio(angulo - np.pi / 2))
    return r_par / r_perp, r_par, r_perp


def vector_flujo_frontera(nodes, elements, D_func):
    """Vector g tal que g @ N = flujo saliente -int (D grad n_h) . n_ext sobre la frontera."""
    _, H, baric = assembly.geometria(nodes, elements)
    D = D_func(baric[:, 0], baric[:, 1])

    # Aristas de frontera: las que pertenecen a un solo triángulo
    loc = np.array([[0, 1], [1, 2], [2, 0]])
    aristas = np.sort(elements[:, loc], axis=2).reshape(-1, 2)
    _, inv, cuenta = np.unique(aristas, axis=0, return_inverse=True, return_counts=True)
    frontera = np.flatnonzero(cuenta[inv.ravel()] == 1)
    e, m = frontera // 3, frontera % 3

    # Normal exterior por la longitud de la arista (triángulos antihorarios)
    d = nodes[elements[e, (m + 1) % 3]] - nodes[elements[e, m]]
    normal = np.column_stack((d[:, 1], -d[:, 0]))

    w = -np.einsum("ei,eij,ejm->em", normal, D[e], H[e])        # (aristas, 3)
    return np.bincount(elements[e].ravel(), weights=w.ravel(), minlength=len(nodes))


def momentos_cuadratura(nodes, elements, N):
    """Como momentos(), pero integrando n_h (x - xc)^2 etc. con la cuadratura de 6 puntos.

    Es exacto para la solución P1 (no interpola (x - xc)^2), útil cuando la
    varianza cambia muy poco y se necesita su pendiente con precisión.
    """
    detJ, _, _ = assembly.geometria(nodes, elements)
    xq = np.einsum("qm,emi->eqi", CUAD_BARI, nodes[elements])       # (E, 6, 2)
    pesos = (detJ / 2.0)[:, None] * CUAD_PESOS                       # (E, 6)
    N = np.atleast_2d(np.asarray(N, dtype=float))
    claves = ("masa", "xc", "yc", "xx", "yy", "xy")
    salida = {k: np.empty(len(N)) for k in claves}
    for s, Ns in enumerate(N):
        w = pesos * (Ns[elements] @ CUAD_BARI.T)                     # n_h por peso
        m = w.sum()
        xc, yc = (w * xq[..., 0]).sum() / m, (w * xq[..., 1]).sum() / m
        dx, dy = xq[..., 0] - xc, xq[..., 1] - yc
        valores = (m, xc, yc, (w * dx * dx).sum() / m, (w * dy * dy).sum() / m,
                   (w * dx * dy).sum() / m)
        for k, v in zip(claves, valores):
            salida[k][s] = v
    return salida


def angulo_principal(mom):
    """Ángulo del eje mayor de la nube, en grados."""
    return 0.5 * np.degrees(np.arctan2(2.0 * mom["xy"], mom["xx"] - mom["yy"]))


def minimo(N):
    """Mínimo de n en cada instante."""
    return np.asarray(N).min(axis=-1)
