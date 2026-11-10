"""Mallas triangulares P1.

Cada malla se describe con:
  nodes          (Mtot, 2)  coordenadas de los nodos
  elements       (E, 3)     conectividad g(e, m), vértices en sentido antihorario
  boundary_nodes            índices de los nodos de frontera
  interior_nodes            índices de los nodos interiores
"""

import numpy as np


def malla_estructurada(Lx, Ly, Nx, Ny):
    """Malla estructurada de [0, Lx] x [0, Ly] con Nx x Ny nodos.

    Los nodos se numeran por filas (primero x, luego y) y cada rectángulo
    se divide en dos triángulos antihorarios: [n0, n1, n2] y [n1, n3, n2].
    """
    if Lx <= 0 or Ly <= 0:
        raise ValueError("Las dimensiones deben ser positivas")
    if Nx < 2 or Ny < 2:
        raise ValueError("Se necesitan al menos dos nodos por dirección")

    x = np.linspace(0.0, Lx, Nx)
    y = np.linspace(0.0, Ly, Ny)
    X, Y = np.meshgrid(x, y)
    nodes = np.column_stack((X.ravel(), Y.ravel()))

    # Esquinas de cada rectángulo: n0 abajo-izq, n1 abajo-der, n2 arriba-izq, n3 arriba-der
    j, i = np.meshgrid(np.arange(Ny - 1), np.arange(Nx - 1), indexing="ij")
    n0 = (j * Nx + i).ravel()
    n1 = n0 + 1
    n2 = n0 + Nx
    n3 = n2 + 1

    elements = np.empty((2 * n0.size, 3), dtype=np.int64)
    elements[0::2] = np.column_stack((n0, n1, n2))
    elements[1::2] = np.column_stack((n1, n3, n2))

    # Frontera: primera y última fila y columna de la grilla
    ids = np.arange(Nx * Ny).reshape(Ny, Nx)
    frontera = np.zeros(Nx * Ny, dtype=bool)
    frontera[ids[0, :]] = frontera[ids[-1, :]] = True
    frontera[ids[:, 0]] = frontera[ids[:, -1]] = True

    return nodes, elements, np.flatnonzero(frontera), np.flatnonzero(~frontera)


def orientar_antihorario(nodes, elements, tol=1e-14):
    """Intercambia los vértices 1 y 2 de los elementos con |J| < 0.

    Lanza error si hay triángulos degenerados (|J| ≈ 0).
    Retorna (elements, número de elementos reorientados).
    """
    elements = np.array(elements, dtype=np.int64, copy=True)
    p0, p1, p2 = (nodes[elements[:, m]] for m in range(3))
    detJ = ((p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1])
            - (p2[:, 0] - p0[:, 0]) * (p1[:, 1] - p0[:, 1]))

    if np.any(np.abs(detJ) <= tol * np.max(np.abs(detJ))):
        raise ValueError("Hay triángulos degenerados en la malla")

    neg = detJ < 0
    elements[neg, 1], elements[neg, 2] = elements[neg, 2].copy(), elements[neg, 1].copy()
    return elements, int(np.count_nonzero(neg))
