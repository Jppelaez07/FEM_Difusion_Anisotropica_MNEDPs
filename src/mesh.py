"""Mallas triangulares P1.

Cada malla se describe con:
  nodes          (Mtot, 2)  coordenadas de los nodos
  elements       (E, 3)     conectividad g(e, m), vértices en sentido antihorario
  boundary_nodes            índices de los nodos de frontera
  interior_nodes            índices de los nodos interiores
"""

import gmsh
import numpy as np


def make_mesh(Lx, Ly, h_global, h_source, xs, ys, r_ref, transicion=None):
    """Malla no estructurada de [0, Lx] x [0, Ly] con refinamiento en la fuente.

    El tamaño de elemento es h_source a distancia <= r_ref de (xs, ys) y crece
    linealmente hasta h_global a distancia r_ref + transicion (por defecto
    transicion = r_ref). Se usa un Field Distance + Threshold de gmsh.
    """
    if Lx <= 0 or Ly <= 0:
        raise ValueError("Las dimensiones deben ser positivas")
    if not 0 < h_source <= h_global:
        raise ValueError("Se requiere 0 < h_source <= h_global")
    if not (0 < xs < Lx and 0 < ys < Ly):
        raise ValueError("La fuente debe estar dentro del dominio")
    if r_ref <= 0:
        raise ValueError("r_ref debe ser positivo")
    if transicion is None:
        transicion = r_ref

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("dominio")
        geo = gmsh.model.geo

        # Rectángulo: esquinas en sentido antihorario
        esquinas = [geo.addPoint(x, y, 0.0, h_global)
                    for x, y in [(0, 0), (Lx, 0), (Lx, Ly), (0, Ly)]]
        lados = [geo.addLine(esquinas[k], esquinas[(k + 1) % 4]) for k in range(4)]
        contorno = geo.addCurveLoop(lados)
        superficie = geo.addPlaneSurface([contorno])

        # Punto de la fuente: solo sirve de referencia para el campo de distancia
        punto_fuente = geo.addPoint(xs, ys, 0.0, h_source)
        geo.synchronize()

        grupo_frontera = gmsh.model.addPhysicalGroup(1, lados, name="frontera")
        gmsh.model.addPhysicalGroup(2, [superficie], name="dominio")

        # Tamaño de elemento: h_source cerca de la fuente, h_global lejos
        distancia = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(distancia, "PointsList", [punto_fuente])
        umbral = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(umbral, "InField", distancia)
        gmsh.model.mesh.field.setNumber(umbral, "SizeMin", h_source)
        gmsh.model.mesh.field.setNumber(umbral, "SizeMax", h_global)
        gmsh.model.mesh.field.setNumber(umbral, "DistMin", r_ref)
        gmsh.model.mesh.field.setNumber(umbral, "DistMax", r_ref + transicion)
        gmsh.model.mesh.field.setAsBackgroundMesh(umbral)

        # Que el tamaño lo controle solo el campo
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.Algorithm", 6)     # Frontal-Delaunay
        gmsh.option.setNumber("Mesh.ElementOrder", 1)  # P1

        gmsh.model.mesh.generate(2)

        etiquetas, coords, _ = gmsh.model.mesh.getNodes()
        _, nodos_tri = gmsh.model.mesh.getElementsByType(2)   # tipo 2 = triángulo de 3 nodos
        etiquetas_frontera, _ = gmsh.model.mesh.getNodesForPhysicalGroup(1, grupo_frontera)
    finally:
        gmsh.finalize()

    # Pasar de etiquetas de gmsh a índices 0..Mtot-1
    etiquetas = np.asarray(etiquetas, dtype=np.int64)
    coords = np.asarray(coords).reshape(-1, 3)[:, :2]
    indice = np.full(etiquetas.max() + 1, -1, dtype=np.int64)
    indice[etiquetas] = np.arange(etiquetas.size)
    elements = indice[np.asarray(nodos_tri, dtype=np.int64).reshape(-1, 3)]
    es_frontera = np.zeros(etiquetas.size, dtype=bool)
    es_frontera[indice[np.asarray(etiquetas_frontera, dtype=np.int64)]] = True

    # Eliminar nodos que no pertenecen a ningún triángulo (p. ej. el de la fuente)
    usados = np.unique(elements)
    nuevo = np.full(etiquetas.size, -1, dtype=np.int64)
    nuevo[usados] = np.arange(usados.size)
    nodes = coords[usados]
    elements = nuevo[elements]
    es_frontera = es_frontera[usados]

    elements, _ = orientar_antihorario(nodes, elements)
    assert np.all(determinantes(nodes, elements) > 0), "Hay elementos con |J| <= 0"

    return nodes, elements, np.flatnonzero(es_frontera), np.flatnonzero(~es_frontera)


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


def determinantes(nodes, elements):
    """|J| de cada elemento: positivo si sus vértices están en sentido antihorario."""
    p0, p1, p2 = (nodes[elements[:, m]] for m in range(3))
    return ((p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1])
            - (p2[:, 0] - p0[:, 0]) * (p1[:, 1] - p0[:, 1]))


def orientar_antihorario(nodes, elements, tol=1e-14):
    """Intercambia los vértices 1 y 2 de los elementos con |J| < 0.

    Lanza error si hay triángulos degenerados (|J| ≈ 0).
    Retorna (elements, número de elementos reorientados).
    """
    elements = np.array(elements, dtype=np.int64, copy=True)
    detJ = determinantes(nodes, elements)

    if np.any(np.abs(detJ) <= tol * np.max(np.abs(detJ))):
        raise ValueError("Hay triángulos degenerados en la malla")

    neg = detJ < 0
    elements[neg, 1], elements[neg, 2] = elements[neg, 2].copy(), elements[neg, 1].copy()
    return elements, int(np.count_nonzero(neg))


def calidad_malla(nodes, elements):
    """Indicadores de calidad de la malla.

    h de un elemento = longitud de su lado más largo.
    Retorna un diccionario con n_nodos, n_elementos, h_min, h_max y
    angulo_min (en grados, mínimo sobre todos los ángulos de todos los triángulos).
    """
    p = nodes[elements]                                   # (E, 3, 2)
    a = p[:, [1, 2, 0]] - p                               # lado del vértice m al m+1
    b = p[:, [2, 0, 1]] - p                               # lado del vértice m al m+2
    h = np.linalg.norm(a, axis=2).max(axis=1)

    # Ángulo en cada vértice m, entre los dos lados que salen de él
    cos = np.einsum("emi,emi->em", a, b) / (np.linalg.norm(a, axis=2) * np.linalg.norm(b, axis=2))
    angulos = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))

    return {
        "n_nodos": len(nodes),
        "n_elementos": len(elements),
        "h_min": float(h.min()),
        "h_max": float(h.max()),
        "angulo_min": float(angulos.min()),
    }
