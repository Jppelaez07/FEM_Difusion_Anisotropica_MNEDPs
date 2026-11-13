"""Mallas triangulares P1.

Todas las funciones de malla devuelven:
  nodes           (Mtot, 2) coordenadas
  elements        (E, 3) conectividad, vértices en sentido antihorario
  boundary_nodes  índices de los nodos de frontera
  interior_nodes  índices de los nodos interiores
"""

import gmsh
import numpy as np


def make_mesh(Lx, Ly, h_global, h_source, xs, ys, r_ref, transicion=None):
    """Malla no estructurada de [0, Lx] x [0, Ly] refinada alrededor de la fuente.

    Tamaño h_source hasta una distancia r_ref de (xs, ys); luego crece
    linealmente hasta h_global en r_ref + transicion (por defecto transicion = r_ref).
    """
    if not 0 < h_source <= h_global:
        raise ValueError("Se requiere 0 < h_source <= h_global")
    if not (0 < xs < Lx and 0 < ys < Ly):
        raise ValueError("La fuente debe estar dentro del dominio")
    if r_ref <= 0:
        raise ValueError("r_ref debe ser positivo")
    if transicion is None:
        transicion = r_ref
    return _mallar_rectangulo(Lx, Ly, h_global, (h_source, xs, ys, r_ref, transicion))


def malla_uniforme(Lx, Ly, h):
    """Malla no estructurada de [0, Lx] x [0, Ly] con tamaño h en todo el dominio."""
    if h <= 0:
        raise ValueError("h debe ser positivo")
    return _mallar_rectangulo(Lx, Ly, h, None)


def malla_caja(Lx, Ly, h_fuera, h_dentro, x0, x1, y0, y1, transicion):
    """Malla no estructurada de [0, Lx] x [0, Ly] con tamaño h_dentro en la caja
    [x0, x1] x [y0, y1] y h_fuera lejos de ella; el tamaño crece linealmente en una
    franja de ancho transicion alrededor de la caja.
    """
    if not 0 < h_dentro <= h_fuera:
        raise ValueError("Se requiere 0 < h_dentro <= h_fuera")
    if not (0 <= x0 < x1 <= Lx and 0 <= y0 < y1 <= Ly):
        raise ValueError("La caja debe estar dentro del dominio")
    return _mallar_rectangulo(Lx, Ly, h_fuera, {"caja": (h_dentro, x0, x1, y0, y1, transicion)})


def _mallar_rectangulo(Lx, Ly, h_global, refinamiento):
    """Malla el rectángulo con gmsh.

    refinamiento: None (tamaño uniforme), (h_source, xs, ys, r_ref, transicion)
    o {"caja": (h_dentro, x0, x1, y0, y1, transicion)}.
    """
    if Lx <= 0 or Ly <= 0:
        raise ValueError("Las dimensiones deben ser positivas")

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("dominio")
        geo = gmsh.model.geo

        # Rectángulo
        esquinas = [geo.addPoint(x, y, 0.0, h_global)
                    for x, y in [(0, 0), (Lx, 0), (Lx, Ly), (0, Ly)]]
        lados = [geo.addLine(esquinas[k], esquinas[(k + 1) % 4]) for k in range(4)]
        contorno = geo.addCurveLoop(lados)
        superficie = geo.addPlaneSurface([contorno])

        if isinstance(refinamiento, tuple):
            h_source, xs, ys, r_ref, transicion = refinamiento
            punto_fuente = geo.addPoint(xs, ys, 0.0, h_source)   # referencia del refinamiento
        geo.synchronize()

        grupo_frontera = gmsh.model.addPhysicalGroup(1, lados, name="frontera")
        gmsh.model.addPhysicalGroup(2, [superficie], name="dominio")

        campo = gmsh.model.mesh.field
        if refinamiento is None:
            tamano = campo.add("MathEval")
            campo.setString(tamano, "F", repr(float(h_global)))
        elif isinstance(refinamiento, dict):
            # Caja con tamaño h_dentro y transición lineal hasta h_global
            h_dentro, x0, x1, y0, y1, transicion = refinamiento["caja"]
            tamano = campo.add("Box")
            campo.setNumber(tamano, "VIn", h_dentro)
            campo.setNumber(tamano, "VOut", h_global)
            for clave, valor in (("XMin", x0), ("XMax", x1), ("YMin", y0), ("YMax", y1)):
                campo.setNumber(tamano, clave, valor)
            campo.setNumber(tamano, "Thickness", transicion)
        else:
            # Campo de distancia + umbral
            distancia = campo.add("Distance")
            campo.setNumbers(distancia, "PointsList", [punto_fuente])
            tamano = campo.add("Threshold")
            campo.setNumber(tamano, "InField", distancia)
            campo.setNumber(tamano, "SizeMin", h_source)
            campo.setNumber(tamano, "SizeMax", h_global)
            campo.setNumber(tamano, "DistMin", r_ref)
            campo.setNumber(tamano, "DistMax", r_ref + transicion)
        campo.setAsBackgroundMesh(tamano)

        # El tamaño lo controla solo el campo
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.Algorithm", 6)     # Frontal-Delaunay
        gmsh.option.setNumber("Mesh.ElementOrder", 1)

        gmsh.model.mesh.generate(2)

        etiquetas, coords, _ = gmsh.model.mesh.getNodes()
        _, nodos_tri = gmsh.model.mesh.getElementsByType(2)   # triángulos de 3 nodos
        etiquetas_frontera, _ = gmsh.model.mesh.getNodesForPhysicalGroup(1, grupo_frontera)
    finally:
        gmsh.finalize()

    # Etiquetas de gmsh -> índices 0..Mtot-1
    etiquetas = np.asarray(etiquetas, dtype=np.int64)
    coords = np.asarray(coords).reshape(-1, 3)[:, :2]
    indice = np.full(etiquetas.max() + 1, -1, dtype=np.int64)
    indice[etiquetas] = np.arange(etiquetas.size)
    elements = indice[np.asarray(nodos_tri, dtype=np.int64).reshape(-1, 3)]
    es_frontera = np.zeros(etiquetas.size, dtype=bool)
    es_frontera[indice[np.asarray(etiquetas_frontera, dtype=np.int64)]] = True

    # Quitar nodos que no están en ningún triángulo y renumerar
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

    Cada rectángulo se divide en dos triángulos.
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

    # Frontera: primera y última fila y columna
    ids = np.arange(Nx * Ny).reshape(Ny, Nx)
    frontera = np.zeros(Nx * Ny, dtype=bool)
    frontera[ids[0, :]] = frontera[ids[-1, :]] = True
    frontera[ids[:, 0]] = frontera[ids[:, -1]] = True

    return nodes, elements, np.flatnonzero(frontera), np.flatnonzero(~frontera)


def determinantes(nodes, elements):
    """|J| de cada elemento (positivo si es antihorario)."""
    p0, p1, p2 = (nodes[elements[:, m]] for m in range(3))
    return ((p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1])
            - (p2[:, 0] - p0[:, 0]) * (p1[:, 1] - p0[:, 1]))


def orientar_antihorario(nodes, elements, tol=1e-14):
    """Reorienta los elementos con |J| < 0. Retorna (elements, número reorientado)."""
    elements = np.array(elements, dtype=np.int64, copy=True)
    detJ = determinantes(nodes, elements)

    if np.any(np.abs(detJ) <= tol * np.max(np.abs(detJ))):
        raise ValueError("Hay triángulos degenerados en la malla")

    neg = detJ < 0
    elements[neg, 1], elements[neg, 2] = elements[neg, 2].copy(), elements[neg, 1].copy()
    return elements, int(np.count_nonzero(neg))


def calidad_malla(nodes, elements):
    """Número de nodos y elementos, h mínimo, medio y máximo, y ángulo mínimo (grados).

    h de un elemento = su lado más largo.
    """
    p = nodes[elements]                     # (E, 3, 2)
    a = p[:, [1, 2, 0]] - p                 # lado del vértice m al m+1
    b = p[:, [2, 0, 1]] - p                 # lado del vértice m al m+2
    h = np.linalg.norm(a, axis=2).max(axis=1)

    cos = np.einsum("emi,emi->em", a, b) / (np.linalg.norm(a, axis=2) * np.linalg.norm(b, axis=2))
    angulos = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))

    return {
        "n_nodos": len(nodes),
        "n_elementos": len(elements),
        "h_min": float(h.min()),
        "h_medio": float(h.mean()),
        "h_max": float(h.max()),
        "angulo_min": float(angulos.min()),
    }
