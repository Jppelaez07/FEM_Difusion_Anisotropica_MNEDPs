"""Simulaciones adicionales en el contexto del remanente de supernova (Nava y Gabici, 2013).

Cada simulación es un diccionario de SIMULACIONES, identificado por un nombre corto; para
agregar otra basta con añadir un diccionario. Todas usan n0 gaussiana y Q = 0. Se trabaja
en pc y kyr.

- Sin la clave "campo", B es uniforme (ángulo "alpha") y la simulación se verifica contra la
  gaussiana exacta en el plano infinito (fields.solucion_gaussiana).
- Con la clave "campo" (una función B(x, y) de fields.py), B puede ser no uniforme; la
  gaussiana exacta no aplica y se reportan masa, extremos, momentos, puntos de prueba y la
  difusión perpendicular efectiva. "mallas" da una o varias mallas (estructuradas o de gmsh;
  la marcada como referencia se usa en las figuras) y "control" una corrida de control con
  otras claves (p. ej. B uniforme), que sí se compara con la exacta.

sim1: difusión isotrópica intensa, D_par = D_perp = 1e29 cm^2/s, en [-350, 350]^2 pc.
sim2: difusión restringida en el campo curvo B = (1, x / 50), D_par = 1e28 y
      D_perp = 1e26 cm^2/s, en [-350, 350] x [-150, 150] pc, con control de B uniforme.
      Malla de referencia: gmsh con h = 1.25 pc en |x| <= 150, |y| <= 100 pc y 5 pc fuera;
      dos mallas estructuradas como comparación.

Guarda results/15_simulaciones.txt, results/15_simulaciones.csv y figuras results/15_<nombre>_*.
"""

import csv
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from src import assembly, fields, mesh, postproc, rutas, timestep, viz  # noqa: E402

# --- Unidades ---
PC_CM = 3.0857e18
KYR_S = 3.156e10


def a_pc2_kyr(D_cgs):
    """Convierte un coeficiente de difusión de cm^2/s a pc^2/kyr."""
    return D_cgs * KYR_S / PC_CM ** 2


# Nubes de prueba de 14_snr.py: A a lo largo de B, B perpendicular, ambas a 50 pc.
NUBES = {"A": (50.0, 0.0), "B": (0.0, 50.0)}


# --- Etiquetas de líneas de campo ---
# c(x, y) es constante sobre cada línea de campo (grad c perpendicular a b). Entonces
# D grad c = D_perp grad c y, lejos de la frontera, d/dt int n c^2 = D_perp int n G,
# con G = 2 |grad c|^2 + 2 c lap c. Comparar las dos cosas mide la difusión perpendicular
# efectiva, incluida la numérica (los detalles están en el informe).

def linea_parabolica(L_B):
    """Líneas de B = (1, x / L_B): c = y - x^2 / (2 L_B)."""
    return {"c": lambda x, y: y - x ** 2 / (2 * L_B),
            "G": lambda x, y: 2 * (1 + x ** 2 / L_B ** 2) - 2 * (y - x ** 2 / (2 * L_B)) / L_B,
            "texto": f"c = y - x^2 / {2 * L_B:g}"}


def linea_horizontal():
    """Líneas de B uniforme horizontal: c = y."""
    return {"c": lambda x, y: y, "G": lambda x, y: 2 + 0 * x, "texto": "c = y"}


# --- Simulaciones ---
L_B2 = 50.0
SIMULACIONES = [
    {
        "nombre": "sim1",
        "titulo": "Difusión isotrópica intensa, B uniforme",
        "D_par_cgs": 1e29, "D_perp_cgs": 1e29,      # cm^2/s
        "alpha": 0.0,                               # ángulo de B (radianes); horizontal
        "L": (350.0, 350.0),                        # dominio [-Lx, Lx] x [-Ly, Ly] pc
        "nodos": (281, 281),                        # malla estructurada, h = 2.5 pc
        "sigma0": 10.0, "n_max": 1.0,               # n0 = n_max exp(-r^2 / (2 sigma0^2))
        "theta": 0.5, "dt": 0.02, "T": 10.0,        # kyr
        "t_metricas": [0.0, 1.0, 2.0, 5.0, 10.0],
        "t_mapas": [0.0, 2.0, 5.0, 10.0],
        "comparar_14": True,
        "notas": [
            "B uniforme horizontal; con D isotrópico su dirección no influye.",
            "B0 = 3 microG no entra en el modelo: solo importa la dirección de B a través de D.",
            "Dominio cuadrado: con difusión isotrópica la nube llega a sigma ~ 82 pc en las dos",
            "direcciones en T = 10 kyr; con y = +-150 pc la frontera absorbería una parte importante.",
        ],
    },
    {
        "nombre": "sim2",
        "titulo": "Difusión restringida en un campo curvo",
        "D_par_cgs": 1e28, "D_perp_cgs": 1e26,
        "campo": fields.campo_parabolico(L_B2),
        "campo_texto": f"B = B0 (1, x / {L_B2:g}): líneas y = x^2 / {2 * L_B2:g} + c, radio de curvatura"
                       f" {L_B2:g} pc en el vértice",
        "L_B": L_B2,
        "linea": linea_parabolica(L_B2),
        "L": (350.0, 150.0),
        "mallas": [
            {"nombre": "281x121", "tipo": "estructurada", "nodos": (281, 121)},
            {"nombre": "561x241", "tipo": "estructurada", "nodos": (561, 241)},
            {"nombre": "gmsh", "tipo": "gmsh_caja", "h_dentro": 1.25, "h_fuera": 5.0,
             "ventana": (150.0, 100.0), "transicion": 20.0, "referencia": True},
        ],
        "sigma0": 10.0, "n_max": 1.0,
        "theta": 0.5, "dt": 0.02, "T": 10.0,
        "t_metricas": [0.0, 1.0, 2.0, 5.0, 10.0],
        "t_mapas": [0.0, 2.0, 5.0, 10.0],
        "puntos": {"A": (50.0, 25.0), "A'": (50.0, 0.0), "B": (0.0, 50.0)},
        "control": {"campo": fields.campo_uniforme(0.0), "alpha": 0.0, "linea": linea_horizontal(),
                    "mallas": [{"nombre": "281x121", "tipo": "estructurada", "nodos": (281, 121)}],
                    "exacta": True,
                    "campo_texto": "B uniforme horizontal"},
        "notas": [
            "B0 = 10 microG no entra en el modelo: solo importa la dirección de B. B no se anula.",
            "A = (50, 25) está sobre la parábola que pasa por el origen; A' = (50, 0) sobre y = 0, que ya",
            "no es línea de campo; B = (0, 50) en la dirección perpendicular.",
        ],
    },
]
if rutas.RAPIDO:
    for s in SIMULACIONES:
        if "mallas" in s:
            s["mallas"] = [m for m in s["mallas"] if m["nombre"] != "561x241"]

RESULTADOS = rutas.RESULTADOS


def tensor(sim):
    """Tensor D constante (pc^2/kyr) de una simulación con B uniforme."""
    Bx, By = np.cos(sim["alpha"]), np.sin(sim["alpha"])
    return np.asarray(fields.tensor_difusion(np.array([Bx]), np.array([By]), sim["D_par"],
                                             sim["D_perp"]))[0]


def matriz_S(sim, t):
    """Matriz de covarianza exacta S(t) = sigma0^2 I + 2 D t."""
    return sim["sigma0"] ** 2 * np.eye(2) + 2.0 * tensor(sim) * t


def aspecto_de_S(S):
    """sqrt(l1 / l2) de una matriz 2x2 simétrica."""
    l = np.linalg.eigvalsh(S)
    return np.sqrt(l[1] / l[0])


def construir_malla(sim, malla):
    """Malla centrada en el origen. malla: (Nx, Ny) estructurada o dict de una malla de sim2.

    Retorna (nodes, elements, bnd, h, h_borde): h es el tamaño donde está la nube y h_borde
    el tamaño junto a la frontera.
    """
    Lx, Ly = sim["L"]
    if isinstance(malla, tuple) or malla["tipo"] == "estructurada":
        nodos = malla if isinstance(malla, tuple) else malla["nodos"]
        nodes, elements, bnd, _ = mesh.malla_estructurada(2 * Lx, 2 * Ly, *nodos)
        h = h_borde = 2 * Lx / (nodos[0] - 1)
    else:
        vx, vy = malla["ventana"]
        nodes, elements, bnd, _ = mesh.malla_caja(2 * Lx, 2 * Ly, malla["h_fuera"], malla["h_dentro"],
                                                  Lx - vx, Lx + vx, Ly - vy, Ly + vy, malla["transicion"])
        h, h_borde = malla["h_dentro"], malla["h_fuera"]
    return nodes - np.array([Lx, Ly]), elements, bnd, h, h_borde


def tamano_malla(sim, malla):
    """Tamaño de la malla donde está la nube, sin construirla."""
    if malla["tipo"] == "estructurada":
        return 2 * sim["L"][0] / (malla["nodos"][0] - 1)
    return malla["h_dentro"]


def correr(sim, nodos, campo, puntos, linea=None):
    """Malla, ensamblaje y avance en el tiempo.

    Registra en cada paso el pico, el mínimo y n en los puntos; si se da la etiqueta de
    líneas de campo, también las integrales de n por 1, x, y, x^2, y^2, xy, c, c^2 y G.
    """
    nodes, elements, bnd, h, h_borde = construir_malla(sim, nodos)
    D_func = fields.funcion_tensor(campo, sim["D_par"], sim["D_perp"])
    n0 = fields.condicion_inicial_gaussiana(sim["n_max"], 0.0, 0.0, sim["sigma0"])
    P = {k: viz.matriz_rejilla(nodes, elements, np.array([p[0]]), np.array([p[1]]))
         for k, p in puntos.items()}
    reg = {"t": [], "pico": [], "minimo": []}
    reg_p = {k: [] for k in puntos}
    integrales = []

    if linea is not None:
        detJ, _, _ = assembly.geometria(nodes, elements)
        xq = np.einsum("qm,emi->eqi", postproc.CUAD_BARI, nodes[elements])
        pesos = (detJ / 2.0)[:, None] * postproc.CUAD_PESOS
        X, Y = xq[..., 0], xq[..., 1]
        c = linea["c"](X, Y)
        Fq = np.stack([np.ones_like(X), X, Y, X * X, Y * Y, X * Y, c, c * c, linea["G"](X, Y)], axis=-1)
        Fq = (pesos[..., None] * Fq).reshape(-1, Fq.shape[-1])

    def al_paso(k, t, N):
        reg["t"].append(t)
        reg["pico"].append(N.max())
        reg["minimo"].append(N.min())
        for nombre in puntos:
            reg_p[nombre].append((P[nombre] @ N)[0])
        if linea is not None:
            nq = (N[elements] @ postproc.CUAD_BARI.T).ravel()
            integrales.append(nq @ Fq)

    res = timestep.simular(nodes, elements, bnd, D_func, n0, sim["T"], sim["dt"],
                           theta=sim["theta"], Q=None, bc="dirichlet",
                           t_guardar=sorted(set(sim["t_metricas"]) | set(sim["t_mapas"])),
                           al_paso=al_paso)
    r = {k: np.array(v) for k, v in reg.items()}
    r["n"] = {k: np.array(v) for k, v in reg_p.items()}
    if "A" in puntos and "B" in puntos:
        r["nA"], r["nB"] = r["n"]["A"], r["n"]["B"]
    if linea is not None:
        r["I"] = np.array(integrales)
    r["estados"] = dict(zip(np.round(res["t"], 6), res["N"]))
    r.update(nodes=nodes, elements=elements, bnd=bnd, M=res["M"], h=h, h_borde=h_borde)
    return r


# =====================================================================================
# Simulaciones con B uniforme (sim1): verificación contra la gaussiana exacta
# =====================================================================================

def metricas(sim, r):
    """Métricas de verificación en cada t de t_metricas."""
    nodes, elements = r["nodes"], r["elements"]
    n_ex = fields.solucion_gaussiana(sim["n_max"], 0.0, 0.0, sim["sigma0"], tensor(sim))
    masa_exacta = 2.0 * np.pi * sim["sigma0"] ** 2 * sim["n_max"]
    filas = []
    for t in sim["t_metricas"]:
        N = r["estados"][round(t, 6)]
        k = int(round(t / sim["dt"]))
        exacta = lambda x, y, t=t: n_ex(x, y, t)
        mom = postproc.momentos_cuadratura(nodes, elements, N)
        S = matriz_S(sim, t)
        filas.append({
            "simulacion": sim["nombre"], "t_kyr": t,
            "error_L1_rel": postproc.error_L1_relativo(nodes, elements, N, exacta),
            "error_L2_rel": postproc.error_L2_relativo(nodes, elements, N, exacta),
            "masa": float(postproc.masa(r["M"], N)), "masa_exacta": masa_exacta,
            "x2": mom["xx"][0], "x2_teo": S[0, 0], "y2": mom["yy"][0], "y2_teo": S[1, 1],
            "aspecto": float(postproc.razon_aspecto(mom)[0]), "aspecto_teo": aspecto_de_S(S),
            "pico": r["pico"][k], "pico_exacto": float(n_ex(0.0, 0.0, t)),
            "minimo": r["minimo"][k], "minimo_hasta_t": r["minimo"][:k + 1].min(),
            "nA": r["nA"][k], "nA_exacta": float(n_ex(*NUBES["A"], t)),
            "nB": r["nB"][k], "nB_exacta": float(n_ex(*NUBES["B"], t)),
        })
    return filas, n_ex


def leer_caso_14():
    """Fila eps = 0.01 de 14_snr_barrido.csv (o None si no existe)."""
    ruta = RESULTADOS / "14_snr_barrido.csv"
    if not ruta.exists():
        return None
    with open(ruta, encoding="utf-8") as fh:
        for fila in csv.DictReader(fh):
            if np.isclose(float(fila["eps"]), 0.01):
                return {k: float(v) for k, v in fila.items()}
    return None


def informe(sim, r, filas, n_ex, reportar, segundos):
    """Sección del .txt de una simulación con B uniforme: parámetros, verificación y comparación."""
    Lx, Ly = sim["L"]
    nx, ny = sim["nodos"]
    h = 2 * Lx / (nx - 1)
    T = sim["T"]
    reportar("=" * 100)
    reportar(f"SIMULACIÓN {sim['nombre']}: {sim['titulo']}")
    reportar("=" * 100)
    reportar("PARÁMETROS")
    reportar(f"  1 pc = {PC_CM:.4e} cm;  1 kyr = {KYR_S:.4e} s")
    reportar(f"  D_par  = {sim['D_par_cgs']:.0e} cm^2/s = {sim['D_par']:.2f} pc^2/kyr")
    reportar(f"  D_perp = {sim['D_perp_cgs']:.0e} cm^2/s = {sim['D_perp']:.2f} pc^2/kyr"
             f"  (D_perp / D_par = {sim['D_perp'] / sim['D_par']:g})")
    reportar(f"  B uniforme con ángulo {np.degrees(sim['alpha']):g} grados respecto al eje x")
    reportar(f"  Dominio [-{Lx:g}, {Lx:g}] x [-{Ly:g}, {Ly:g}] pc; malla estructurada {nx}x{ny}"
             f" ({len(r['nodes'])} nodos, h = {h:g} pc)")
    reportar(f"  n0 = {sim['n_max']:g} exp(-(x^2 + y^2) / (2 {sim['sigma0']:g}^2)) centrada en (0, 0); Q = 0;"
             " Dirichlet n = 0")
    reportar(f"  theta = {sim['theta']:g}, dt = {sim['dt']:g} kyr, T = {T:g} kyr"
             f" ({int(round(T / sim['dt']))} pasos);  D_par dt / h^2 = {sim['D_par'] * sim['dt'] / h ** 2:.2f}")
    for nota in sim["notas"]:
        reportar("  " + nota)
    reportar(f"  Tiempo de cómputo: {segundos:.0f} s")
    reportar()

    reportar("VERIFICACIÓN CONTRA LA GAUSSIANA EXACTA EN EL PLANO INFINITO")
    reportar(f"  {'t (kyr)':>7s} {'L1 rel.':>9s} {'L2 rel.':>9s} {'masa/exacta':>11s} {'<x^2>':>9s} {'teo':>9s}"
             f" {'<y^2>':>9s} {'teo':>9s} {'aspecto':>8s} {'pico':>10s} {'exacto':>10s} {'min n':>10s}")
    for f in filas:
        reportar(f"  {f['t_kyr']:7g} {f['error_L1_rel']:9.2e} {f['error_L2_rel']:9.2e}"
                 f" {f['masa'] / f['masa_exacta']:11.6f} {f['x2']:9.2f} {f['x2_teo']:9.2f}"
                 f" {f['y2']:9.2f} {f['y2_teo']:9.2f} {f['aspecto']:8.4f} {f['pico']:10.4e}"
                 f" {f['pico_exacto']:10.4e} {f['minimo']:10.2e}")
    reportar(f"  masa exacta = 2 pi sigma0^2 n_max = {filas[0]['masa_exacta']:.3f} pc^2 (constante en el plano"
             " infinito); <x^2>, <y^2> en pc^2, teórico sigma0^2 + 2 D t.")
    reportar(f"  aspecto = sqrt(l1 / l2) de la matriz de momentos; teórico en t = {T:g}: {filas[-1]['aspecto_teo']:.4f}.")
    exceso0 = filas[0]["x2"] - filas[0]["x2_teo"]
    reportar(f"  En t = 0, <x^2> excede al teórico en {exceso0:.2f} pc^2 (h^2 / 6 = {h ** 2 / 6:.2f}): es el error de")
    reportar("  interpolar n0 en P1, y se mantiene mientras la nube no toca la frontera. En t = T la")
    reportar(f"  diferencia es {filas[-1]['x2'] - filas[-1]['x2_teo']:+.2f} pc^2: la frontera absorbente recorta la cola.")
    reportar()

    bnd = r["bnd"]
    borde_T = n_ex(r["nodes"][bnd, 0], r["nodes"][bnd, 1], T).max()
    min_global = r["minimo"].min()
    k_min = int(np.argmin(r["minimo"]))
    reportar("FRONTERA Y POSITIVIDAD")
    reportar(f"  Valor exacto máximo de n en la frontera en t = {T:g} kyr: {borde_T:.2e}"
             f" (pico exacto {filas[-1]['pico_exacto']:.2e}, cociente {borde_T / filas[-1]['pico_exacto']:.1e}).")
    reportar(f"  Masa perdida hasta t = {T:g} kyr: {1 - filas[-1]['masa'] / filas[0]['masa']:.2e} de la masa inicial.")
    if min_global < -1e-12 * filas[0]["pico"]:
        reportar(f"  Mínimo de n en toda la corrida: {min_global:.3e} en t = {r['t'][k_min]:g} kyr"
                 f" ({abs(min_global) / filas[0]['pico']:.1e} del pico inicial).")
    elif min_global < 0:
        reportar(f"  Mínimo de n en toda la corrida: {min_global:.1e} en t = {r['t'][k_min]:g} kyr, del orden del"
                 " redondeo: no hay valores negativos apreciables.")
    else:
        reportar("  Mínimo de n en toda la corrida: 0 (solo en la frontera Dirichlet); no hay valores negativos.")
    reportar()

    reportar(f"DENSIDAD EN LAS NUBES DE PRUEBA (A en {NUBES['A']} pc, B en {NUBES['B']} pc)")
    reportar(f"  {'t (kyr)':>7s} {'n_A':>11s} {'exacta':>11s} {'n_B':>11s} {'exacta':>11s} {'n_A/n_B':>9s}")
    for f in filas:
        cociente = f["nA"] / f["nB"] if f["nB"] > 0 else np.nan
        reportar(f"  {f['t_kyr']:7g} {f['nA']:11.4e} {f['nA_exacta']:11.4e} {f['nB']:11.4e}"
                 f" {f['nB_exacta']:11.4e} {cociente:9.4f}")
    reportar()

    if sim.get("comparar_14"):
        comparar_con_14(sim, filas[-1], reportar)


def comparar_con_14(sim, fT, reportar):
    """Tabla de comparación en t = T con el caso anisotrópico (eps = 0.01) de 14_snr.py."""
    T = sim["T"]
    reportar(f"COMPARACIÓN CON EL CASO ANISOTRÓPICO DE 14_snr.py (eps = 0.01, mismo D_par), t = {T:g} kyr")
    c14 = leer_caso_14()
    # Gaussiana anisotrópica exacta con el mismo sigma0 de esta simulación (separa el efecto de n0).
    aniso = dict(sim, D_perp=0.01 * sim["D_par"])
    ex_an = fields.solucion_gaussiana(sim["n_max"], 0.0, 0.0, sim["sigma0"], tensor(aniso))
    filas = [
        (f"{sim['nombre']} isotrópica, numérica", fT["pico"], fT["nA"], fT["nB"], fT["aspecto"]),
        (f"{sim['nombre']} isotrópica, exacta", fT["pico_exacto"], fT["nA_exacta"], fT["nB_exacta"],
         fT["aspecto_teo"]),
    ]
    if c14 is not None:
        filas += [("14_snr eps = 0.01, numérica", c14["pico"], c14["nA"], c14["nB"], c14["aspecto"]),
                  ("14_snr eps = 0.01, exacta", c14["pico_exacto"], c14["nA_exacta"], c14["nB_exacta"],
                   c14["aspecto_teo"])]
    filas.append((f"eps = 0.01 exacta, sigma0 = {sim['sigma0']:g}", float(ex_an(0, 0, T)),
                  float(ex_an(*NUBES["A"], T)), float(ex_an(*NUBES["B"], T)),
                  aspecto_de_S(matriz_S(aniso, T))))
    reportar(f"  {'caso':<34s} {'pico':>10s} {'n_A':>11s} {'n_B':>11s} {'n_A/n_B':>10s} {'aspecto':>8s}")
    for nombre, pico, nA, nB, asp in filas:
        reportar(f"  {nombre:<34s} {pico:10.4e} {nA:11.4e} {nB:11.4e} {nA / nB:10.3e} {asp:8.3f}")
    if c14 is None:
        reportar(f"  No se encontró {RESULTADOS.name}/14_snr_barrido.csv: corra antes scripts/14_snr.py.")
    else:
        reportar("  14_snr.py usa sigma0 = 5 pc, theta = 1 y el dominio [-350, 350] x [-150, 150] pc; la")
        reportar(f"  última fila es la exacta anisotrópica con el mismo n0 de esta simulación (sigma0 = {sim['sigma0']:g} pc).")
    pico_an, nA_an, nB_an = float(ex_an(0, 0, T)), float(ex_an(*NUBES["A"], T)), float(ex_an(*NUBES["B"], T))
    reportar(f"  Con el mismo D_par, la difusión isotrópica reparte los rayos cósmicos en las dos direcciones:")
    reportar(f"  el pico es {pico_an / fT['pico_exacto']:.1f} veces menor que con eps = 0.01, la nube A recibe"
             f" {nA_an / fT['nA_exacta']:.1f} veces menos")
    reportar(f"  y la nube B {fT['nB_exacta'] / nB_an:.1e} veces más (exactas, mismo sigma0). n_A/n_B pasa de"
             f" {nA_an / nB_an:.1e} a {fT['nA'] / fT['nB']:.3f}")
    reportar(f"  y la razón de aspecto de {aspecto_de_S(matriz_S(aniso, T)):.2f} a {fT['aspecto']:.3f}.")
    reportar()


def figuras(sim, r, n_ex):
    nodes, elements = r["nodes"], r["elements"]
    x, y = nodes[:, 0], nodes[:, 1]
    Lx, Ly = sim["L"]
    T = sim["T"]
    base = RESULTADOS / f"15_{sim['nombre']}"

    # Mapas (misma escala logarítmica que 14_snr.py).
    norma = LogNorm(vmin=1e-5, vmax=1.0)
    niveles = np.logspace(-5, 0, 31)
    fig, axes = plt.subplots(2, 2, figsize=(11, 10), constrained_layout=True)
    for ax, t in zip(axes.ravel(), sim["t_mapas"]):
        N = r["estados"][round(t, 6)]
        cf = ax.tricontourf(x, y, elements, np.clip(N, 1e-5, None), levels=niveles, norm=norma,
                            cmap="viridis", extend="min")
        for nombre, (px, py) in NUBES.items():
            ax.plot(px, py, "o", markersize=7, markerfacecolor="none", markeredgecolor="#f97316", mew=2)
            ax.annotate(nombre, (px, py), textcoords="offset points", xytext=(6, 6), color="#f97316",
                        fontsize=11, fontweight="bold")
        ax.set_aspect("equal")
        ax.set_xlabel("x (pc)")
        ax.set_ylabel("y (pc)")
        ax.set_title(f"t = {t:g} kyr", fontsize=11)
        ax.grid(False)
    cb = fig.colorbar(cf, ax=axes.ravel().tolist(), shrink=0.8, label="n (escala logarítmica, adimensional)")
    cb.set_ticks([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1])
    fig.suptitle(f"{sim['nombre']}: {sim['titulo']}\n"
                 rf"$D_\parallel$ = $D_\perp$ = {sim['D_par']:.1f} pc$^2$/kyr, theta = {sim['theta']:g}")
    viz.guardar(fig, f"{base}_mapas")

    # Cortes en t = T: numérica frente a exacta.
    N = r["estados"][round(T, 6)]
    s_x, s_y = np.linspace(-Lx, Lx, 1401), np.linspace(-Ly, Ly, 1401)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), constrained_layout=True)
    for ax, s, eje in ((axes[0], s_x, "x"), (axes[1], s_y, "y")):
        px, py = (s, 0 * s) if eje == "x" else (0 * s, s)
        ax.plot(s, postproc.evaluar_en_puntos(nodes, elements, N, px, py), color="#0072B2",
                linewidth=2, label="numérica")
        ax.plot(s, n_ex(px, py, T), color="#111827", linestyle="--", linewidth=1.2, label="exacta")
        ax.set_xlabel(f"{eje} (pc), corte por {'y' if eje == 'x' else 'x'} = 0")
        ax.set_ylabel("n (adimensional)")
    axes[0].legend(fontsize=9)
    fig.suptitle(f"{sim['nombre']}: cortes en t = {T:g} kyr, numérica frente a exacta")
    viz.guardar(fig, f"{base}_cortes")
    return [f"15_{sim['nombre']}_mapas", f"15_{sim['nombre']}_cortes"]


def procesar_uniforme(sim, reportar):
    t1 = time.time()
    r = correr(sim, sim["nodos"], fields.campo_uniforme(sim["alpha"]), NUBES)
    segundos = time.time() - t1
    filas, n_ex = metricas(sim, r)
    informe(sim, r, filas, n_ex, reportar, segundos)
    return filas, figuras(sim, r, n_ex)


# =====================================================================================
# Simulaciones con B general (sim2): sin solución exacta, con mallas y control
# =====================================================================================

def metricas_campo(sim, r, corrida, malla):
    """Métricas en cada t de t_metricas para una corrida con B general."""
    nodes, elements = r["nodes"], r["elements"]
    h = r["h"]
    I, dt = r["I"], sim["dt"]
    tt = r["t"]
    m0 = I[0, 0]
    x2o_0, yc_0 = I[0, 3] / m0, I[0, 2] / m0
    exacta = sim.get("exacta", False)
    if exacta:
        n_ex = fields.solucion_gaussiana(sim["n_max"], 0.0, 0.0, sim["sigma0"], tensor(sim))
    filas = []
    for t in sim["t_metricas"]:
        k = int(round(t / dt))
        N = r["estados"][round(t, 6)]
        m = I[k, 0]
        xc, yc = I[k, 1] / m, I[k, 2] / m
        var_x, var_y = I[k, 3] / m - xc ** 2, I[k, 4] / m - yc ** 2
        cov = I[k, 5] / m - xc * yc
        C = np.array([[var_x, cov], [cov, var_y]])
        c_media = I[k, 6] / m
        integral_G = np.trapezoid(I[:k + 1, 8], tt[:k + 1]) if k > 0 else np.nan
        i_max, i_min = int(np.argmax(N)), int(np.argmin(N))
        f = {
            "simulacion": sim["nombre"], "corrida": corrida, "malla": malla["nombre"],
            "h_pc": h, "D_par_pc2_kyr": sim["D_par"], "D_perp_pc2_kyr": sim["D_perp"], "t_kyr": t,
            "masa": m, "masa_rel": m / m0, "minimo": r["minimo"][k], "minimo_hasta_t": r["minimo"][:k + 1].min(),
            "pico": N.max(), "x_pico": nodes[i_max, 0], "y_pico": nodes[i_max, 1],
            "x_min": nodes[i_min, 0], "y_min": nodes[i_min, 1],
            "xc": xc, "yc": yc, "x2": var_x, "y2": var_y, "xy": cov, "aspecto": aspecto_de_S(C),
            "x2_origen": I[k, 3] / m,
            "c_media": c_media, "var_c": I[k, 7] / m - c_media ** 2,
            "Dperp_ef_rel": (I[k, 7] - I[0, 7]) / (sim["D_perp"] * integral_G) if k > 0 else np.nan,
        }
        if "L_B" in sim:
            dx2 = f["x2_origen"] - x2o_0
            f.update(dyc_medido=yc - yc_0, dyc_pred_aprox=2 * sim["D_par"] * t / (2 * sim["L_B"]),
                     dyc_pred_x2=dx2 / (2 * sim["L_B"]),
                     dyc_identidad=dx2 / (2 * sim["L_B"]) - sim["D_perp"] * t / sim["L_B"])
        for nombre in sim["puntos"]:
            f["n" + nombre.replace("'", "_prima")] = r["n"][nombre][k]
        if exacta:
            S = matriz_S(sim, t)
            ex = lambda x, y, t=t: n_ex(x, y, t)
            f.update(error_L1_rel=postproc.error_L1_relativo(nodes, elements, N, ex),
                     error_L2_rel=postproc.error_L2_relativo(nodes, elements, N, ex),
                     x2_teo=S[0, 0], y2_teo=S[1, 1], aspecto_teo=aspecto_de_S(S),
                     pico_exacto=float(n_ex(0.0, 0.0, t)))
            for nombre, p in sim["puntos"].items():
                f["n" + nombre.replace("'", "_prima") + "_exacta"] = float(n_ex(*p, t))
        filas.append(f)
    return filas


def borde(sim, r):
    """Máximo de |n| en los nodos interiores vecinos a la frontera (todos los instantes guardados)."""
    Lx, Ly = sim["L"]
    h = r["h_borde"]
    x, y = r["nodes"][:, 0], r["nodes"][:, 1]
    cerca = (Lx - np.abs(x) < 1.5 * h) | (Ly - np.abs(y) < 1.5 * h)
    cerca[r["bnd"]] = False
    return max(np.abs(N[cerca]).max() for N in r["estados"].values())


def informe_campo(sim, corridas, filas, reportar):
    """Sección del .txt de una simulación con B general."""
    Lx, Ly = sim["L"]
    T = sim["T"]
    reportar("=" * 100)
    reportar(f"SIMULACIÓN {sim['nombre']}: {sim['titulo']}")
    reportar("=" * 100)
    reportar("PARÁMETROS")
    reportar(f"  D_par  = {sim['D_par_cgs']:.0e} cm^2/s = {sim['D_par']:.3f} pc^2/kyr")
    reportar(f"  D_perp = {sim['D_perp_cgs']:.0e} cm^2/s = {sim['D_perp']:.5f} pc^2/kyr"
             f"  (eps = D_perp / D_par = {sim['D_perp'] / sim['D_par']:g})")
    reportar(f"  {sim['campo_texto']}")
    reportar(f"  Dominio [-{Lx:g}, {Lx:g}] x [-{Ly:g}, {Ly:g}] pc; n0 = {sim['n_max']:g} exp(-(x^2 + y^2) /"
             f" (2 {sim['sigma0']:g}^2)); Q = 0; Dirichlet n = 0")
    reportar(f"  theta = {sim['theta']:g}, dt = {sim['dt']:g} kyr, T = {T:g} kyr ({int(round(T / sim['dt']))} pasos)")
    s_par = np.sqrt(sim["sigma0"] ** 2 + 2 * sim["D_par"] * T)
    s_perp = np.sqrt(sim["sigma0"] ** 2 + 2 * sim["D_perp"] * T)
    reportar(f"  Anchos de referencia en t = {T:g} kyr (B recto): sigma_par = {s_par:.1f} pc, sigma_perp = {s_perp:.1f} pc")
    for nota in sim["notas"]:
        reportar("  " + nota)
    reportar("  Corridas:")
    for c in corridas:
        m = c["malla"]
        if m["tipo"] == "estructurada":
            desc = f"estructurada {m['nodos'][0]}x{m['nodos'][1]}, h = {c['r']['h']:g} pc"
        else:
            q = mesh.calidad_malla(c["r"]["nodes"], c["r"]["elements"])
            desc = (f"gmsh, h = {m['h_dentro']:g} pc en |x| <= {m['ventana'][0]:g}, |y| <= {m['ventana'][1]:g} pc,"
                    f" {m['h_fuera']:g} pc fuera (transición de {m['transicion']:g} pc), ángulo mínimo"
                    f" {q['angulo_min']:.1f} grados")
        ref = " [referencia]" if m.get("referencia") else ""
        reportar(f"    {c['corrida']:<16s} {len(c['r']['nodes'])} nodos, {desc}{ref}")
        reportar(f"    {'':<16s} {c['sim']['campo_texto']}; {c['segundos']:.0f} s")
    if "L_B" in sim:
        reportar("  Difusión perpendicular numérica esperada por la curvatura (08_sovinec.py):"
                 f" kappa_num ~ 0.5 (h / R)^2 D_par, R = {sim['L_B']:g} pc")
        for h in sorted({tamano_malla(sim, m) for m in sim["mallas"]}, reverse=True):
            k_num = 0.5 * (h / sim["L_B"]) ** 2 * sim["D_par"]
            reportar(f"    h = {h:g} pc: kappa_num ~ {k_num:.2e} pc^2/kyr = {k_num / sim['D_perp']:.3f} D_perp"
                     " (en el vértice; R crece lejos de x = 0)")
    reportar()

    por = {c["corrida"]: [f for f in filas if f["corrida"] == c["corrida"]] for c in corridas}

    reportar("FRONTERA Y POSITIVIDAD")
    for c in corridas:
        fs = por[c["corrida"]]
        b = borde(c["sim"], c["r"])
        reportar(f"  {c['corrida']:<18s} max |n| junto a la frontera = {b:.1e} ({b / fs[0]['pico']:.1e} del pico inicial);"
                 f" masa perdida {1 - fs[-1]['masa_rel']:.1e}; min n = {c['r']['minimo'].min():.1e}")
        if c["sim"].get("exacta"):
            ex = fields.solucion_gaussiana(sim["n_max"], 0, 0, sim["sigma0"], tensor(c["sim"]))
            nb = c["r"]["nodes"][c["r"]["bnd"]]
            reportar(f"  {'':<18s} valor exacto máximo en la frontera en t = {T:g} kyr: {ex(nb[:, 0], nb[:, 1], T).max():.1e}")
    for c in corridas:
        if c["sim"].get("exacta"):
            continue
        r = c["r"]
        k = int(np.argmin(r["minimo"]))
        N = r["estados"].get(round(r["t"][k], 6))
        lugar = ""
        if N is not None:
            i = int(np.argmin(N))
            lugar = f" en ({r['nodes'][i, 0]:g}, {r['nodes'][i, 1]:g}) pc"
        reportar(f"  {c['corrida']}: mínimo {r['minimo'][k]:.2e} en t = {r['t'][k]:g} kyr{lugar}"
                 f" (pico en ese instante {r['pico'][k]:.3f}).")
    reportar("  En las mallas estructuradas los negativos aparecen justo debajo de la nube, cerca de A' (ver la")
    reportar("  tabla de abajo, x_min y y_min): con anisotropía fuerte y B inclinado respecto a la malla, el")
    reportar("  esquema P1 no garantiza el principio del máximo discreto. Bajan unas 20 veces al pasar de")
    reportar("  h = 2.5 a 1.25 pc; en la malla de gmsh (sin dirección preferida) y en el control (B alineado con")
    reportar("  la malla) son despreciables. Por eso n_A' en la malla de 281x121 no es confiable mientras es")
    reportar("  comparable a esos negativos (t <= 5 kyr).")
    reportar()

    reportar("MASA, EXTREMOS Y MOMENTOS (x2, y2, xy: momentos centrados en pc^2; aspecto = sqrt(l1 / l2))")
    reportar(f"  {'corrida':<18s} {'t':>4s} {'M/M0':>9s} {'min n':>10s} {'max n':>10s} {'x_pico':>7s} {'y_pico':>7s}"
             f" {'x_min':>7s} {'y_min':>7s} {'<x>':>7s} {'<y>':>7s} {'x2':>8s} {'y2':>8s} {'xy':>8s} {'aspecto':>8s}")
    for c in corridas:
        for f in por[c["corrida"]]:
            reportar(f"  {c['corrida']:<18s} {f['t_kyr']:4g} {f['masa_rel']:9.6f} {f['minimo']:10.2e} {f['pico']:10.4e}"
                     f" {f['x_pico']:7.2f} {f['y_pico']:7.2f} {f['x_min']:7.2f} {f['y_min']:7.2f} {f['xc']:7.3f} {f['yc']:7.3f} {f['x2']:8.2f}"
                     f" {f['y2']:8.2f} {f['xy']:8.2f} {f['aspecto']:8.3f}")
    reportar()

    ctrl = [c for c in corridas if c["sim"].get("exacta")]
    for c in ctrl:
        reportar(f"VERIFICACIÓN DEL CONTROL ({c['corrida']}) CONTRA LA GAUSSIANA ANISOTRÓPICA EXACTA")
        reportar(f"  {'t':>4s} {'L1 rel.':>9s} {'L2 rel.':>9s} {'x2':>8s} {'teo':>8s} {'y2':>8s} {'teo':>8s}"
                 f" {'aspecto':>8s} {'teo':>8s} {'pico':>10s} {'exacto':>10s}")
        for f in por[c["corrida"]]:
            reportar(f"  {f['t_kyr']:4g} {f['error_L1_rel']:9.2e} {f['error_L2_rel']:9.2e} {f['x2']:8.2f}"
                     f" {f['x2_teo']:8.2f} {f['y2']:8.3f} {f['y2_teo']:8.3f} {f['aspecto']:8.3f}"
                     f" {f['aspecto_teo']:8.3f} {f['pico']:10.4e} {f['pico_exacto']:10.4e}")
        reportar()

    if "L_B" in sim:
        LB = sim["L_B"]
        reportar("CENTRO DE MASA: ASCENSO A LO LARGO DE LAS PARÁBOLAS")
        reportar(f"  Predicción aproximada: las partículas se mueven sobre y = x^2 / {2 * LB:g} + c, así que")
        reportar(f"  D<y> ~ D<x^2> / {2 * LB:g}. Con D<x^2> ~ 2 D_par t (B recto) da D<y> ~ D_par t / {LB:g}; esa es")
        reportar("  una aproximación: la difusión es por longitud de arco y no en x, así que <x^2> crece")
        reportar("  menos que 2 D_par t. Usando el D<x^2> medido, la relación es además exacta salvo un")
        reportar(f"  término pequeño: con c = y - x^2 / {2 * LB:g} (constante sobre cada línea), grad c es")
        reportar(f"  perpendicular a b, D grad c = D_perp grad c, y d<c>/dt = D_perp lap c = -D_perp / {LB:g}. Así")
        reportar(f"  D<y> = D<x^2> / {2 * LB:g} - D_perp t / {LB:g} (columna 'identidad'), si la frontera no pierde masa.")
        reportar(f"  {'corrida':<18s} {'t':>4s} {'D<x^2>':>9s} {'2 D_par t':>10s} {'D<y> medido':>12s} {'aprox.':>8s}"
                 f" {'con D<x^2>':>11s} {'identidad':>10s}")
        for c in corridas:
            if c["sim"].get("exacta"):
                continue
            for f in por[c["corrida"]]:
                dx2 = f["dyc_pred_x2"] * 2 * LB
                reportar(f"  {c['corrida']:<18s} {f['t_kyr']:4g} {dx2:9.2f} {2 * sim['D_par'] * f['t_kyr']:10.2f}"
                         f" {f['dyc_medido']:12.4f} {f['dyc_pred_aprox']:8.4f} {f['dyc_pred_x2']:11.4f}"
                         f" {f['dyc_identidad']:10.4f}")
        reportar("  D<.> = valor en t menos valor en t = 0; distancias en pc. En el control (B horizontal) <y> = 0.")
        reportar()

    reportar("DIFUSIÓN PERPENDICULAR EFECTIVA")
    reportar("  Con c constante sobre las líneas de campo, d/dt int n c^2 = D_perp int n (2 |grad c|^2 + 2 c lap c)")
    reportar("  (sin pérdidas por la frontera). D_perp,ef / D_perp = cambio medido de int n c^2 / (D_perp por la")
    reportar("  integral en el tiempo del lado derecho); vale 1 sin difusión numérica, y el exceso mide kappa_num / D_perp.")
    reportar(f"  {'corrida':<18s} {'c':<16s} " + " ".join(f"{'t = ' + format(t, 'g'):>10s}" for t in sim["t_metricas"][1:])
             + f" {'var c (t = T)':>14s}")
    for c in corridas:
        fs = por[c["corrida"]]
        reportar(f"  {c['corrida']:<18s} {c['sim']['linea']['texto']:<16s} "
                 + " ".join(f"{f['Dperp_ef_rel']:10.4f}" for f in fs[1:]) + f" {fs[-1]['var_c']:14.3f}")
    reportar(f"  var c en pc^2; sin difusión numérica, en el control var c = sigma0^2 + 2 D_perp t = {s_perp ** 2:.3f}.")
    curvas = [c for c in corridas if not c["sim"].get("exacta") and c["malla"]["tipo"] == "estructurada"]
    if len(curvas) >= 2:
        g, f_ = por[curvas[0]["corrida"]][-1], por[curvas[-1]["corrida"]][-1]
        q = g["h_pc"] / f_["h_pc"]
        rich = f_["Dperp_ef_rel"] + (f_["Dperp_ef_rel"] - g["Dperp_ef_rel"]) / (q ** 2 - 1)
        orden = np.log((1 - g["Dperp_ef_rel"]) / (1 - f_["Dperp_ef_rel"])) / np.log(q)
        reportar(f"  En t = {T:g}: la desviación respecto a 1 pasa de {g['Dperp_ef_rel'] - 1:+.4f} (h = {g['h_pc']:g}) a"
                 f" {f_['Dperp_ef_rel'] - 1:+.4f} (h = {f_['h_pc']:g}), orden {orden:.2f};")
        reportar(f"  la extrapolación de Richardson con orden 2 (mallas estructuradas) da {rich:.4f}.")
    reportar()

    nombres = list(sim["puntos"])
    reportar("DENSIDAD EN LOS PUNTOS DE PRUEBA: " + ";  ".join(f"{k} = {p}" for k, p in sim["puntos"].items()))
    reportar(f"  {'corrida':<18s} {'t':>4s} " + " ".join(f"{'n_' + k:>11s}" for k in nombres) + f" {'n_A/n_A' + chr(39):>11s}")
    clave = lambda k: "n" + k.replace("'", "_prima")
    for c in corridas:
        for f in por[c["corrida"]]:
            reportar(f"  {c['corrida']:<18s} {f['t_kyr']:4g} " + " ".join(f"{f[clave(k)]:11.4e}" for k in nombres)
                     + f" {f['nA'] / f['nA_prima']:11.3e}")
        if c["sim"].get("exacta"):
            fT = por[c["corrida"]][-1]
            reportar(f"  {'  exacta control':<18s} {T:4g} " + " ".join(f"{fT[clave(k) + '_exacta']:11.4e}" for k in nombres)
                     + f" {fT['nA_exacta'] / fT['nA_prima_exacta']:11.3e}")
    reportar()

    # El problema es simétrico respecto a x -> -x (b(-x) es el reflejo de b(x)); la malla
    # estructurada no lo es (todas sus diagonales van en la misma dirección).
    reportar(f"SIMETRÍA x -> -x EN t = {T:g} kyr (el problema la tiene; la malla estructurada no)")
    reportar(f"  {'corrida':<18s} {'punto':>12s} {'n(x, y)':>11s} {'n(-x, y)':>11s} {'cociente':>9s}")
    for c in corridas:
        r = c["r"]
        N = r["estados"][round(T, 6)]
        for nombre, (px, py) in sim["puntos"].items():
            if px == 0:
                continue
            v = postproc.evaluar_en_puntos(r["nodes"], r["elements"], N, np.array([px, -px]), np.array([py, py]))
            reportar(f"  {c['corrida']:<18s} {nombre + ' ' + str((px, py)):>12s} {v[0]:11.4e} {v[1]:11.4e}"
                     f" {v[0] / v[1]:9.4f}")
    reportar("  En las mallas estructuradas, a un lado de x = 0 B corta las diagonales y al otro las sigue: en el")
    reportar("  campo curvo el lado x > 0 muestra una franja espuria con n ~ 1e-4 debajo de la nube y los")
    reportar("  negativos de arriba. A' está en ese lado, así que su densidad en esas mallas tiene ese error,")
    reportar("  que se reduce al refinar. La malla de gmsh no tiene una dirección preferida.")
    reportar()

    mallas = [c for c in corridas if not c["sim"].get("exacta")]
    ref = next((c for c in mallas if c["malla"].get("referencia")), mallas[-1])
    if len(mallas) >= 2:
        fr = por[ref["corrida"]][-1]
        otras = [c for c in mallas if c is not ref]
        reportar(f"COMPARACIÓN ENTRE MALLAS EN t = {T:g} kyr (referencia: {ref['corrida']}; dif. rel. = (malla - ref) / ref)")
        reportar(f"  {'métrica':<20s} " + " ".join(f"{c['corrida']:>14s}" for c in otras) + f" {ref['corrida']:>14s} "
                 + " ".join(f"{'dif. ' + c['malla']['nombre']:>13s}" for c in otras))
        for k, nombre in (("masa_rel", "M/M0"), ("minimo_hasta_t", "min n (hasta T)"), ("pico", "max n"),
                          ("yc", "<y> (pc)"), ("x2", "x2 (pc^2)"), ("y2", "y2 (pc^2)"), ("var_c", "var c (pc^2)"),
                          ("Dperp_ef_rel", "D_perp,ef / D_perp"), ("dyc_medido", "D<y> (pc)"),
                          ("dyc_identidad", "identidad (pc)"), ("nA", "n_A"), ("nA_prima", "n_A'"), ("nB", "n_B")):
            vals = [por[c["corrida"]][-1][k] for c in otras]
            dif = " ".join(f"{(v - fr[k]) / abs(fr[k]):13.2e}" if k != "minimo_hasta_t" else f"{'':>13s}" for v in vals)
            reportar(f"  {nombre:<20s} " + " ".join(f"{v:14.6e}" for v in vals) + f" {fr[k]:14.6e} " + dif)
        reportar()

    # Conclusiones a partir de los números.
    base = por[ref["corrida"]][-1]
    reportar("RESUMEN")
    reportar(f"  En {ref['corrida']}, n_A / n_A' = {base['nA'] / base['nA_prima']:.2e} en t = {T:g} kyr: los rayos")
    reportar("  cósmicos siguen la parábola que pasa por la fuente y no la recta y = 0.")
    for c in ctrl:
        fc = por[c["corrida"]][-1]
        reportar(f"  En el control (B recto), n_A / n_A' = {fc['nA'] / fc['nA_prima']:.2e}: al revés, A' está sobre la"
                 " línea de campo de la fuente.")
    if "L_B" in sim:
        reportar(f"  El centro de masa sube {base['dyc_medido']:.2f} pc (aproximación con 2 D_par t:"
                 f" {base['dyc_pred_aprox']:.2f} pc; con el <x^2> medido: {base['dyc_identidad']:.2f} pc).")
    reportar()


def figuras_campo(sim, corridas):
    """Mapas de sim2 con líneas de campo, comparación con el control y perfiles."""
    T = sim["T"]
    base = RESULTADOS / f"15_{sim['nombre']}"
    norma = LogNorm(vmin=1e-5, vmax=1.0)
    niveles = np.logspace(-5, 0, 31)
    Lx, Ly = sim["L"]
    ventana = (-200.0, 200.0)
    mallas = [c for c in corridas if not c["sim"].get("exacta")]
    principal = next((c for c in mallas if c["malla"].get("referencia")), mallas[0])
    control = next((c for c in corridas if c["sim"].get("exacta")), None)
    colores = {"A": "#f97316", "A'": "#ef4444", "B": "#e5e7eb"}

    def mapa(ax, c, t, titulo):
        r = c["r"]
        x, y = r["nodes"][:, 0], r["nodes"][:, 1]
        cf = ax.tricontourf(x, y, r["elements"], np.clip(r["estados"][round(t, 6)], 1e-5, None),
                            levels=niveles, norm=norma, cmap="viridis", extend="min")
        xg, yg = np.linspace(*ventana, 81), np.linspace(-Ly, Ly, 61)
        X, Y = np.meshgrid(xg, yg)
        Bx, By = c["sim"]["campo"](X, Y)
        ax.streamplot(xg, yg, Bx, By, color="#d1d5db", linewidth=0.5, density=0.9, arrowstyle="-")
        for nombre, (px, py) in sim["puntos"].items():
            ax.plot(px, py, "o", markersize=7, markerfacecolor="none", markeredgecolor=colores[nombre], mew=2)
            ax.annotate(nombre, (px, py), textcoords="offset points", xytext=(6, 5), color=colores[nombre],
                        fontsize=11, fontweight="bold")
        ax.set_xlim(*ventana)
        ax.set_ylim(-Ly, Ly)
        ax.set_aspect("equal")
        ax.set_xlabel("x (pc)")
        ax.set_ylabel("y (pc)")
        ax.set_title(titulo, fontsize=11)
        ax.grid(False)
        return cf

    def barra(fig, axes, cf):
        cb = fig.colorbar(cf, ax=axes, shrink=0.8, label="n (escala logarítmica, adimensional)")
        cb.set_ticks([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1])

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    for ax, t in zip(axes.ravel(), sim["t_mapas"]):
        cf = mapa(ax, principal, t, f"t = {t:g} kyr")
    barra(fig, axes.ravel().tolist(), cf)
    fig.suptitle(f"{sim['nombre']}: {sim['titulo']}, B = (1, x/{sim['L_B']:g}), malla {principal['malla']['nombre']}\n"
                 rf"$D_\parallel$ = {sim['D_par']:.2f} pc$^2$/kyr, $D_\perp$ = {sim['D_perp']:.4f} pc$^2$/kyr"
                 f" (ventana x en [-200, 200] pc; líneas de campo en gris)")
    viz.guardar(fig, f"{base}_mapas")

    nombres = [f"15_{sim['nombre']}_mapas"]
    if control is not None:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), constrained_layout=True)
        mapa(axes[0], control, T, f"Control: B uniforme horizontal, t = {T:g} kyr (malla {control['malla']['nombre']})")
        cf = mapa(axes[1], principal, T, f"{sim['nombre']}: B = (1, x/{sim['L_B']:g}), t = {T:g} kyr"
                                         f" (malla {principal['malla']['nombre']})")
        barra(fig, axes.tolist(), cf)
        fig.suptitle("Mismos D_par y D_perp y misma escala de colores")
        viz.guardar(fig, f"{base}_comparacion")
        nombres.append(f"15_{sim['nombre']}_comparacion")

    # Perfiles a lo largo de la parábola que pasa por la fuente y de y = 0.
    LB = sim["L_B"]
    xs = np.linspace(-0.98 * np.sqrt(2 * LB * Ly), 0.98 * np.sqrt(2 * LB * Ly), 1201)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    curvas = [(principal, "-", 2.0, f"{sim['nombre']} ({principal['malla']['nombre']})")]
    if control is not None:
        curvas.append((control, "--", 1.4, "control"))
    for ax in axes:
        for c, estilo, ancho, etiqueta in curvas:
            r = c["r"]
            N = r["estados"][round(T, 6)]
            par = postproc.evaluar_en_puntos(r["nodes"], r["elements"], N, xs, xs ** 2 / (2 * LB))
            rec = postproc.evaluar_en_puntos(r["nodes"], r["elements"], N, xs, 0 * xs)
            ax.plot(xs, par, estilo, color="#0072B2", linewidth=ancho,
                    label=f"{etiqueta}, parábola y = x²/{2 * LB:g}")
            ax.plot(xs, rec, estilo, color="#D55E00", linewidth=ancho, label=f"{etiqueta}, recta y = 0")
        ax.axvline(sim["puntos"]["A"][0], color="#9ca3af", linestyle=":", linewidth=1)
        ax.set_xlabel("x (pc)")
        ax.set_ylabel("n (adimensional)")
    axes[1].set_yscale("log")
    axes[1].set_ylim(1e-8, 0.1)
    axes[0].set_title("Escala lineal", fontsize=11)
    axes[1].set_title("Escala logarítmica (línea punteada: x = 50 pc, puntos A y A')", fontsize=11)
    axes[0].legend(fontsize=8)
    fig.suptitle(f"Perfiles de n en t = {T:g} kyr a lo largo de la parábola que pasa por la fuente y de y = 0")
    viz.guardar(fig, f"{base}_perfiles")
    nombres.append(f"15_{sim['nombre']}_perfiles")
    return nombres


def procesar_campo(sim, reportar):
    corridas = []
    configs = [(sim, m, f"{sim['nombre']} {m['nombre']}") for m in sim["mallas"]]
    if sim.get("control"):
        ctrl = dict(sim, **sim["control"])
        ctrl.pop("L_B", None)                       # la predicción de las parábolas no aplica
        configs += [(ctrl, m, f"control {m['nombre']}") for m in ctrl["mallas"]]
    for s, malla, nombre in configs:
        t1 = time.time()
        r = correr(s, malla, s["campo"], s["puntos"], s["linea"])
        corridas.append({"sim": s, "malla": malla, "corrida": nombre, "r": r, "segundos": time.time() - t1})
        print(f"  {nombre}: {len(r['nodes'])} nodos ({time.time() - t1:.0f} s)", flush=True)
    filas = []
    for c in corridas:
        filas += metricas_campo(c["sim"], c["r"], c["corrida"], c["malla"])
    informe_campo(sim, corridas, filas, reportar)
    return filas, figuras_campo(sim, corridas)


def main():
    rutas.avisar()
    RESULTADOS.mkdir(exist_ok=True)
    viz.estilo()
    lineas = []

    def reportar(texto=""):
        print(texto, flush=True)
        lineas.append(texto)

    reportar("SIMULACIONES ADICIONALES: REMANENTE DE SUPERNOVA (contexto de Nava y Gabici, 2013)")
    reportar("Se trabaja en pc y kyr; la verificación es contra la gaussiana exacta en el plano infinito.")
    reportar()

    todas, nombres_fig = [], []
    for sim in SIMULACIONES:
        sim = dict(sim, D_par=a_pc2_kyr(sim["D_par_cgs"]), D_perp=a_pc2_kyr(sim["D_perp_cgs"]))
        filas, figs = (procesar_campo if "campo" in sim else procesar_uniforme)(sim, reportar)
        todas += filas
        nombres_fig += figs

    columnas = []
    for f in todas:
        columnas += [k for k in f if k not in columnas]
    with open(RESULTADOS / "15_simulaciones.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=columnas)
        w.writeheader()
        w.writerows(todas)
    (RESULTADOS / "15_simulaciones.txt").write_text("\n".join(lineas) + "\n", encoding="utf-8")
    print(f"Guardado en {RESULTADOS.name}/: 15_simulaciones.txt, 15_simulaciones.csv y figuras "
          + ", ".join(nombres_fig) + " (png, pdf)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
