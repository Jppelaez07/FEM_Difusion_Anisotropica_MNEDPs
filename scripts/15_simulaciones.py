"""Simulaciones adicionales en el contexto del remanente de supernova (Nava y Gabici, 2013).

Cada simulación es un diccionario de SIMULACIONES, identificado por un nombre corto; para
agregar otra basta con añadir un diccionario. Todas usan B uniforme, n0 gaussiana y Q = 0,
así que se verifican contra la gaussiana exacta en el plano infinito (fields.solucion_gaussiana).
Se trabaja en pc y kyr.

Simulación 1 (sim1): difusión isotrópica intensa, D_par = D_perp = 1e29 cm^2/s, en
[-350, 350]^2 pc, Crank-Nicolson, dt = 0.02 kyr, T = 10 kyr. Se compara con el caso
anisotrópico de 14_snr.py (eps = 0.01, mismo D_par).

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

from src import fields, mesh, postproc, rutas, timestep, viz  # noqa: E402

# --- Unidades ---
PC_CM = 3.0857e18
KYR_S = 3.156e10


def a_pc2_kyr(D_cgs):
    """Convierte un coeficiente de difusión de cm^2/s a pc^2/kyr."""
    return D_cgs * KYR_S / PC_CM ** 2


# Nubes de prueba de 14_snr.py: A a lo largo de B, B perpendicular, ambas a 50 pc.
NUBES = {"A": (50.0, 0.0), "B": (0.0, 50.0)}

# --- Simulaciones ---
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
]

RESULTADOS = rutas.RESULTADOS


def tensor(sim):
    """Tensor D constante (pc^2/kyr) de la simulación."""
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


def correr(sim):
    """Malla, ensamblaje y avance en el tiempo; registra pico, mínimo y nubes en cada paso."""
    Lx, Ly = sim["L"]
    nodes, elements, bnd, _ = mesh.malla_estructurada(2 * Lx, 2 * Ly, *sim["nodos"])
    nodes = nodes - np.array([Lx, Ly])
    D_func = fields.funcion_tensor(fields.campo_uniforme(sim["alpha"]), sim["D_par"], sim["D_perp"])
    n0 = fields.condicion_inicial_gaussiana(sim["n_max"], 0.0, 0.0, sim["sigma0"])
    P = {k: viz.matriz_rejilla(nodes, elements, np.array([p[0]]), np.array([p[1]]))
         for k, p in NUBES.items()}
    reg = {"t": [], "pico": [], "minimo": [], "nA": [], "nB": []}

    def al_paso(k, t, N):
        reg["t"].append(t)
        reg["pico"].append(N.max())
        reg["minimo"].append(N.min())
        reg["nA"].append((P["A"] @ N)[0])
        reg["nB"].append((P["B"] @ N)[0])

    res = timestep.simular(nodes, elements, bnd, D_func, n0, sim["T"], sim["dt"],
                           theta=sim["theta"], Q=None, bc="dirichlet",
                           t_guardar=sorted(set(sim["t_metricas"]) | set(sim["t_mapas"])),
                           al_paso=al_paso)
    r = {k: np.array(v) for k, v in reg.items()}
    r["estados"] = dict(zip(np.round(res["t"], 6), res["N"]))
    r.update(nodes=nodes, elements=elements, bnd=bnd, M=res["M"])
    return r


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
    """Sección del .txt de una simulación: parámetros, verificación y comparación."""
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
        t1 = time.time()
        r = correr(sim)
        segundos = time.time() - t1
        filas, n_ex = metricas(sim, r)
        informe(sim, r, filas, n_ex, reportar, segundos)
        nombres_fig += figuras(sim, r, n_ex)
        todas += filas

    with open(RESULTADOS / "15_simulaciones.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(todas[0].keys()))
        w.writeheader()
        w.writerows(todas)
    (RESULTADOS / "15_simulaciones.txt").write_text("\n".join(lineas) + "\n", encoding="utf-8")
    print(f"Guardado en {RESULTADOS.name}/: 15_simulaciones.txt, 15_simulaciones.csv y figuras "
          + ", ".join(nombres_fig) + " (png, pdf)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
