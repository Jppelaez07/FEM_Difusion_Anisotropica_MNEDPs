"""Escenarios de simulación y ejecución de un caso.

Cada caso es un diccionario de datos; los valores comunes están en COMUN
y cada caso solo indica lo que cambia. Unidades adimensionales.
"""

import copy
import json
from pathlib import Path

import numpy as np

from . import assembly, fields, mesh, postproc, timestep

RAIZ = Path(__file__).resolve().parents[1]
DIR_CASOS = RAIZ / "results" / "casos"

COMUN = {
    "dominio": {"Lx": 20.0, "Ly": 20.0},
    "malla": {"h_global": 0.5, "h_source": 0.1, "r_ref": 4.0},
    "D_par": 1.0,
    "D_perp": 0.1,
    "campo": {"tipo": "uniforme", "alpha_grados": 0.0},
    "fuente": {"Q0": 1.0, "sigma": 0.5, "xs": 10.0, "ys": 10.0, "t_apagado": None},
    "n0": {"tipo": "cero"},
    "bc": "dirichlet",
    "theta": 1.0,
    "dt": 0.05,
    "T": 40.0,
    "n_guardar": 50,
}

CASOS = {
    "caso1": {"titulo": "Caso 1: isotrópico", "D_par": 1.0, "D_perp": 1.0},
    "caso2": {"titulo": "Caso 2: D_par/D_perp = 10, B horizontal", "D_perp": 0.1},
    "caso3": {"titulo": "Caso 3: D_par/D_perp = 100, B horizontal", "D_perp": 0.01},
    "caso4a": {"titulo": "Caso 4a: B sinusoidal",
               "campo": {"tipo": "sinusoidal", "A": 1.0, "lam": 10.0}},
    "caso4b": {"titulo": "Caso 4b: B circular, D_par/D_perp = 100", "D_perp": 0.01,
               "campo": {"tipo": "circular", "xc": 10.0, "yc": 10.0},
               "fuente": {"xs": 14.0, "ys": 10.0}},
    "caso5_dirichlet": {"titulo": "Caso 5: fuente pulsada, Dirichlet",
                        "campo": {"tipo": "uniforme", "alpha_grados": 30.0},
                        "fuente": {"t_apagado": 5.0}},
    "caso5_neumann": {"titulo": "Caso 5: fuente pulsada, Neumann", "bc": "neumann",
                      "campo": {"tipo": "uniforme", "alpha_grados": 30.0},
                      "fuente": {"t_apagado": 5.0}},
}


def _combinar(base, cambios):
    """Copia de base con los cambios aplicados (también en subdiccionarios)."""
    salida = copy.deepcopy(base)
    for k, v in cambios.items():
        if isinstance(v, dict) and isinstance(salida.get(k), dict):
            salida[k] = _combinar(salida[k], v)
        else:
            salida[k] = copy.deepcopy(v)
    return salida


def config(nombre):
    """Configuración completa de un caso."""
    c = _combinar(COMUN, CASOS[nombre])
    c["nombre"] = nombre
    return c


def construir_campo(c):
    tipo = c["campo"]["tipo"]
    if tipo == "uniforme":
        return fields.campo_uniforme(np.radians(c["campo"]["alpha_grados"]))
    if tipo == "sinusoidal":
        return fields.campo_sinusoidal(c["campo"]["A"], c["campo"]["lam"])
    if tipo == "circular":
        return fields.campo_circular(c["campo"]["xc"], c["campo"]["yc"])
    raise ValueError(f"Campo desconocido: {tipo}")


def construir_fuente(c):
    f = c["fuente"]
    factor = None
    if f["t_apagado"] is not None:
        # Encendida en (0, t_apagado]: con dt que divide a t_apagado, el
        # esquema integra exactamente Q0 * t_apagado
        t_off = f["t_apagado"]
        factor = lambda t: 1.0 if t <= t_off + 1e-12 else 0.0
    return fields.fuente_gaussiana(f["Q0"], f["xs"], f["ys"], f["sigma"], factor)


def construir_n0(c):
    if c["n0"]["tipo"] == "cero":
        return lambda x, y: np.zeros_like(np.asarray(x, dtype=float))
    if c["n0"]["tipo"] == "gaussiana":
        f = c["fuente"]
        return fields.condicion_inicial_gaussiana(c["n0"]["n_max"], f["xs"], f["ys"],
                                                  c["n0"]["sigma"])
    raise ValueError(f"Condición inicial desconocida: {c['n0']['tipo']}")


def correr_caso(c, guardar=True):
    """Corre un caso, calcula las métricas y las guarda en results/casos/<nombre>.npz."""
    Lx, Ly = c["dominio"]["Lx"], c["dominio"]["Ly"]
    f = c["fuente"]
    nodes, elements, bnd, _ = mesh.make_mesh(Lx, Ly, c["malla"]["h_global"],
                                             c["malla"]["h_source"], f["xs"], f["ys"],
                                             c["malla"]["r_ref"])
    B = construir_campo(c)
    D_func = fields.funcion_tensor(B, c["D_par"], c["D_perp"])
    Q = construir_fuente(c)
    n0 = construir_n0(c)
    T, dt, theta = c["T"], c["dt"], c["theta"]
    t_guardar = np.linspace(0.0, T, c["n_guardar"] + 1)

    # Operadores para las métricas por paso
    geom = assembly.geometria(nodes, elements)
    M, _ = assembly.ensamblar_MK(nodes, elements, D_func, geom=geom)
    fila_masa = np.asarray(M.sum(axis=0)).ravel()          # 1^T M
    g_flujo = postproc.vector_flujo_frontera(nodes, elements, D_func)

    reg = {k: [] for k in ("t", "masa", "entrada", "flujo_directo", "cambio_rel",
                           "n_min", "n_max")}
    previo = {"N": None}

    def al_paso(k, t, N):
        reg["t"].append(t)
        reg["masa"].append(fila_masa @ N)
        reg["entrada"].append(assembly.ensamblar_F(nodes, elements, Q, t, geom=geom).sum())
        reg["flujo_directo"].append(g_flujo @ N)
        if previo["N"] is None:
            reg["cambio_rel"].append(np.nan)
        else:
            d = N - previo["N"]
            reg["cambio_rel"].append(np.sqrt(d @ (M @ d)) / max(np.sqrt(N @ (M @ N)), 1e-300))
        reg["n_min"].append(N.min())
        reg["n_max"].append(N.max())
        previo["N"] = N.copy()

    res = timestep.simular(nodes, elements, bnd, D_func, n0, T, dt, theta=theta, Q=Q,
                           bc=c["bc"], t_guardar=t_guardar, al_paso=al_paso)
    r = {k: np.array(v) for k, v in reg.items()}
    dt_real = r["t"][1] - r["t"][0]

    # Entrada y salida en cada paso, con la misma ponderación theta del esquema
    entrada_th = theta * r["entrada"][1:] + (1.0 - theta) * r["entrada"][:-1]
    flujo_balance = entrada_th - np.diff(r["masa"]) / dt_real       # int Q - dM/dt
    salida_directa_th = theta * r["flujo_directo"][1:] + (1.0 - theta) * r["flujo_directo"][:-1]
    acum = lambda v: np.concatenate(([0.0], np.cumsum(v) * dt_real))

    # Métricas en los instantes guardados
    N_g = res["N"]
    with np.errstate(invalid="ignore", divide="ignore"):       # masa nula en t = 0 si n0 = 0
        mom = postproc.momentos(nodes, M, N_g)
        aspecto = postproc.razon_aspecto(mom)

    salida = {
        "nombre": c["nombre"],
        "config": json.dumps(c, ensure_ascii=False),
        "nodes": nodes, "elements": elements, "boundary_nodes": bnd,
        "t_guardados": res["t"], "N_guardados": N_g,
        "mom_xx": mom["xx"], "mom_yy": mom["yy"], "mom_xy": mom["xy"],
        "mom_xc": mom["xc"], "mom_yc": mom["yc"], "razon_aspecto": aspecto,
        "t": r["t"], "masa": r["masa"], "entrada": r["entrada"],
        "flujo_balance": np.concatenate(([np.nan], flujo_balance)),
        "flujo_directo": r["flujo_directo"],
        "entrada_acum": acum(entrada_th),
        "salida_balance_acum": acum(flujo_balance),
        "salida_directa_acum": acum(salida_directa_th),
        "cambio_rel": r["cambio_rel"], "n_min": r["n_min"], "n_max": r["n_max"],
        "t_cuasi_estacionario": postproc.tiempo_cuasi_estacionario(r["t"], r["cambio_rel"]),
    }
    if guardar:
        DIR_CASOS.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(DIR_CASOS / f"{c['nombre']}.npz", **salida)
    return salida


def cargar_caso(nombre):
    """Lee results/casos/<nombre>.npz como diccionario."""
    with np.load(DIR_CASOS / f"{nombre}.npz", allow_pickle=False) as d:
        salida = {k: d[k] for k in d.files}
    salida["config"] = json.loads(str(salida["config"]))
    salida["nombre"] = str(salida["nombre"])
    salida["t_cuasi_estacionario"] = float(salida["t_cuasi_estacionario"])
    return salida
