"""Escenario físico: modelo de "leaky box" en el plano (x, z), basado en
Al-Zetoun y Achterberg (2018, MNRAS 477, 1258).

Dominio [-20, 20] x [-1, 1] (x horizontal, z vertical; L0 = 1 kpc), B uniforme,
D_par = 1, D_perp = eps D_par, n0 gaussiana, Q = 0, theta = 1.
Fronteras mixtas: Dirichlet n = 0 en z = +-1 (escape del halo) y Neumann en x = +-20.
Se mide la masa, el flujo de escape, la distribución de tiempos de residencia, el tiempo
medio de residencia y la distancia recorrida a lo largo del campo, y se comparan con
las ecuaciones (33)-(35) del artículo y con la solución exacta separable.
Unidades: L0 = 1 kpc, D_par = 3e28 cm^2/s, t0 = L0^2 / D_par. Se simula en adimensional
y se grafica en kpc y Myr. Guarda CSV, figuras PNG/PDF y results/13_halo.txt.
"""

import csv
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from src import assembly, fields, mesh, postproc, rutas, timestep, viz  # noqa: E402

# --- Unidades ---
KPC_CM = 3.0857e21
MYR_S = 3.156e13
D_PAR_FISICO = 3e28                       # cm^2/s
T0_S = KPC_CM ** 2 / D_PAR_FISICO
T0_MYR = T0_S / MYR_S                     # ~ 10.05 Myr

# --- Problema ---
X_MAX, H = 20.0, 1.0                      # dominio [-X_MAX, X_MAX] x [-H, H]
sigma_x0, sigma_z0 = 0.1, 0.05
D_par = 1.0
eps_base = 0.01
dt_base = 0.5 / T0_MYR                    # 0.5 Myr
T_base = 2500.0 / T0_MYR                  # 2500 Myr
T_propuesto_myr = 1000.0
t_mapas_myr = [0.0, 50.0, 200.0, 500.0, 1000.0]
eps_barrido = [0.01, 0.1, 0.5, 1.0]
tan_alfa_barrido = [0.1, 0.3]
pasos_por_tmedio = 500                    # dt = min(dt_base, <t>/500) en el barrido
tmedios_por_T = 12                        # T = min(T_base, 12 <t>)
n_momentos = 250                          # instantes con momentos por corrida
OBSERVADO_MYR = 100.0                     # t_res ~ 1e8 años a ~1 GeV

RESULTADOS = rutas.RESULTADOS
COLOR_EPS = {0.01: "#0072B2", 0.1: "#009E73", 0.5: "#E69F00", 1.0: "#D55E00"}

# (nombre, tipo, parámetros): estructurada (Nx, Nz) o gmsh (h_global, h_fuente)
MALLAS = [("estructurada 401x101", "estructurada", (401, 101)),
          ("gmsh h_g = 0.1", "gmsh", (0.1, 0.02)),      # contraste: no resuelve z fuera de la fuente
          ("gmsh h_g = 0.04", "gmsh", (0.04, 0.02)),
          ("estructurada 801x201", "estructurada", (801, 201))]
if rutas.RAPIDO:
    MALLAS = [m for m in MALLAS if m[0] != "estructurada 801x201"]


def a_myr(t):
    return np.asarray(t) * T0_MYR


def D_normal(eps, alfa):
    """Ecuación (33): D_n = D_perp (1 - sin^2 a) + D_par sin^2 a."""
    s2 = np.sin(alfa) ** 2
    return eps * D_par * (1.0 - s2) + D_par * s2


def t_ecuacion_34(Dn):
    return H ** 2 / (2.0 * Dn)


def t_primer_paso(Dn):
    """Media exacta del tiempo de primer paso con la gaussiana inicial en z."""
    return (H ** 2 - sigma_z0 ** 2) / (2.0 * Dn)


def serie_z(Dn, t, n_modos=400):
    """Masa relativa M(t)/M(0) y p(t) de la serie de cosenos de Dirichlet en [-H, H]."""
    t = np.atleast_1d(np.asarray(t, dtype=float))
    m = np.arange(n_modos)
    k = (2 * m + 1) * np.pi / (2 * H)
    a = sigma_z0 * np.sqrt(2 * np.pi) * np.exp(-0.5 * (k * sigma_z0) ** 2)   # coef. de la gaussiana
    integral_cos = 2.0 * (-1.0) ** m / k
    M0 = sigma_z0 * np.sqrt(2 * np.pi)
    e = np.exp(-Dn * np.outer(t, k ** 2))
    masa = e @ (a * integral_cos) / M0
    p = e @ (a * integral_cos * Dn * k ** 2) / M0
    return masa, p


def n0(x, z):
    return np.exp(-x ** 2 / (2 * sigma_x0 ** 2) - z ** 2 / (2 * sigma_z0 ** 2))


def construir_malla(tipo, parametros):
    if tipo == "estructurada":
        Nx, Nz = parametros
        nodes, elements, _, _ = mesh.malla_estructurada(2 * X_MAX, 2 * H, Nx, Nz)
    else:
        h_global, h_fuente = parametros
        nodes, elements, _, _ = mesh.make_mesh(2 * X_MAX, 2 * H, h_global, h_fuente, X_MAX, H, 0.5)
    nodes = nodes - np.array([X_MAX, H])
    dirichlet = np.flatnonzero(np.isclose(np.abs(nodes[:, 1]), H))     # solo z = +-1
    return nodes, elements, dirichlet


def correr(nodes, elements, dirichlet, eps, alfa, dt, T, t_mapas=()):
    """Simula y retorna las métricas por paso, los momentos y los mapas pedidos."""
    D_func = fields.funcion_tensor(fields.campo_uniforme(alfa), D_par, eps * D_par)
    M, _ = assembly.ensamblar_MK(nodes, elements, D_func)
    fila_masa = np.asarray(M.sum(axis=0)).ravel()
    G, medios = postproc.matriz_flujo_aristas(nodes, elements, D_func)
    arriba_abajo = np.isclose(np.abs(medios[:, 1]), H)
    Gz, x_arista = G[arriba_abajo], medios[arriba_abajo, 0]

    Nt = int(round(T / dt))
    cada = max(1, Nt // n_momentos)
    reg = {"t": [], "masa": [], "flujo_directo": [], "t_mom": [], "sx": []}
    acum = {"x2": 0.0, "f": 0.0, "previo": None}

    def al_paso(k, t, N):
        reg["t"].append(t)
        reg["masa"].append(fila_masa @ N)
        f = Gz @ N
        reg["flujo_directo"].append(f.sum())
        if k > 0:                                         # theta = 1: flujo al final del paso
            acum["x2"] += dt * np.sum(x_arista ** 2 * f)
            acum["f"] += dt * np.sum(f)
        if k % cada == 0:
            mom = postproc.momentos_cuadratura(nodes, elements, N)
            reg["t_mom"].append(t)
            reg["sx"].append(np.sqrt(mom["xx"][0]))

    res = timestep.simular(nodes, elements, dirichlet, D_func, n0, T, dt, theta=1.0, Q=None,
                           bc="dirichlet", t_guardar=list(t_mapas) or [T], al_paso=al_paso)
    r = {k: np.array(v) for k, v in reg.items()}
    dt_real = r["t"][1] - r["t"][0]
    M0 = r["masa"][0]
    r["phi_balance"] = np.concatenate(([0.0], -np.diff(r["masa"]) / dt_real))
    r["p"] = r["phi_balance"] / M0

    # Tiempo medio de residencia: integral hasta T + cola exponencial
    integral = np.trapezoid(r["masa"], r["t"]) / M0
    ultimo = r["t"] >= 0.8 * r["t"][-1]
    lam = -np.polyfit(r["t"][ultimo], np.log(r["masa"][ultimo]), 1)[0]
    cola = r["masa"][-1] / (lam * M0)
    r.update({"t_medio": integral + cola, "cola": cola, "lam_final": lam,
              "x_escape_rms": np.sqrt(acum["x2"] / acum["f"]),
              "masa_final_rel": r["masa"][-1] / M0, "dt": dt_real,
              "mapas": dict(zip(res["t"], res["N"])) if t_mapas else {}})
    return r


def guardar_csv(nombre, filas):
    with open(RESULTADOS / nombre, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(filas[0].keys()))
        w.writeheader()
        w.writerows(filas)


def resumen_corrida(nombre, malla, eps, alfa, r):
    Dn = D_normal(eps, alfa)
    t34, t_pp = t_ecuacion_34(Dn), t_primer_paso(Dn)
    masa_ex, p_ex = serie_z(Dn, r["t"])
    sel = r["t"] > 0
    err_masa = np.max(np.abs(r["masa"] / r["masa"][0] - masa_ex))
    err_p = np.max(np.abs(r["p"][sel] - p_ex[sel])) / np.max(p_ex[sel])
    sx_tmedio = np.interp(r["t_medio"], r["t_mom"], r["sx"])
    return {
        "corrida": nombre, "malla": malla, "eps": eps, "tan_alfa": np.tan(alfa), "D_n": Dn,
        "dt_myr": a_myr(r["dt"]), "T_myr": a_myr(r["t"][-1]),
        "t_medio_myr": a_myr(r["t_medio"]), "t_ec34_myr": a_myr(t34), "t_primer_paso_myr": a_myr(t_pp),
        "error_rel_vs_primer_paso": r["t_medio"] / t_pp - 1.0, "cola_pct": 100.0 * r["cola"] / r["t_medio"],
        "masa_final_rel": r["masa_final_rel"], "error_max_masa": err_masa, "error_max_p_rel": err_p,
        "ds_par_ec35_kpc": H * np.sqrt(D_par / Dn), "x_escape_rms_kpc": r["x_escape_rms"],
        "sigma_x_en_t_medio_kpc": sx_tmedio,
    }


def main():
    rutas.avisar()
    RESULTADOS.mkdir(exist_ok=True)
    viz.estilo()
    lineas = []

    def reportar(texto=""):
        print(texto, flush=True)
        lineas.append(texto)

    reportar("ESCENARIO FÍSICO: LEAKY BOX EN EL PLANO (x, z), BASADO EN AL-ZETOUN Y ACHTERBERG (2018)")
    reportar("UNIDADES")
    reportar(f"  L0 = 1 kpc = {KPC_CM:.4e} cm;  D_par = {D_PAR_FISICO:.1e} cm^2/s;  1 Myr = {MYR_S:.4e} s")
    reportar(f"  t0 = L0^2 / D_par = {T0_S:.4e} s = {T0_MYR:.4f} Myr")
    reportar(f"  {'magnitud':<26s} {'adimensional':>13s} {'físico':>16s}")
    for nombre, valor, fisico in [("longitud 1", 1.0, "1 kpc"), ("tiempo 1", 1.0, f"{T0_MYR:.3f} Myr"),
                                  ("dt base", dt_base, "0.5 Myr"), ("T", T_base, "2500 Myr"),
                                  ("T propuesto", T_propuesto_myr / T0_MYR, "1000 Myr"),
                                  ("D_par", 1.0, f"{D_PAR_FISICO:.0e} cm^2/s"),
                                  ("t observado ~1 GeV", OBSERVADO_MYR / T0_MYR, "100 Myr")]:
        reportar(f"  {nombre:<26s} {valor:13.4f} {fisico:>16s}")
    reportar("  B0 = 1 microG no entra en el modelo: solo importa la dirección de B a través de D.")
    reportar()
    reportar("PROBLEMA")
    reportar(f"  Dominio [-{X_MAX:g}, {X_MAX:g}] x [-{H:g}, {H:g}] kpc; n0 gaussiana con sigma_x = {sigma_x0:g},"
             f" sigma_z = {sigma_z0:g} kpc; Q = 0, theta = 1")
    reportar("  Dirichlet n = 0 en z = +-1 (escape) y Neumann en x = +-20 (reflejante).")

    # --- Mallas ---
    mallas = {}
    for nombre, tipo, par in MALLAS:
        nodes, elements, dirichlet = construir_malla(tipo, par)
        en_x = np.flatnonzero(np.isclose(np.abs(nodes[:, 0]), X_MAX) & ~np.isclose(np.abs(nodes[:, 1]), H))
        restringidos_x = np.intersect1d(en_x, dirichlet).size
        mallas[nombre] = (nodes, elements, dirichlet)
        reportar(f"  Malla {nombre}: {len(nodes)} nodos; {len(dirichlet)} nodos con Dirichlet (|z| = 1);"
                 f" de {en_x.size} nodos en x = +-20 (sin esquinas), {restringidos_x} restringidos")
    reportar()

    # --- Caso base en las mallas ---
    filas = []
    corridas = {}
    for i, (nombre, _, _) in enumerate(MALLAS):
        t1 = time.time()
        nodes, elements, dirichlet = mallas[nombre]
        mapas = [t / T0_MYR for t in t_mapas_myr] if i == 0 else ()
        r = correr(nodes, elements, dirichlet, eps_base, 0.0, dt_base, T_base, mapas)
        corridas[("base", nombre)] = r
        filas.append(resumen_corrida("base", nombre, eps_base, 0.0, r))
        print(f"  base, {nombre}: {time.time() - t1:.0f} s", flush=True)

    # --- Barridos en la malla de referencia ---
    malla_ref = MALLAS[0][0]
    nodes, elements, dirichlet = mallas[malla_ref]
    barridos = [("eps", e, 0.0) for e in eps_barrido if e != eps_base] + \
               [("inclinado", eps_base, np.arctan(ta)) for ta in tan_alfa_barrido]
    corridas[("eps", eps_base)] = corridas[("base", malla_ref)]
    for tipo, eps, alfa in barridos:
        t1 = time.time()
        t_pred = t_primer_paso(D_normal(eps, alfa))
        dt = min(dt_base, t_pred / pasos_por_tmedio)
        T = min(T_base, tmedios_por_T * t_pred)
        r = correr(nodes, elements, dirichlet, eps, alfa, dt, T)
        clave = ("eps", eps) if tipo == "eps" else ("inclinado", np.tan(alfa))
        corridas[clave] = r
        nombre = f"eps = {eps:g}" if tipo == "eps" else f"tan a = {np.tan(alfa):g}"
        filas.append(resumen_corrida(nombre, malla_ref, eps, alfa, r))
        print(f"  {nombre}: {time.time() - t1:.0f} s", flush=True)
    guardar_csv("13_halo_corridas.csv", filas)

    # --- Tablas ---
    reportar("TIEMPO MEDIO DE RESIDENCIA Y DISTANCIA A LO LARGO DEL CAMPO")
    reportar("  <t> FEM = int_0^T M dt / M(0) + cola exponencial; ec. 34: H^2/(2 D_n);"
             " exacta: (H^2 - sigma_z^2)/(2 D_n)")
    reportar(f"  {'corrida':<14s} {'malla':<21s} {'D_n':>7s} {'dt Myr':>7s} {'<t> FEM':>9s} {'ec. 34':>9s}"
             f" {'exacta':>9s} {'error':>8s} {'cola %':>7s} {'ds ec.35':>8s} {'x esc.':>7s} {'sx(<t>)':>7s}")
    for f in filas:
        reportar(f"  {f['corrida']:<14s} {f['malla']:<21s} {f['D_n']:7.4f} {f['dt_myr']:7.3f}"
                 f" {f['t_medio_myr']:9.2f} {f['t_ec34_myr']:9.2f} {f['t_primer_paso_myr']:9.2f}"
                 f" {f['error_rel_vs_primer_paso']:8.1e} {f['cola_pct']:7.3f} {f['ds_par_ec35_kpc']:8.2f}"
                 f" {f['x_escape_rms_kpc']:7.2f} {f['sigma_x_en_t_medio_kpc']:7.2f}")
    reportar("  tiempos en Myr y distancias en kpc. x esc. = raíz del promedio de x^2 ponderado por el")
    reportar("  flujo de escape por z = +-1; sx(<t>) = ancho a lo largo de x en t = <t>.")
    reportar()

    reportar("COMPARACIÓN CON LA SOLUCIÓN EXACTA SEPARABLE (serie de cosenos en z)")
    reportar(f"  {'corrida':<14s} {'malla':<21s} {'max |M/M0 - exacta|':>20s} {'max |p - p_ex| / max p_ex':>26s}"
             f" {'M(T)/M0':>9s}")
    for f in filas:
        reportar(f"  {f['corrida']:<14s} {f['malla']:<21s} {f['error_max_masa']:20.2e}"
                 f" {f['error_max_p_rel']:26.2e} {f['masa_final_rel']:9.2e}")
    reportar()

    reportar("CONVERGENCIA DEL CASO BASE ENTRE MALLAS")
    base = [f for f in filas if f["corrida"] == "base"]
    for f in base:
        reportar(f"  {f['malla']:<21s} <t> = {f['t_medio_myr']:.3f} Myr  (exacta {f['t_primer_paso_myr']:.3f};"
                 f" error {f['error_rel_vs_primer_paso']:+.2e})")
    reportar("  Las dos mallas de gmsh están refinadas alrededor de la fuente (h_fuente = 0.02, r_ref = 0.5).")
    reportar("  Con h_global = 0.1 la nube, que se estira rápido a lo largo de x pero sigue delgada en z,")
    reportar("  sale de la zona refinada a elementos que no la resuelven en z (H/h = 10), y el error de p")
    reportar("  se concentra en las primeras decenas de Myr. Con h_global = 0.04 coincide con las")
    reportar("  estructuradas: en este problema hay que resolver z en toda la franja, no solo cerca de la fuente.")
    r0 = corridas[("base", malla_ref)]
    k1000 = int(np.argmin(np.abs(a_myr(r0["t"]) - T_propuesto_myr)))
    reportar(f"  En T = {T_propuesto_myr:g} Myr (malla {malla_ref}): M/M0 = {r0['masa'][k1000] / r0['masa'][0]:.4e};"
             f" la integral hasta 1000 Myr da {a_myr(np.trapezoid(r0['masa'][:k1000 + 1], r0['t'][:k1000 + 1]) / r0['masa'][0]):.2f} Myr"
             f" de los {a_myr(r0['t_medio']):.2f} Myr totales.")
    reportar()

    reportar("ESCALAMIENTO CON eps Y CAMPO INCLINADO")
    for f in filas:
        if f["corrida"] == "base" and f["malla"] != malla_ref:
            continue
        reportar(f"  {f['corrida']:<14s} D_n = {f['D_n']:.4f}:  <t> D_n = {f['t_medio_myr'] * f['D_n']:.3f} Myr"
                 f"  (exacto (H^2 - sigma_z^2) t0 / 2 = {(H ** 2 - sigma_z0 ** 2) * T0_MYR / 2:.3f})")
    reportar("  Con B horizontal D_n = D_perp, así que <t> es exactamente proporcional a 1/eps.")
    f_inc = [f for f in filas if f["corrida"].startswith("tan")]
    for f in f_inc:
        reportar(f"  {f['corrida']}: D_n = {f['D_n']:.4f} = {f['D_n'] / eps_base:.2f} D_perp;"
                 f" <t> = {f['t_medio_myr']:.1f} Myr frente a {filas[0]['t_medio_myr']:.1f} Myr con B horizontal")
    reportar()

    # --- Discusión física ---
    t_base = filas[0]["t_medio_myr"]
    t_eps = {f["eps"]: f["t_medio_myr"] for f in filas if f["corrida"].startswith("eps") or f["corrida"] == "base"}
    eps_obs = (H ** 2 - sigma_z0 ** 2) * T0_MYR / (2 * OBSERVADO_MYR)
    sin2_obs = (eps_obs - eps_base) / (1 - eps_base)
    reportar("DISCUSIÓN FÍSICA")
    reportar(f"  El tiempo de residencia observado a ~1 GeV es t_res ~ 1e8 años = {OBSERVADO_MYR:g} Myr.")
    reportar(f"  Con B horizontal: eps = 0.01 da <t> = {t_base:.0f} Myr (artificialmente largo, como dice el")
    reportar(f"  artículo para D_perp pequeño); eps = 0.5 da {t_eps.get(0.5, np.nan):.1f} Myr y eps = 1 da"
             f" {t_eps.get(1.0, np.nan):.1f} Myr")
    reportar("  (demasiado corto con D_par = 3e28 cm^2/s).")
    reportar(f"  Para <t> = {OBSERVADO_MYR:g} Myr hace falta D_n = {eps_obs:.3f} D_par: con B horizontal,"
             f" eps ~ {eps_obs:.2f}; con eps = 0.01,")
    reportar(f"  basta una inclinación con sin^2 a ~ {sin2_obs:.3f} (tan a ~ {np.tan(np.arcsin(np.sqrt(sin2_obs))):.2f}).")
    reportar("  Una componente vertical pequeña de B reduce mucho el tiempo de residencia (ec. 33): con")
    for f in f_inc:
        reportar(f"    tan a = {f['tan_alfa']:.1f}: <t> = {f['t_medio_myr']:.1f} Myr ({t_base / f['t_medio_myr']:.1f} veces menor)")
    reportar("  Esa es la explicación del artículo de por qué el efecto de eps sobre el tiempo de")
    reportar("  residencia es menor que 1/eps en su modelo: el campo no es exactamente horizontal.")
    reportar(f"  Distancia a lo largo del campo: con eps = 0.01, la ec. 35 da {filas[0]['ds_par_ec35_kpc']:.1f} kpc;"
             f" la raíz del promedio de x^2 al escapar es {filas[0]['x_escape_rms_kpc']:.1f} kpc")
    reportar(f"  y el ancho de la nube en t = <t> es {filas[0]['sigma_x_en_t_medio_kpc']:.1f} kpc (menores por las"
             " paredes reflejantes en x = +-20).")
    reportar("  Fronteras: el artículo deja escapar en r = 20 kpc; aquí x = +-20 es reflejante. Con")
    reportar("  eps = 0.01 la nube llega a x = +-20 antes de escapar (en t = <t> su ancho es"
             f" {filas[0]['sigma_x_en_t_medio_kpc']:.1f} kpc;")
    reportar("  una distribución uniforme en [-20, 20] tendría 11.5 kpc),")
    reportar("  así que con escape lateral el tiempo de residencia sería más corto que el calculado aquí,")
    reportar("  que solo cuenta el escape vertical. Con eps grande la nube escapa antes de llegar a x = +-20.")
    reportar()

    # --- Figuras ---
    r0 = corridas[("base", malla_ref)]
    nodes, elements, _ = mallas[malla_ref]
    fig, axes = plt.subplots(len(t_mapas_myr), 1, figsize=(12, 2.0 * len(t_mapas_myr) + 0.6),
                             constrained_layout=True)
    for ax, (t, N) in zip(axes, sorted(r0["mapas"].items())):
        cf = ax.tricontourf(nodes[:, 0], nodes[:, 1], elements, N, levels=30, cmap="viridis")
        fig.colorbar(cf, ax=ax, pad=0.01, label="n", format="%.2e" if N.max() < 0.1 else "%.2f")
        ax.set_aspect("auto")
        ax.set_ylabel("z (kpc)")
        ax.set_title(f"t = {a_myr(t):.0f} Myr", fontsize=11)
        ax.grid(False)
    axes[-1].set_xlabel("x (kpc)")
    fig.suptitle(f"Caso base: B horizontal, eps = {eps_base:g}, malla {malla_ref}")
    viz.guardar(fig, RESULTADOS / "13_halo_mapas")

    fig, ax = plt.subplots(figsize=(7.5, 4.8), constrained_layout=True)
    for eps in eps_barrido:
        r = corridas[("eps", eps)]
        ax.semilogy(a_myr(r["t"]), r["masa"] / r["masa"][0], color=COLOR_EPS[eps], label=f"eps = {eps:g}")
        ax.semilogy(a_myr(r["t"]), serie_z(D_normal(eps, 0.0), r["t"])[0], color="#111827",
                    linestyle="--", linewidth=1.0)
    ax.plot([], [], color="#111827", linestyle="--", linewidth=1.0, label="exacta (serie)")
    ax.set_ylim(1e-4, 1.5)
    ax.set_xlim(0, 2500)
    ax.set_xlabel("t (Myr)")
    ax.set_ylabel("M(t) / M(0)")
    ax.set_title("Masa en el halo (escape por z = +-1 kpc)")
    ax.legend()
    viz.guardar(fig, RESULTADOS / "13_halo_masa")

    fig, ax = plt.subplots(figsize=(7.5, 4.8), constrained_layout=True)
    for eps in eps_barrido:
        r = corridas[("eps", eps)]
        sel = r["t"] > 0
        ax.loglog(a_myr(r["t"][sel]), r["p"][sel] / T0_MYR, color=COLOR_EPS[eps], label=f"eps = {eps:g}")
        ax.loglog(a_myr(r["t"][sel]), serie_z(D_normal(eps, 0.0), r["t"][sel])[1] / T0_MYR, color="#111827",
                  linestyle="--", linewidth=1.0)
        ax.axvline(a_myr(r["t_medio"]), color=COLOR_EPS[eps], linestyle=":", linewidth=1.2)
    ax.plot([], [], color="#111827", linestyle="--", linewidth=1.0, label="exacta (serie)")
    ax.plot([], [], color="#6b7280", linestyle=":", label=r"$\langle t \rangle$")
    ax.set_ylim(1e-7, 1)
    ax.set_xlabel("t (Myr)")
    ax.set_ylabel("p(t) (1/Myr)")
    ax.set_title("Distribución de tiempos de residencia")
    ax.legend(fontsize=9)
    viz.guardar(fig, RESULTADOS / "13_halo_residencia")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    fe = [f for f in filas if f["corrida"].startswith("eps") or (f["corrida"] == "base" and f["malla"] == malla_ref)]
    fe = sorted(fe, key=lambda f: f["eps"])
    e_arr = np.array([f["eps"] for f in fe])
    axes[0].loglog(e_arr, [f["t_medio_myr"] for f in fe], "o", color="#0072B2", markersize=8, label="FEM")
    ee = np.logspace(-2.2, 0.1, 50)
    axes[0].loglog(ee, a_myr(H ** 2 / (2 * ee * D_par)), color="#6b7280", linestyle="--",
                   label=r"$H^2 / (2 \varepsilon D_\parallel)$ (ec. 34)")
    axes[0].axhline(OBSERVADO_MYR, color="#D55E00", linestyle=":", label="observado ~100 Myr")
    axes[0].set_xlabel(r"$\varepsilon = D_\perp / D_\parallel$")
    axes[0].set_ylabel(r"$\langle t \rangle$ (Myr)")
    axes[0].set_title("B horizontal")
    axes[0].legend(fontsize=9)
    Dn_todos = [f["D_n"] for f in fe] + [f["D_n"] for f in f_inc]
    axes[1].loglog([f["D_n"] for f in fe], [f["t_medio_myr"] for f in fe], "o", color="#0072B2",
                   markersize=8, label="B horizontal")
    axes[1].loglog([f["D_n"] for f in f_inc], [f["t_medio_myr"] for f in f_inc], "s", color="#D55E00",
                   markersize=8, label=r"B inclinado, $\varepsilon$ = 0.01, tan a = 0.1 y 0.3")
    dd = np.logspace(np.log10(min(Dn_todos)) - 0.2, np.log10(max(Dn_todos)) + 0.1, 50)
    axes[1].loglog(dd, a_myr(H ** 2 / (2 * dd)), color="#6b7280", linestyle="--", label=r"$H^2/(2 D_n)$")
    axes[1].set_xlabel(r"$D_n / D_\parallel$ (ec. 33)")
    axes[1].set_ylabel(r"$\langle t \rangle$ (Myr)")
    axes[1].set_title("Todas las corridas contra D_n")
    axes[1].legend(fontsize=9)
    viz.guardar(fig, RESULTADOS / "13_halo_tiempo_medio")

    fig, ax = plt.subplots(figsize=(7.5, 4.6), constrained_layout=True)
    for eps in eps_barrido:
        r = corridas[("eps", eps)]
        ax.plot(a_myr(r["t_mom"]), r["sx"], color=COLOR_EPS[eps], label=f"eps = {eps:g}")
    tt = np.linspace(0, T_base, 400)
    ax.plot(a_myr(tt), np.sqrt(sigma_x0 ** 2 + 2 * D_par * tt), color="#111827", linestyle="--",
            linewidth=1.2, label=r"$\sqrt{\sigma_{x0}^2 + 2 D_\parallel t}$ (sin paredes)")
    ax.axhline(2 * X_MAX / np.sqrt(12), color="#9ca3af", linestyle=":", label="uniforme en [-20, 20] (11.5 kpc)")
    ax.set_xlim(0, 2500)
    ax.set_ylim(0, 14)
    ax.set_xlabel("t (Myr)")
    ax.set_ylabel(r"$\sigma_x$ (kpc)")
    ax.set_title("Ancho a lo largo del campo")
    ax.legend(fontsize=9)
    viz.guardar(fig, RESULTADOS / "13_halo_ancho")

    # Serie temporal del caso base
    with open(RESULTADOS / "13_halo_series_base.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        masa_ex, p_ex = serie_z(D_normal(eps_base, 0.0), r0["t"])
        w.writerow(["t_myr", "masa_rel", "masa_rel_exacta", "p_por_myr", "p_exacta_por_myr",
                    "flujo_directo_rel"])
        for k in range(len(r0["t"])):
            w.writerow([a_myr(r0["t"][k]), r0["masa"][k] / r0["masa"][0], masa_ex[k],
                        r0["p"][k] / T0_MYR, p_ex[k] / T0_MYR, r0["flujo_directo"][k] / r0["masa"][0]])

    (RESULTADOS / "13_halo.txt").write_text("\n".join(lineas) + "\n", encoding="utf-8")
    print("Guardado en " + RESULTADOS.name + "/: 13_halo.txt, 13_halo_corridas.csv, 13_halo_series_base.csv"
          " y figuras 13_halo_mapas/masa/residencia/tiempo_medio/ancho")
    return 0


if __name__ == "__main__":
    sys.exit(main())
