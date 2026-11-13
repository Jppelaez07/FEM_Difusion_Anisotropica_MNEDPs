"""Verificación con solución manufacturada en [0,1]^2 con Dirichlet homogénea.

Solución exacta: n = exp(-t) sin(pi x) sin(pi y), D_par = 1, D_perp = 0.1, B a 30 grados.
1. Verifica con sympy la fuente Q.
2. Convergencia espacial: errores L2 y H1 en mallas uniformes (esperado h^2 y h).
3. Convergencia temporal: error contra una referencia con dt muy pequeño
   (esperado dt para theta = 1 y dt^2 para theta = 0.5), con dos condiciones
   iniciales: interpolación de n0 y proyección de Ritz de n0.
Guarda tablas CSV y figuras PNG/PDF en results/.
"""

import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import sympy as sp
from scipy.sparse.linalg import spsolve

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from src import assembly, fields, mesh, postproc, timestep, rutas  # noqa: E402

# Parámetros
D_par, D_perp = 1.0, 0.1
alpha = np.radians(30.0)

h_espacial = [0.2, 0.1, 0.05, 0.025, 0.0125]
theta_esp, dt_esp, T_esp = 0.5, 1e-3, 0.5

h_temporal = 0.0125
T_tem = 1.0
dt_temporal = [0.1, 0.05, 0.025, 0.0125, 0.00625]
thetas = [1.0, 0.5]
theta_ref, dt_ref = 0.5, 1e-4

RESULTADOS = rutas.RESULTADOS
COLOR = {1.0: "#1d4ed8", 0.5: "#c2410c"}
COLOR_REF = "#6b7280"

# Tensor D (constante) y sus componentes
D0 = fields.tensor_difusion(np.cos(alpha), np.sin(alpha), D_par, D_perp)[0]
Dxx, Dyy, Dxy = D0[0, 0], D0[1, 1], D0[0, 1]
D_func = fields.funcion_tensor(fields.campo_uniforme(alpha), D_par, D_perp)


def n_exacta(x, y, t):
    return np.exp(-t) * np.sin(np.pi * x) * np.sin(np.pi * y)


def grad_exacta(x, y, t):
    e = np.exp(-t) * np.pi
    return (e * np.cos(np.pi * x) * np.sin(np.pi * y),
            e * np.sin(np.pi * x) * np.cos(np.pi * y))


def menos_div_D_grad(x, y, t):
    """-div(D grad n) de la solución exacta."""
    pi2 = np.pi ** 2
    return np.exp(-t) * (pi2 * (Dxx + Dyy) * np.sin(np.pi * x) * np.sin(np.pi * y)
                         - 2.0 * pi2 * Dxy * np.cos(np.pi * x) * np.cos(np.pi * y))


def Q(x, y, t):
    pi2 = np.pi ** 2
    return np.exp(-t) * ((pi2 * (Dxx + Dyy) - 1.0) * np.sin(np.pi * x) * np.sin(np.pi * y)
                         - 2.0 * pi2 * Dxy * np.cos(np.pi * x) * np.cos(np.pi * y))


def verificar_fuente(reportar):
    """Comprueba Q = dn/dt - div(D grad n) simbólicamente y numéricamente."""
    x, y, t = sp.symbols("x y t", real=True)
    dxx, dyy, dxy = sp.symbols("Dxx Dyy Dxy", real=True)
    n = sp.exp(-t) * sp.sin(sp.pi * x) * sp.sin(sp.pi * y)
    flujo_x = dxx * sp.diff(n, x) + dxy * sp.diff(n, y)
    flujo_y = dxy * sp.diff(n, x) + dyy * sp.diff(n, y)
    residuo = sp.diff(n, t) - (sp.diff(flujo_x, x) + sp.diff(flujo_y, y))
    Q_dada = sp.exp(-t) * ((sp.pi ** 2 * (dxx + dyy) - 1) * sp.sin(sp.pi * x) * sp.sin(sp.pi * y)
                           - 2 * sp.pi ** 2 * dxy * sp.cos(sp.pi * x) * sp.cos(sp.pi * y))
    diferencia = sp.simplify(residuo - Q_dada)
    ok_simb = diferencia == 0

    # Comparación numérica con las funciones que usa el solver
    num = {dxx: Dxx, dyy: Dyy, dxy: Dxy}
    Q_sym = sp.lambdify((x, y, t), residuo.subs(num), "numpy")
    L_sym = sp.lambdify((x, y, t), (-(sp.diff(flujo_x, x) + sp.diff(flujo_y, y))).subs(num), "numpy")
    rng = np.random.default_rng(0)
    xp, yp, tp = rng.random(1000), rng.random(1000), rng.random(1000)
    dif_Q = np.max(np.abs(Q_sym(xp, yp, tp) - Q(xp, yp, tp)))
    dif_L = np.max(np.abs(L_sym(xp, yp, tp) - menos_div_D_grad(xp, yp, tp)))
    ok_num = max(dif_Q, dif_L) < 1e-12

    reportar("1. VERIFICACIÓN DE LA FUENTE (sympy)")
    reportar(f"   dn/dt - div(D grad n) - Q simplificado: {diferencia}"
             f"  ({'OK' if ok_simb else 'FALLA'})")
    reportar(f"   max |Q_sympy - Q_numpy| en 1000 puntos: {dif_Q:.2e}"
             f"  ({'OK' if dif_Q < 1e-12 else 'FALLA'})")
    reportar(f"   D = [[{Dxx:.6f}, {Dxy:.6f}], [{Dxy:.6f}, {Dyy:.6f}]]")
    reportar()
    return ok_simb and ok_num


def tasas(x, e):
    """Tasa observada log(e_i/e_{i+1}) / log(x_i/x_{i+1}) entre valores consecutivos."""
    x, e = np.asarray(x), np.asarray(e)
    return np.concatenate(([np.nan], np.log(e[:-1] / e[1:]) / np.log(x[:-1] / x[1:])))


def guardar_csv(nombre, encabezado, filas):
    with open(RESULTADOS / nombre, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(encabezado)
        for fila in filas:
            w.writerow(["" if (isinstance(v, float) and np.isnan(v)) else v for v in fila])


def fmt_tasa(v):
    return "    -" if np.isnan(v) else f"{v:5.2f}"


def linea_referencia(ax, x, y_ancla, pendiente, texto):
    """Recta de pendiente dada que pasa por (x[-1], y_ancla)."""
    x = np.asarray(x)
    y = y_ancla * (x / x[-1]) ** pendiente
    ax.loglog(x, y, color=COLOR_REF, linestyle="--", linewidth=1.0)
    k = len(x) // 2
    ax.annotate(texto, (x[k], y[k]), textcoords="offset points", xytext=(6, -12),
                fontsize=9, color=COLOR_REF)


def marcas_eje_x(ax, valores, formato):
    """Marcas del eje x en los valores usados, sin las marcas menores."""
    ax.set_xticks(valores)
    ax.set_xticklabels([formato.format(v) for v in valores])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())


def guardar_figura(fig, nombre):
    for ext in ("png", "pdf"):
        fig.savefig(RESULTADOS / f"{nombre}.{ext}", dpi=200)
    plt.close(fig)


def convergencia_espacial(reportar):
    filas = []
    for h in h_espacial:
        nodes, elements, bnd, _ = mesh.malla_uniforme(1.0, 1.0, h)
        q = mesh.calidad_malla(nodes, elements)
        res = timestep.simular(nodes, elements, bnd, D_func, lambda x, y: n_exacta(x, y, 0.0),
                               T_esp, dt_esp, theta=theta_esp, Q=Q, bc="dirichlet")
        eL2, eH1 = postproc.errores_L2_H1(nodes, elements, res["N"][-1],
                                          lambda x, y: n_exacta(x, y, T_esp),
                                          lambda x, y: grad_exacta(x, y, T_esp))
        filas.append({"h": h, "h_max": q["h_max"], "h_medio": q["h_medio"],
                      "nodos": q["n_nodos"], "elementos": q["n_elementos"],
                      "L2": eL2, "H1": eH1})

    col = {k: np.array([f[k] for f in filas]) for k in filas[0]}
    tL2, tH1 = tasas(col["h_max"], col["L2"]), tasas(col["h_max"], col["H1"])
    tL2m, tH1m = tasas(col["h_medio"], col["L2"]), tasas(col["h_medio"], col["H1"])

    reportar(f"2. CONVERGENCIA ESPACIAL (theta = {theta_esp:g}, dt = {dt_esp:g}, T = {T_esp:g})")
    reportar("   Tasas calculadas con h max (diámetro máximo) y, entre paréntesis, con h medio.")
    reportar(f"   {'h max':>8s} {'h medio':>8s} {'nodos':>6s} {'elem.':>6s}"
             f" {'error L2':>11s} {'tasa':>5s} {'(medio)':>7s} {'error H1':>11s} {'tasa':>5s} {'(medio)':>7s}")
    for k, f in enumerate(filas):
        reportar(f"   {f['h_max']:8.5f} {f['h_medio']:8.5f} {f['nodos']:6d} {f['elementos']:6d}"
                 f" {f['L2']:11.4e} {fmt_tasa(tL2[k])} ({fmt_tasa(tL2m[k])})"
                 f" {f['H1']:11.4e} {fmt_tasa(tH1[k])} ({fmt_tasa(tH1m[k])})")
    reportar()

    guardar_csv("03_mms_espacial.csv",
                ["h_objetivo", "h_max", "h_medio", "nodos", "elementos",
                 "error_L2", "tasa_L2", "tasa_L2_h_medio", "error_H1", "tasa_H1", "tasa_H1_h_medio"],
                [[f["h"], f["h_max"], f["h_medio"], f["nodos"], f["elementos"],
                  f["L2"], tL2[k], tL2m[k], f["H1"], tH1[k], tH1m[k]] for k, f in enumerate(filas)])

    fig, ax = plt.subplots(figsize=(6.4, 5.0), constrained_layout=True)
    # La figura usa h medio (menos sensible a un solo elemento extremo); la tabla y el CSV
    # conservan las tasas con h máximo y con h medio.
    ax.loglog(col["h_medio"], col["L2"], "o-", color=COLOR[1.0], linewidth=2, markersize=7,
              label="Error en norma L2")
    ax.loglog(col["h_medio"], col["H1"], "s-", color=COLOR[0.5], linewidth=2, markersize=7,
              label="Error en seminorma H1")
    linea_referencia(ax, col["h_medio"], col["L2"][-1] * 0.5, 2, "pendiente 2")
    linea_referencia(ax, col["h_medio"], col["H1"][-1] * 0.5, 1, "pendiente 1")
    marcas_eje_x(ax, col["h_medio"], "{:.3f}")
    ax.set_xlabel("h medio de los elementos (adimensional)")
    ax.set_ylabel("Error (adimensional)")
    ax.set_title(f"Convergencia espacial (t = {T_esp:g})")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    guardar_figura(fig, "03_mms_espacial")
    return tL2, tH1, tL2m, tH1m


def convergencia_temporal(reportar):
    nodes, elements, bnd, inte = mesh.malla_uniforme(1.0, 1.0, h_temporal)
    geom = assembly.geometria(nodes, elements)
    M, K = assembly.ensamblar_MK(nodes, elements, D_func, geom=geom)
    Mi, Ki = assembly.restringir(M, inte), assembly.restringir(K, inte)

    def F(t):
        return assembly.ensamblar_F(nodes, elements, Q, t, geom=geom)[inte]

    # Carga integrada con la cuadratura de 6 puntos: b_i = int g phi_i
    p = nodes[elements]
    xq = np.einsum("qm,emi->eqi", postproc.CUAD_BARI, p)
    g = menos_div_D_grad(xq[..., 0], xq[..., 1], 0.0) * postproc.CUAD_PESOS
    b_loc = (geom[0] / 2.0)[:, None] * (g @ postproc.CUAD_BARI)
    b = np.bincount(elements.ravel(), weights=b_loc.ravel(), minlength=len(nodes))[inte]

    iniciales = {
        "interpolación": n_exacta(nodes[inte, 0], nodes[inte, 1], 0.0),
        "Ritz": spsolve(Ki.tocsc(), b),          # K N0 = b
    }

    errores = {}
    for nombre, N0 in iniciales.items():
        _, ref = timestep.metodo_theta(Mi, Ki, N0, T_tem, dt_ref, theta=theta_ref, F=F)
        for th in thetas:
            errores[(nombre, th)] = np.array([
                postproc.norma_L2_discreta(
                    Mi, timestep.metodo_theta(Mi, Ki, N0, T_tem, dt, theta=th, F=F)[1][-1] - ref[-1])
                for dt in dt_temporal])
    t_tasas = {k: tasas(dt_temporal, e) for k, e in errores.items()}
    claves = [(c, th) for c in iniciales for th in thetas]

    reportar(f"3. CONVERGENCIA TEMPORAL (malla h = {h_temporal:g}, {len(nodes)} nodos, T = {T_tem:g})")
    reportar(f"   Referencia: theta = {theta_ref:g}, dt = {dt_ref:g}, misma malla y misma N0")
    reportar("   Error en norma L2 en t = T. N0: interpolación de n0 o proyección de Ritz de n0.")
    encab = "".join(f" {c[:6] + ' th=' + f'{th:g}':>15s} {'tasa':>5s}" for c, th in claves)
    reportar(f"   {'dt':>8s}{encab}")
    for k, dt in enumerate(dt_temporal):
        fila = "".join(f" {errores[c][k]:15.4e} {fmt_tasa(t_tasas[c][k])}" for c in claves)
        reportar(f"   {dt:8.5f}{fila}")
    reportar()

    nombres_csv = {"interpolación": "interp", "Ritz": "ritz"}
    encabezado = ["dt"] + [f"{s}_theta_{th:g}_{nombres_csv[c]}"
                           for c, th in claves for s in ("error", "tasa")]
    guardar_csv("03_mms_temporal.csv", encabezado,
                [[dt] + [v for c in claves for v in (errores[c][k], t_tasas[c][k])]
                 for k, dt in enumerate(dt_temporal)])

    fig, ax = plt.subplots(figsize=(6.4, 5.0), constrained_layout=True)
    estilos = {"interpolación": dict(linestyle="-", marker="o", markerfacecolor=None),
               "Ritz": dict(linestyle=":", marker="s", markerfacecolor="white")}
    nombres = {1.0: "Euler implícito", 0.5: "Crank-Nicolson"}
    for c, th in claves:
        e = estilos[c]
        ax.loglog(dt_temporal, errores[(c, th)], color=COLOR[th], linewidth=2, markersize=7,
                  linestyle=e["linestyle"], marker=e["marker"],
                  markerfacecolor=e["markerfacecolor"] or COLOR[th],
                  label=rf"$\theta$ = {th:g} ({nombres[th]}), N0 por {c}")
    linea_referencia(ax, dt_temporal, errores[("interpolación", 1.0)][-1] * 0.5, 1, "pendiente 1")
    linea_referencia(ax, dt_temporal, errores[("Ritz", 0.5)][-1] * 0.5, 2, "pendiente 2")
    marcas_eje_x(ax, dt_temporal, "{:g}")
    ax.set_xlabel(r"$\Delta t$ (adimensional)")
    ax.set_ylabel("Error L2 en t = 1 frente a la referencia (adimensional)")
    ax.set_title(f"Convergencia temporal (h = {h_temporal:g})")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8)
    guardar_figura(fig, "03_mms_temporal")
    return t_tasas


def main():
    RESULTADOS.mkdir(exist_ok=True)
    lineas = []

    def reportar(texto=""):
        print(texto, flush=True)
        lineas.append(texto)

    if not verificar_fuente(reportar):
        reportar("La fuente Q no coincide con la EDP; se detiene.")
        return 1
    tL2, tH1, tL2m, tH1m = convergencia_espacial(reportar)
    t_tasas = convergencia_temporal(reportar)

    reportar("RESUMEN (tasa entre las dos últimas mallas o pasos)")
    reportar(f"   Espacial, h max:   L2 {tL2[-1]:.2f} (esperado 2), H1 {tH1[-1]:.2f} (esperado 1)")
    reportar(f"   Espacial, h medio: L2 {tL2m[-1]:.2f} (esperado 2), H1 {tH1m[-1]:.2f} (esperado 1)")
    for c in ("interpolación", "Ritz"):
        reportar(f"   Temporal, N0 por {c}: theta = 1 {t_tasas[(c, 1.0)][-1]:.2f} (esperado 1),"
                 f" theta = 0.5 {t_tasas[(c, 0.5)][-1]:.2f} (esperado 2)")

    (RESULTADOS / "03_mms.txt").write_text("\n".join(lineas) + "\n", encoding="utf-8")
    print("\nGuardado en results/: 03_mms_espacial.csv/png/pdf, "
          "03_mms_temporal.csv/png/pdf, 03_mms.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
