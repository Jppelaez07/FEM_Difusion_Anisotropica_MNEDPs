"""Sensibilidad a la anisotropía y difusión perpendicular numérica.

Gaussiana inicial (sigma0 = 1), Q = 0, Neumann, B uniforme, D_par = 1, theta = 1, T = 4.
Los coeficientes efectivos salen de la pendiente de los momentos de segundo orden
en los ejes paralelo y perpendicular a B: d<sigma^2>/dt = 2 D. Los momentos se
integran con la cuadratura de 6 puntos y las mallas son uniformes, para que toda
la nube (incluidas sus colas) quede en elementos de tamaño h.

A. Barrido de D_par/D_perp en {1, 3, 10, 30, 100, 1000} con B a 30 grados:
   D_eff / D y razón de aspecto final frente a la teórica.
B. Difusión perpendicular numérica: error D_perp_eff / D_perp - 1 en 4 mallas,
   con B a 0 y 30 grados, para D_par/D_perp = 1000 y 100.
C. Difusión perpendicular numérica con campo curvo: B circular, anillo inicial
   axisimétrico, D_perp = 0.001 fijo. La solución exacta solo depende de D_perp
   (referencia 1D radial con diferencias finitas); toda dependencia de D_par es error.
Guarda CSV, figuras PNG/PDF y resúmenes en results/.
Uso: python scripts/12_sensibilidad.py [A] [B] [C]   (sin argumentos corre todo)
"""

import csv
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from src import assembly, fields, mesh, postproc, timestep, viz  # noqa: E402

# Parámetros comunes (unidades adimensionales)
Lx, Ly = 20.0, 20.0
xs, ys = Lx / 2.0, Ly / 2.0
D_par = 1.0
sigma0, n_max = 1.0, 1.0
T, dt, theta = 4.0, 0.04, 1.0
t_guardar = np.round(np.arange(0.0, T + 1e-9, 0.2), 10)
t_ajuste_min = 0.4                       # el ajuste lineal usa t >= t_ajuste_min

# A. Barrido
razones = [1, 3, 10, 30, 100, 1000]
alpha_barrido = 30.0
h_barrido = 0.1

# B. Difusión numérica
h_numerica = [0.4, 0.2, 0.1, 0.05]
razones_numerica = [1000, 100]
angulos_numerica = [0.0, 30.0]

# C. Campo curvo
D_perp_C = 0.001
razones_C = [1, 100, 1000]
r_anillo, w_anillo = 4.0, 0.5
T_C, dt_C = 20.0, 0.1
t_guardar_C = np.round(np.arange(0.0, T_C + 1e-9, 1.0), 10)
h_C = [0.4, 0.2, 0.1, 0.05]
R_ref, dr_ref, dt_ref = 10.0, 0.002, 0.005      # referencia 1D

RESULTADOS = RAIZ / "results"
COLOR_RAZON = {1000: "#D55E00", 100: "#0072B2", 1: "#009E73"}
LINEA_ANGULO = {0.0: "-", 30.0: "--"}


def malla(h):
    return mesh.malla_uniforme(Lx, Ly, h)


def h_medio_malla(nodes, elements):
    """h medio (lado más largo) de los elementos."""
    return mesh.calidad_malla(nodes, elements)["h_medio"]


def coeficientes_efectivos(nodes, elements, bnd, razon, alpha_grados):
    """Corre el caso y retorna D_par_eff, D_perp_eff y la razón de aspecto final."""
    a = np.radians(alpha_grados)
    D_func = fields.funcion_tensor(fields.campo_uniforme(a), D_par, D_par / razon)
    n0 = fields.condicion_inicial_gaussiana(n_max, xs, ys, sigma0)
    res = timestep.simular(nodes, elements, bnd, D_func, n0, T, dt, theta=theta, Q=None,
                           bc="neumann", t_guardar=t_guardar)
    mom = postproc.momentos_cuadratura(nodes, elements, res["N"])
    c, s = np.cos(a), np.sin(a)
    var_par = c * c * mom["xx"] + 2 * s * c * mom["xy"] + s * s * mom["yy"]
    var_perp = s * s * mom["xx"] - 2 * s * c * mom["xy"] + c * c * mom["yy"]
    sel = res["t"] >= t_ajuste_min
    D_par_eff = np.polyfit(res["t"][sel], var_par[sel], 1)[0] / 2.0
    D_perp_eff = np.polyfit(res["t"][sel], var_perp[sel], 1)[0] / 2.0
    aspecto = float(postproc.razon_aspecto(mom)[-1])
    return D_par_eff, D_perp_eff, aspecto


def guardar_csv(nombre, filas):
    with open(RESULTADOS / nombre, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(filas[0].keys()))
        w.writeheader()
        w.writerows(filas)


def parte_A(reportar):
    nodes, elements, bnd, _ = malla(h_barrido)
    hm = h_medio_malla(nodes, elements)
    filas = []
    for razon in razones:
        Dp_eff, Dt_eff, asp = coeficientes_efectivos(nodes, elements, bnd, razon, alpha_barrido)
        D_perp = D_par / razon
        teo = np.sqrt((sigma0 ** 2 + 2 * D_par * T) / (sigma0 ** 2 + 2 * D_perp * T))
        filas.append({"razon": razon, "D_perp": D_perp, "D_par_eff": Dp_eff,
                      "D_par_eff_sobre_D_par": Dp_eff / D_par, "D_perp_eff": Dt_eff,
                      "D_perp_eff_sobre_D_perp": Dt_eff / D_perp,
                      "aspecto_fem": asp, "aspecto_teorico": teo})

    reportar(f"A. BARRIDO DE ANISOTROPÍA (B a {alpha_barrido:g} grados, Neumann, T = {T:g})")
    reportar(f"   Malla uniforme: {len(nodes)} nodos, h medio = {hm:.3f}; ajuste lineal en t >= {t_ajuste_min:g}")
    reportar(f"   {'Dp/Dt':>6s} {'D_perp':>8s} {'D_par_eff':>10s} {'/D_par':>8s} {'D_perp_eff':>11s}"
             f" {'/D_perp':>8s} {'aspecto FEM':>11s} {'teórico':>8s} {'error':>7s}")
    for f in filas:
        reportar(f"   {f['razon']:6g} {f['D_perp']:8.4f} {f['D_par_eff']:10.5f} {f['D_par_eff_sobre_D_par']:8.4f}"
                 f" {f['D_perp_eff']:11.6f} {f['D_perp_eff_sobre_D_perp']:8.4f} {f['aspecto_fem']:11.4f}"
                 f" {f['aspecto_teorico']:8.4f} {abs(f['aspecto_fem'] / f['aspecto_teorico'] - 1):7.2%}")
    reportar("   aspecto teórico = sqrt((sigma0^2 + 2 D_par T) / (sigma0^2 + 2 D_perp T))")
    reportar()
    guardar_csv("12_barrido_anisotropia.csv", filas)

    r = np.array([f["razon"] for f in filas], dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    ax = axes[0]
    ax.semilogx(r, [f["D_par_eff_sobre_D_par"] for f in filas], "o-", color="#0072B2",
                markersize=7, label=r"$D_{\parallel,ef} / D_\parallel$")
    ax.semilogx(r, [f["D_perp_eff_sobre_D_perp"] for f in filas], "s-", color="#D55E00",
                markersize=7, label=r"$D_{\perp,ef} / D_\perp$")
    ax.axhline(1.0, color="#6b7280", linestyle=":", linewidth=1.2)
    ax.set_xlabel(r"$D_\parallel / D_\perp$")
    ax.set_ylabel("Coeficiente efectivo / nominal")
    ax.set_title(f"Coeficientes efectivos (h medio = {hm:.2f})")
    ax.legend()
    ax = axes[1]
    ax.loglog(r, [f["aspecto_teorico"] for f in filas], "-", color="#6b7280", linewidth=1.5,
              label="Teórica")
    ax.loglog(r, [f["aspecto_fem"] for f in filas], "o", color="#0072B2", markersize=8,
              label="FEM (momentos)")
    ax.set_xlabel(r"$D_\parallel / D_\perp$")
    ax.set_ylabel(f"Razón de aspecto en t = {T:g}")
    ax.set_title("Razón de aspecto final de la nube")
    ax.legend()
    viz.guardar(fig, RESULTADOS / "12_barrido_anisotropia")


def parte_B(reportar):
    mallas = []
    for h in h_numerica:
        t0 = time.time()
        nodes, elements, bnd, _ = malla(h)
        mallas.append((h, nodes, elements, bnd, h_medio_malla(nodes, elements)))
        print(f"   malla h = {h:g}: {len(nodes)} nodos ({time.time() - t0:.1f} s)", flush=True)

    filas = []
    for razon in razones_numerica:
        for ang in angulos_numerica:
            for h, nodes, elements, bnd, hm in mallas:
                t0 = time.time()
                Dp_eff, Dt_eff, _ = coeficientes_efectivos(nodes, elements, bnd, razon, ang)
                filas.append({"razon": razon, "angulo": ang, "h": h, "h_medio": hm,
                              "nodos": len(nodes), "error_D_perp": Dt_eff / (D_par / razon) - 1.0,
                              "error_D_par": Dp_eff / D_par - 1.0})
                print(f"   razón {razon}, {ang:g} grados, h = {h:g}: {time.time() - t0:.1f} s", flush=True)

    reportar("B. DIFUSIÓN PERPENDICULAR NUMÉRICA (error relativo D_eff / D - 1)")
    reportar(f"   {'Dp/Dt':>6s} {'ángulo':>6s} {'h':>5s} {'h medio':>8s} {'nodos':>7s}"
             f" {'error D_perp':>13s} {'tasa':>6s} {'error D_par':>12s}")
    for razon in razones_numerica:
        for ang in angulos_numerica:
            sel = [f for f in filas if f["razon"] == razon and f["angulo"] == ang]
            e = np.abs([f["error_D_perp"] for f in sel])
            hm = np.array([f["h_medio"] for f in sel])
            tasas = np.concatenate(([np.nan], np.log(e[:-1] / e[1:]) / np.log(hm[:-1] / hm[1:])))
            for f, tasa in zip(sel, tasas):
                f["tasa"] = tasa
                reportar(f"   {razon:6g} {ang:6g} {f['h']:5.2f} {f['h_medio']:8.4f} {f['nodos']:7d}"
                         f" {f['error_D_perp']:13.4e} {('     -' if np.isnan(tasa) else f'{tasa:6.2f}')}"
                         f" {f['error_D_par']:12.4e}")
            reportar()
    guardar_csv("12_difusion_numerica.csv", filas)

    fig, ax = plt.subplots(figsize=(6.8, 5.0), constrained_layout=True)
    for razon in razones_numerica:
        for ang in angulos_numerica:
            sel = [f for f in filas if f["razon"] == razon and f["angulo"] == ang]
            ax.loglog([f["h_medio"] for f in sel], np.abs([f["error_D_perp"] for f in sel]),
                      marker="o", linestyle=LINEA_ANGULO[ang], color=COLOR_RAZON[razon],
                      markersize=7, label=rf"$D_\parallel/D_\perp$ = {razon}, B a {ang:g}°")
    hm = np.array([m[4] for m in mallas])
    ref = max(abs(f["error_D_perp"]) for f in filas if f["h"] == h_numerica[-1])
    ax.loglog(hm, 0.5 * ref * (hm / hm[-1]) ** 2, color="#6b7280", linestyle=":", linewidth=1.2,
              label="pendiente 2")
    ax.set_xticks(hm)
    ax.set_xticklabels([f"{v:.3f}" for v in hm])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("h medio de la malla uniforme (adimensional)")
    ax.set_ylabel(r"$|D_{\perp,ef} / D_\perp - 1|$")
    ax.set_title("Difusión perpendicular numérica")
    ax.legend(fontsize=8)
    viz.guardar(fig, RESULTADOS / "12_difusion_numerica")


def anillo(r):
    return np.exp(-(r - r_anillo) ** 2 / (2.0 * w_anillo ** 2))


def referencia_radial(D, tiempos, R=R_ref, dr=dr_ref, dt_r=dt_ref):
    """Solución de dn/dt = (1/r) d/dr (r D dn/dr) con volúmenes finitos y Crank-Nicolson.

    Flujo nulo en r = 0 y r = R. Retorna (r de los centros de celda, n en cada instante).
    """
    from scipy.sparse import diags
    from scipy.sparse.linalg import splu
    nc = int(round(R / dr))
    r = (np.arange(nc) + 0.5) * dr
    caras = np.arange(1, nc) * dr                          # caras interiores
    c = D * caras / dr / (r[:-1] * dr)                     # acoplamiento i -> i+1
    d = D * caras / dr / (r[1:] * dr)                      # acoplamiento i+1 -> i
    diag = np.zeros(nc)
    diag[:-1] -= c
    diag[1:] -= d
    L = diags([diag, c, d], [0, 1, -1], format="csc")
    I = diags([np.ones(nc)], [0], format="csc")
    lu = splu((I - 0.5 * dt_r * L).tocsc())
    B = (I + 0.5 * dt_r * L).tocsr()

    n = anillo(r)
    pasos = np.rint(np.asarray(tiempos) / dt_r).astype(int)
    salida = np.empty((len(pasos), nc))
    k = 0
    for j, p in enumerate(pasos):
        while k < p:
            n = lu.solve(B @ n)
            k += 1
        salida[j] = n
    return r, salida


def varianza_radial_ref(r, n):
    w = n * r                                              # medida 2 pi r dr (2 pi se cancela)
    rm = (w * r).sum(axis=-1) / w.sum(axis=-1)
    return (w * (r - rm[..., None]) ** 2).sum(axis=-1) / w.sum(axis=-1)


def varianza_radial_fem(nodes, elements, N):
    """<(r - r_medio)^2> de la solución FEM, con la cuadratura de 6 puntos."""
    detJ, _, _ = assembly.geometria(nodes, elements)
    xq = np.einsum("qm,emi->eqi", postproc.CUAD_BARI, nodes[elements])
    rq = np.hypot(xq[..., 0] - xs, xq[..., 1] - ys)
    pesos = (detJ / 2.0)[:, None] * postproc.CUAD_PESOS
    var = np.empty(len(N))
    for s, Ns in enumerate(N):
        w = pesos * (Ns[elements] @ postproc.CUAD_BARI.T)
        rm = (w * rq).sum() / w.sum()
        var[s] = (w * (rq - rm) ** 2).sum() / w.sum()
    return var


def parte_C(reportar):
    t0 = time.time()
    r, n_ref = referencia_radial(D_perp_C, t_guardar_C)
    r2, n_ref2 = referencia_radial(D_perp_C, t_guardar_C, dr=dr_ref / 2, dt_r=dt_ref / 2)
    var_ref = varianza_radial_ref(r, n_ref)
    var_ref2 = varianza_radial_ref(r2, n_ref2)
    crec_ref = var_ref[-1] - var_ref[0]
    print(f"   referencia 1D: {time.time() - t0:.1f} s", flush=True)

    reportar("C. DIFUSIÓN PERPENDICULAR NUMÉRICA CON CAMPO CURVO (B circular, Neumann, Q = 0)")
    reportar(f"   n0 = exp(-(r - {r_anillo:g})^2 / (2 {w_anillo:g}^2)), D_perp = {D_perp_C:g},"
             f" T = {T_C:g}, theta = 1, dt = {dt_C:g}")
    reportar(f"   Referencia 1D radial: dr = {dr_ref:g}, dt = {dt_ref:g} (Crank-Nicolson).")
    reportar(f"   Crecimiento de la varianza radial de referencia: {crec_ref:.6e}"
             f" (2 D_perp T = {2 * D_perp_C * T_C:.6e})")
    reportar(f"   Con dr y dt a la mitad cambia en {abs((var_ref2[-1] - var_ref2[0]) - crec_ref):.1e};"
             f" error L2 entre ambas referencias en t = T:"
             f" {np.sqrt(np.trapezoid((np.interp(r2, r, n_ref[-1]) - n_ref2[-1]) ** 2 * r2, r2) / np.trapezoid(n_ref2[-1] ** 2 * r2, r2)):.1e}")
    reportar("   D_perp_num = (1/2) pendiente de [var_FEM(t) - var_ref(t)] (ajuste lineal en todo t).")
    reportar()

    filas = []
    for h in h_C:
        t1 = time.time()
        nodes, elements, bnd, _ = malla(h)
        hm = h_medio_malla(nodes, elements)
        print(f"   malla h = {h:g}: {len(nodes)} nodos ({time.time() - t1:.1f} s)", flush=True)
        n0 = lambda x, y: anillo(np.hypot(x - xs, y - ys))
        n_T = lambda x, y: np.interp(np.hypot(x - xs, y - ys), r, n_ref[-1])
        for razon in razones_C:
            t1 = time.time()
            D_func = fields.funcion_tensor(fields.campo_circular(xs, ys), razon * D_perp_C, D_perp_C)
            res = timestep.simular(nodes, elements, bnd, D_func, n0, T_C, dt_C, theta=1.0, Q=None,
                                   bc="neumann", t_guardar=t_guardar_C)
            var = varianza_radial_fem(nodes, elements, res["N"])
            D_num = np.polyfit(res["t"], var - var_ref, 1)[0] / 2.0
            err = postproc.error_L2_relativo(nodes, elements, res["N"][-1], n_T)
            filas.append({"razon": razon, "D_par": razon * D_perp_C, "h": h, "h_medio": hm,
                          "nodos": len(nodes), "D_perp_num": D_num,
                          "D_perp_num_sobre_D_perp": D_num / D_perp_C,
                          "D_perp_num_sobre_D_par": D_num / (razon * D_perp_C),
                          "error_L2_rel": err, "crecimiento_var_fem": var[-1] - var[0]})
            print(f"   razón {razon}, h = {h:g}: {time.time() - t1:.1f} s", flush=True)

    reportar(f"   {'Dp/Dt':>6s} {'h':>5s} {'h medio':>8s} {'nodos':>7s} {'Dnum/D_perp':>12s}"
             f" {'Dnum/D_par':>11s} {'tasa':>6s} {'error L2':>10s} {'tasa':>6s}")
    for razon in razones_C:
        sel = [f for f in filas if f["razon"] == razon]
        hm = np.array([f["h_medio"] for f in sel])
        tasa = lambda v: np.concatenate(([np.nan], np.log(v[:-1] / v[1:]) / np.log(hm[:-1] / hm[1:])))
        tD = tasa(np.abs([f["D_perp_num"] for f in sel]))
        tE = tasa(np.array([f["error_L2_rel"] for f in sel]))
        for f, a, b in zip(sel, tD, tE):
            f["tasa_D_perp_num"], f["tasa_error_L2"] = a, b
            fa = "     -" if np.isnan(a) else f"{a:6.2f}"
            fb = "     -" if np.isnan(b) else f"{b:6.2f}"
            reportar(f"   {razon:6g} {f['h']:5.2f} {f['h_medio']:8.4f} {f['nodos']:7d}"
                     f" {f['D_perp_num_sobre_D_perp']:12.4e} {f['D_perp_num_sobre_D_par']:11.4e} {fa}"
                     f" {f['error_L2_rel']:10.3e} {fb}")
        reportar()
    guardar_csv("12_campo_curvo.csv", filas)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), constrained_layout=True)
    for razon in razones_C:
        sel = [f for f in filas if f["razon"] == razon]
        hm = [f["h_medio"] for f in sel]
        estilo = dict(color=COLOR_RAZON[razon], marker="o", markersize=7,
                      label=rf"$D_\parallel/D_\perp$ = {razon}")
        axes[0].loglog(hm, np.abs([f["D_perp_num_sobre_D_perp"] for f in sel]), **estilo)
        axes[1].loglog(hm, [f["error_L2_rel"] for f in sel], **estilo)
    hm = np.array(sorted({f["h_medio"] for f in filas}, reverse=True))
    for ax, clave in ((axes[0], "D_perp_num_sobre_D_perp"), (axes[1], "error_L2_rel")):
        ancla = max(abs(f[clave]) for f in filas if f["h"] == h_C[-1])
        ax.loglog(hm, 0.5 * ancla * (hm / hm[-1]) ** 2, color="#6b7280", linestyle=":",
                  linewidth=1.2, label="pendiente 2")
        ax.set_xticks(hm)
        ax.set_xticklabels([f"{v:.3f}" for v in hm])
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_xlabel("h medio de la malla uniforme (adimensional)")
        ax.legend(fontsize=9)
    axes[0].set_ylabel(r"$|D_{\perp,num}| / D_\perp$")
    axes[0].set_title("Difusión perpendicular numérica (B circular)")
    axes[1].set_ylabel(f"Error L2 relativo en t = {T_C:g}")
    axes[1].set_title("Error frente a la referencia radial 1D")
    viz.guardar(fig, RESULTADOS / "12_campo_curvo")


def main():
    RESULTADOS.mkdir(exist_ok=True)
    viz.estilo()
    partes = [p.upper() for p in sys.argv[1:]] or ["A", "B", "C"]
    funciones = {"A": parte_A, "B": parte_B, "C": parte_C}
    salidas = {"A": "12_sensibilidad.txt", "B": "12_sensibilidad.txt", "C": "12_campo_curvo.txt"}
    textos = {}
    for p in partes:
        lineas = textos.setdefault(salidas[p], [])

        def reportar(texto="", lineas=lineas):
            print(texto, flush=True)
            lineas.append(texto)

        funciones[p](reportar)
    for nombre, lineas in textos.items():
        (RESULTADOS / nombre).write_text("\n".join(lineas) + "\n", encoding="utf-8")
    print("Guardado en results/:", ", ".join(sorted(textos)), "y figuras/CSV de las partes", partes)
    return 0


if __name__ == "__main__":
    sys.exit(main())
