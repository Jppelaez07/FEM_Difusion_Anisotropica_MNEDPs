"""Estabilidad del esquema explícito y oscilaciones de Crank-Nicolson.

A. Para 3 mallas: lambda_max de K v = lambda M v, dt crítico dt_c = 2 / lambda_max
   del esquema explícito (theta = 0), corridas con 0.9 dt_c y 1.1 dt_c, y
   verificación de dt_c proporcional a h^2.
B. Gaussiana estrecha (sigma0 = 2h) con dt grande: theta = 0.5 vs theta = 1.
   Mínimo de n en el tiempo y cortes de la solución.
Dominio [0,20]^2, Dirichlet, Q = 0, D_par = 1, D_perp = 0.1, B a 30 grados.
Guarda figuras PNG/PDF, CSV y un resumen en results/.
"""

import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
from scipy.sparse.linalg import eigsh

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from src import assembly, fields, mesh, postproc, timestep  # noqa: E402

# Parámetros comunes (unidades adimensionales)
Lx, Ly = 20.0, 20.0
xs, ys = Lx / 2.0, Ly / 2.0
D_par, D_perp = 1.0, 0.1
alpha = np.radians(30.0)

# Parte A
h_estab = [1.0, 0.5, 0.25]
factores = [0.9, 1.1]
n_pasos = 400
sigma0_A = 1.0

# Parte B: malla refinada en la fuente; sigma0 = 2h con sigma0^2 << D_par dt
h_osc, h_global_osc, r_ref_osc = 0.05, 0.5, 2.0
sigma0_B = 2.0 * h_osc
dt_osc, T_osc = 0.5, 5.0
t_cortes = [0.5, 1.0]

RESULTADOS = RAIZ / "results"
COLOR = {0.9: "#1d4ed8", 1.1: "#c2410c", 1.0: "#1d4ed8", 0.5: "#c2410c"}

D_func = fields.funcion_tensor(fields.campo_uniforme(alpha), D_par, D_perp)
D0 = D_func(np.array([xs]), np.array([ys]))[0]


def matrices_interiores(nodes, elements, bnd):
    M, K = assembly.ensamblar_MK(nodes, elements, D_func)
    inte = np.setdiff1d(np.arange(len(nodes)), bnd)
    return M, assembly.restringir(M, inte), assembly.restringir(K, inte), inte


def guardar_figura(fig, nombre):
    for ext in ("png", "pdf"):
        fig.savefig(RESULTADOS / f"{nombre}.{ext}", dpi=200)
    plt.close(fig)


def parte_A(reportar):
    n0 = fields.condicion_inicial_gaussiana(1.0, xs, ys, sigma0_A)
    filas, series = [], []
    for h in h_estab:
        nodes, elements, bnd, _ = mesh.malla_uniforme(Lx, Ly, h)
        q = mesh.calidad_malla(nodes, elements)
        _, Mi, Ki, inte = matrices_interiores(nodes, elements, bnd)
        lam_max = eigsh(Ki, k=1, M=Mi, which="LA", return_eigenvectors=False)[0]
        dt_c = 2.0 / lam_max
        N0 = n0(nodes[inte, 0], nodes[inte, 1])
        norma0 = postproc.norma_L2_discreta(Mi, N0)

        for f in factores:
            dt = f * dt_c
            with np.errstate(over="ignore", invalid="ignore"):
                t, S = timestep.metodo_theta(Mi, Ki, N0, n_pasos * dt, dt, theta=0.0,
                                             t_guardar=np.arange(n_pasos + 1) * dt)
                normas = np.sqrt(np.einsum("si,si->s", S, (Mi @ S.T).T)) / norma0
            series.append({"h": h, "f": f, "t": t, "norma": normas, "dt_c": dt_c})
            filas.append([h, q["h_medio"], q["h_max"], len(inte), lam_max, dt_c, f, dt,
                          normas[-1]])

    reportar("A. ESTABILIDAD DEL ESQUEMA EXPLÍCITO (theta = 0)")
    reportar(f"   {n_pasos} pasos; norma = sqrt(N^T M N) relativa a la inicial")
    reportar(f"   {'h':>5s} {'h medio':>8s} {'libres':>7s} {'lambda_max':>11s} {'dt_c':>10s}"
             f" {'dt_c/h_medio^2':>15s} {'norma final 0.9dt_c':>20s} {'norma final 1.1dt_c':>20s}")
    for k in range(0, len(filas), 2):
        a, b = filas[k], filas[k + 1]
        reportar(f"   {a[0]:5.2f} {a[1]:8.4f} {a[3]:7d} {a[4]:11.4e} {a[5]:10.4e}"
                 f" {a[5] / a[1] ** 2:15.4f} {a[8]:20.3e} {b[8]:20.3e}")
    h_med = np.array(filas[::2])[:, 1]
    h_mx = np.array(filas[::2])[:, 2]
    dtc = np.array(filas[::2])[:, 5]
    pend_med = np.polyfit(np.log(h_med), np.log(dtc), 1)[0]
    pend_max = np.polyfit(np.log(h_mx), np.log(dtc), 1)[0]
    reportar(f"   Pendiente de log(dt_c) vs log(h): {pend_med:.2f} con h medio,"
             f" {pend_max:.2f} con h max (esperado 2)")
    estables = all(f[8] < 1.0 for f in filas if f[6] == 0.9)
    explotan = all(f[8] > 1e3 for f in filas if f[6] == 1.1)
    reportar(f"   0.9 dt_c estable en las 3 mallas: {'sí' if estables else 'NO'};"
             f" 1.1 dt_c explota en las 3 mallas: {'sí' if explotan else 'NO'}")
    reportar()

    with open(RESULTADOS / "05_estabilidad_dtc.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["h_objetivo", "h_medio", "h_max", "nodos_libres", "lambda_max", "dt_c",
                    "factor", "dt", "norma_final_relativa"])
        w.writerows(filas)

    # Figura: norma contra el tiempo
    fig, axes = plt.subplots(1, len(h_estab), figsize=(13, 4.2), constrained_layout=True)
    for ax, h in zip(axes, h_estab):
        for s in sorted((s for s in series if s["h"] == h), key=lambda s: s["f"]):
            ax.semilogy(s["t"], s["norma"], color=COLOR[s["f"]], linewidth=2,
                        zorder=3 if s["f"] < 1 else 2,
                        label=rf"$\Delta t$ = {s['f']:g} $\Delta t_c$")
        dt_c = next(s["dt_c"] for s in series if s["h"] == h)
        ax.set_title(rf"h = {h:g},  $\Delta t_c$ = {dt_c:.3e}")
        ax.set_xlabel("t (adimensional)")
        ax.grid(True, which="both", alpha=0.25)
    axes[0].set_ylabel("Norma L2 relativa a la inicial")
    axes[0].legend()
    fig.suptitle(r"Esquema explícito ($\theta$ = 0): estable con 0.9 $\Delta t_c$, explota con 1.1 $\Delta t_c$")
    guardar_figura(fig, "05_estabilidad_normas")

    # Figura: dt_c contra h
    fig, ax = plt.subplots(figsize=(5.6, 4.4), constrained_layout=True)
    ax.loglog(h_med, dtc, "o-", color=COLOR[0.9], linewidth=2, markersize=7,
              label=r"$\Delta t_c$ = 2 / $\lambda_{max}$")
    ax.loglog(h_med, dtc[-1] * (h_med / h_med[-1]) ** 2 * 0.6, color="#6b7280",
              linestyle="--", linewidth=1, label="pendiente 2")
    ax.set_xticks(h_med)
    ax.set_xticklabels([f"{v:.3f}" for v in h_med])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("h medio de los elementos (adimensional)")
    ax.set_ylabel(r"$\Delta t_c$ (adimensional)")
    ax.set_title(r"Paso crítico del esquema explícito")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    guardar_figura(fig, "05_estabilidad_dtc")
    return estables and explotan


def parte_B(reportar):
    nodes, elements, bnd, _ = mesh.make_mesh(Lx, Ly, h_global_osc, h_osc, xs, ys, r_ref_osc)
    n_ana = fields.solucion_gaussiana(1.0, xs, ys, sigma0_B, D0)
    n0 = lambda x, y: n_ana(x, y, 0.0)

    res = {}
    for th in (0.5, 1.0):
        res[th] = timestep.simular(nodes, elements, bnd, D_func, n0, T_osc, dt_osc, theta=th,
                                   Q=None, bc="dirichlet",
                                   t_guardar=np.arange(0.0, T_osc + 1e-9, dt_osc))
    t = res[1.0]["t"]
    n_min = {th: postproc.minimo(res[th]["N"]) for th in res}
    n_max = {th: res[th]["N"].max(axis=1) for th in res}

    reportar("B. OSCILACIONES CON GAUSSIANA ESTRECHA")
    reportar(f"   Malla refinada: h = {h_osc:g} hasta r = {r_ref_osc:g} de la fuente,"
             f" h_global = {h_global_osc:g} ({len(nodes)} nodos)")
    reportar(f"   sigma0 = {sigma0_B:g} (= 2h), dt = {dt_osc:g},"
             f" D_par dt / sigma0^2 = {D_par * dt_osc / sigma0_B ** 2:.0f}")
    reportar(f"   {'t':>5s} {'min n CN':>11s} {'max n CN':>10s} {'min n EI':>11s} {'max n EI':>10s}"
             f" {'max n exacta':>13s}")
    for k, tk in enumerate(t):
        exacta_max = n_ana(xs, ys, tk)
        reportar(f"   {tk:5.1f} {n_min[0.5][k]:11.3e} {n_max[0.5][k]:10.3e}"
                 f" {n_min[1.0][k]:11.3e} {n_max[1.0][k]:10.3e} {exacta_max:13.3e}")
    reportar("   CN = Crank-Nicolson (theta = 0.5), EI = Euler implícito (theta = 1)")
    peor_CN, peor_EI = n_min[0.5][1:].min(), n_min[1.0][1:].min()
    reportar(f"   Mínimo en toda la simulación: CN {peor_CN:.3e}, EI {peor_EI:.3e}"
             f"  (razón {abs(peor_CN) / max(abs(peor_EI), 1e-300):.1e})")
    reportar()

    with open(RESULTADOS / "05_oscilaciones.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["t", "min_n_theta_0.5", "max_n_theta_0.5", "min_n_theta_1", "max_n_theta_1"])
        for k, tk in enumerate(t):
            w.writerow([tk, n_min[0.5][k], n_max[0.5][k], n_min[1.0][k], n_max[1.0][k]])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), constrained_layout=True)
    nombres = {0.5: r"$\theta$ = 0.5 (Crank-Nicolson)", 1.0: r"$\theta$ = 1 (Euler implícito)"}
    ax = axes[0]
    for th in (0.5, 1.0):
        ax.plot(t, n_min[th], "o-", color=COLOR[th], linewidth=2, markersize=6, label=nombres[th])
    ax.axhline(0.0, color="#6b7280", linewidth=1)
    ax.set_xlabel("t (adimensional)")
    ax.set_ylabel("min n (adimensional)")
    ax.set_title("Mínimo de n en el tiempo")
    ax.grid(True, alpha=0.25)
    ax.legend()

    s = np.linspace(-3.0, 3.0, 1201)
    xp, yp = xs + s * np.cos(alpha), ys + s * np.sin(alpha)
    for ax, tc in zip(axes[1:], t_cortes):
        k = int(np.argmin(np.abs(t - tc)))
        for th in (0.5, 1.0):
            ax.plot(s, postproc.evaluar_en_puntos(nodes, elements, res[th]["N"][k], xp, yp),
                    color=COLOR[th], linewidth=2, label=nombres[th])
        ax.plot(s, n_ana(xp, yp, t[k]), color="#111827", linestyle="--", linewidth=1.2,
                label="Analítica")
        ax.axhline(0.0, color="#6b7280", linewidth=1)
        ax.set_title(f"Corte a lo largo de B, t = {t[k]:g}")
        ax.set_xlabel("Distancia al centro (adimensional)")
        ax.set_ylabel("n (adimensional)")
        ax.grid(True, alpha=0.25)
    axes[1].legend(loc="lower left", fontsize=9)
    fig.suptitle(rf"Gaussiana estrecha ($\sigma_0$ = {sigma0_B:g} = 2h) con $\Delta t$ = {dt_osc:g}")
    guardar_figura(fig, "05_oscilaciones")
    return peor_CN < 0 and abs(peor_CN) > 10 * abs(min(peor_EI, 0.0))


def main():
    RESULTADOS.mkdir(exist_ok=True)
    lineas = []

    def reportar(texto=""):
        print(texto, flush=True)
        lineas.append(texto)

    ok_A = parte_A(reportar)
    ok_B = parte_B(reportar)
    reportar("RESULTADO: "
             f"estabilidad {'OK' if ok_A else 'FALLA'}; "
             f"CN oscila y EI mucho menos: {'sí' if ok_B else 'NO'}")

    (RESULTADOS / "05_estabilidad.txt").write_text("\n".join(lineas) + "\n", encoding="utf-8")
    print("\nGuardado en results/: 05_estabilidad_normas, 05_estabilidad_dtc, 05_oscilaciones"
          " (png/pdf), 05_estabilidad_dtc.csv, 05_oscilaciones.csv, 05_estabilidad.txt")
    return 0 if ok_A and ok_B else 1


if __name__ == "__main__":
    sys.exit(main())
