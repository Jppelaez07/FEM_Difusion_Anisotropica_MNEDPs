"""Prueba 3.4 de Pakmor et al. (2016): medición del coeficiente de difusión perpendicular
numérico con la prueba de Sovinec (Sovinec et al. 2004).

Dominio [-0.5, 0.5]^2, B = (cos(pi x) sin(pi y), -sin(pi x) cos(pi y)), D_par = 1,
Q = 2 pi^2 psi con psi = cos(pi x) cos(pi y), Dirichlet u = 0. Las líneas de B son las
curvas de nivel de psi, así que con difusión perpendicular kappa el estado estacionario
exacto es u = psi / kappa. Con D_perp = 0, el estado estacionario numérico mide la
difusión perpendicular numérica: kappa_num = 1 / u_h(0, 0).
Las mallas se generan en [0, 1]^2 y se desplazan a [-0.5, 0.5]^2.

1. Calibración con D_perp = 1e-2 y 1e-3 en la malla de 51 x 51.
2. Medición con D_perp = 0 en mallas estructuradas y no estructuradas.
3. Evolución temporal hacia el equilibrio en la malla base (theta = 1).
Guarda CSV, figuras PNG/PDF y results/08_sovinec.txt.
"""

import csv
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse.linalg import eigsh, spsolve

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from src import assembly, fields, mesh, postproc, rutas, timestep, viz  # noqa: E402

D_par = 1.0
D_perp_calibracion = [1e-2, 1e-3]
N_lista = [17, 33, 51, 65, 129, 257]
if rutas.RAPIDO:
    N_lista = [N for N in N_lista if N < 257]
N_base = 51
N_mapas = [17, 51, 129]
tipos = ["estructurada", "no estructurada"]
pasos_evolucion = 1000

RESULTADOS = rutas.RESULTADOS
COLOR_TIPO = {"estructurada": "#0072B2", "no estructurada": "#D55E00"}


def psi(x, y):
    return np.cos(np.pi * x) * np.cos(np.pi * y)


def grad_psi(x, y):
    return (-np.pi * np.sin(np.pi * x) * np.cos(np.pi * y),
            -np.pi * np.cos(np.pi * x) * np.sin(np.pi * y))


def B(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    return (np.cos(np.pi * x) * np.sin(np.pi * y), -np.sin(np.pi * x) * np.cos(np.pi * y))


def Q(x, y, t=0.0):
    return 2.0 * np.pi ** 2 * psi(x, y)


def malla(tipo, N):
    if tipo == "estructurada":
        nodes, elements, bnd, _ = mesh.malla_estructurada(1.0, 1.0, N, N)
    else:
        nodes, elements, bnd, _ = mesh.malla_uniforme(1.0, 1.0, 1.0 / (N - 1))
    return nodes - 0.5, elements, bnd


def estacionario(nodes, elements, bnd, D_perp):
    """Resuelve K_II N = F_I y retorna N (todos los nodos), M, K, interiores y el residuo relativo."""
    D_func = fields.funcion_tensor(B, D_par, D_perp)
    geom = assembly.geometria(nodes, elements)
    M, K = assembly.ensamblar_MK(nodes, elements, D_func, geom=geom)
    F = assembly.ensamblar_F(nodes, elements, Q, 0.0, geom=geom)
    inte = np.setdiff1d(np.arange(len(nodes)), bnd)
    K_ii, F_i = assembly.restringir(K, inte), F[inte]
    N_i = spsolve(K_ii.tocsc(), F_i)
    residuo = np.linalg.norm(K_ii @ N_i - F_i) / np.linalg.norm(F_i)
    N = np.zeros(len(nodes))
    N[inte] = N_i
    return N, M, K, inte, residuo, geom


def valor_centro(nodes, elements, N):
    """u_h(0, 0): el valor nodal si (0, 0) es un nodo; si no, interpolado."""
    d = np.hypot(nodes[:, 0], nodes[:, 1])
    k = int(np.argmin(d))
    if d[k] < 1e-12:
        return float(N[k])
    return float(postproc.evaluar_en_puntos(nodes, elements, N, np.array([0.0]), np.array([0.0]))[0])


def kappa_minimos_cuadrados(nodes, elements, N, geom):
    """kappa_eff = int psi^2 / int psi u_h: el kappa que mejor ajusta u_h ~ psi / kappa."""
    detJ = geom[0]
    xq = np.einsum("qm,emi->eqi", postproc.CUAD_BARI, nodes[elements])
    pq = psi(xq[..., 0], xq[..., 1])
    uq = N[elements] @ postproc.CUAD_BARI.T
    pesos = (detJ / 2.0)[:, None] * postproc.CUAD_PESOS
    return np.sum(pesos * pq * pq) / np.sum(pesos * pq * uq)


def kappa_valor_propio(M, K, inte):
    """lambda_min / (2 pi^2), con lambda_min el menor valor propio de K v = lambda M v (Dirichlet)."""
    K_ii, M_ii = assembly.restringir(K, inte), assembly.restringir(M, inte)
    lam = eigsh(K_ii.tocsc(), k=1, M=M_ii.tocsc(), sigma=0.0, which="LM", return_eigenvectors=False)[0]
    return lam / (2.0 * np.pi ** 2)


def tasas(h, e):
    h, e = np.asarray(h), np.abs(np.asarray(e))
    return np.concatenate(([np.nan], np.log(e[:-1] / e[1:]) / np.log(h[:-1] / h[1:])))


def guardar_csv(nombre, filas):
    with open(RESULTADOS / nombre, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(filas[0].keys()))
        w.writeheader()
        w.writerows(filas)


def main():
    rutas.avisar()
    RESULTADOS.mkdir(exist_ok=True)
    viz.estilo()
    lineas = []

    def reportar(texto=""):
        print(texto, flush=True)
        lineas.append(texto)

    # --- Comprobaciones del problema ---
    g = np.linspace(-0.5, 0.5, 801)
    X, Y = np.meshgrid(g, g)
    Bx, By = B(X, Y)
    gx, gy = grad_psi(X, Y)
    B_grad_psi = np.max(np.abs(Bx * gx + By * gy))
    psi_borde = max(np.max(np.abs(psi(g, -0.5))), np.max(np.abs(psi(-0.5, g))))
    nb, eb, _ = malla("estructurada", N_base)
    baric = nb[eb].mean(axis=1)
    nulos = int(np.sum(np.hypot(*B(baric[:, 0], baric[:, 1])) < fields.TOL_B))
    B_min_baric = float(np.min(np.hypot(*B(baric[:, 0], baric[:, 1]))))

    reportar("PRUEBA 3.4 DE PAKMOR ET AL. (2016): PRUEBA DE SOVINEC (DIFUSIÓN PERPENDICULAR NUMÉRICA)")
    reportar("  Dominio [-0.5, 0.5]^2, B = (cos(pi x) sin(pi y), -sin(pi x) cos(pi y)), D_par = 1,")
    reportar("  Q = 2 pi^2 psi, psi = cos(pi x) cos(pi y), Dirichlet u = 0.")
    reportar(f"  max |B . grad psi| en una rejilla de 801 x 801: {B_grad_psi:.1e} (las líneas de B son curvas de nivel de psi)")
    reportar(f"  max |psi| en la frontera: {psi_borde:.1e} (la condición u = 0 es compatible)")
    reportar("  B = 0 en el centro y en las cuatro esquinas; donde |B| < 1e-14 el código usa D = D_perp I.")
    reportar(f"  En la malla estructurada {N_base} x {N_base}, {nulos} baricentros caen en |B| < 1e-14"
             f" (|B| mínimo en un baricentro: {B_min_baric:.2e}),")
    reportar("  así que con D_perp = 0 ningún elemento queda con D = 0; los cercanos al centro y a las")
    reportar("  esquinas tienen |B| pequeño pero dirección definida.")
    reportar("  Estado estacionario: K_II N = F_I (spsolve). Estimadores de kappa_num:")
    reportar("    centro: 1 / u_h(0, 0);  mín. cuadrados: int psi^2 / int psi u_h;"
             "  valor propio: lambda_min(K, M) / (2 pi^2)")
    reportar()

    # --- 1. Calibración ---
    reportar(f"1. CALIBRACIÓN EN LA MALLA ESTRUCTURADA {N_base} x {N_base} (D_perp > 0, exacta u = psi / D_perp)")
    reportar(f"  {'D_perp':>7s} {'u_h(0,0) D_perp':>15s} {'error L1':>10s} {'L1 rel.':>8s} {'error L2':>10s}"
             f" {'L2 rel.':>8s} {'kappa_num (centro)':>19s} {'kappa_num (mín. cuad.)':>23s}")
    filas_cal = []
    nodes, elements, bnd = malla("estructurada", N_base)
    for Dp in D_perp_calibracion:
        N, M, K, inte, res_, geom = estacionario(nodes, elements, bnd, Dp)
        exacta = lambda x, y, Dp=Dp: psi(x, y) / Dp
        u0 = valor_centro(nodes, elements, N)
        fila = {
            "D_perp": Dp, "u_centro_por_D_perp": u0 * Dp,
            "error_L1": postproc.error_L1(nodes, elements, N, exacta),
            "error_L1_rel": postproc.error_L1_relativo(nodes, elements, N, exacta),
            "error_L2": postproc.errores_L2_H1(nodes, elements, N, exacta, lambda x, y: (0 * x, 0 * y))[0],
            "error_L2_rel": postproc.error_L2_relativo(nodes, elements, N, exacta),
            "kappa_num_centro": 1.0 / u0 - Dp,
            "kappa_num_minimos_cuadrados": kappa_minimos_cuadrados(nodes, elements, N, geom) - Dp,
        }
        filas_cal.append(fila)
        reportar(f"  {Dp:7.0e} {fila['u_centro_por_D_perp']:15.6f} {fila['error_L1']:10.3e} {fila['error_L1_rel']:8.2e}"
                 f" {fila['error_L2']:10.3e} {fila['error_L2_rel']:8.2e} {fila['kappa_num_centro']:19.4e}"
                 f" {fila['kappa_num_minimos_cuadrados']:23.4e}")
    reportar("  kappa_num inferido = (estimador) - D_perp: con D_perp > 0 debe coincidir con la medición")
    reportar("  directa con D_perp = 0 (tabla 2) si la difusión numérica no depende de D_perp.")
    reportar()
    guardar_csv("08_sovinec_calibracion.csv", filas_cal)

    # --- 2. Medición con D_perp = 0 ---
    filas = []
    estados = {}
    for tipo in tipos:
        for N_lado in N_lista:
            t1 = time.time()
            nodes, elements, bnd = malla(tipo, N_lado)
            N, M, K, inte, residuo, geom = estacionario(nodes, elements, bnd, 0.0)
            u0 = valor_centro(nodes, elements, N)
            k_c = 1.0 / u0
            k_mc = kappa_minimos_cuadrados(nodes, elements, N, geom)
            k_vp = kappa_valor_propio(M, K, inte)
            estados[(tipo, N_lado)] = (nodes, elements, N, k_c)
            filas.append({"malla": tipo, "N": N_lado, "h": 1.0 / (N_lado - 1),
                          "h_medio": mesh.calidad_malla(nodes, elements)["h_medio"], "nodos": len(nodes),
                          "u_centro": u0, "kappa_centro": k_c / D_par, "kappa_minimos_cuadrados": k_mc / D_par,
                          "kappa_valor_propio": k_vp / D_par, "residuo_spsolve": residuo})
            print(f"  {tipo}, N = {N_lado}: {len(nodes)} nodos ({time.time() - t1:.1f} s)", flush=True)

    reportar("2. MEDICIÓN CON D_perp = 0 (kappa_num / kappa_par; tasas con h = 1/(N-1))")
    reportar(f"  {'malla':<16s} {'N':>4s} {'nodos':>6s} {'u_h(0,0)':>10s} {'centro':>10s} {'tasa':>5s}"
             f" {'mín. cuad.':>10s} {'tasa':>5s} {'val. propio':>11s} {'residuo':>8s}")
    for tipo in tipos:
        sel = [f for f in filas if f["malla"] == tipo]
        h = [f["h"] for f in sel]
        t_c, t_m = tasas(h, [f["kappa_centro"] for f in sel]), tasas(h, [f["kappa_minimos_cuadrados"] for f in sel])
        for f, a, b in zip(sel, t_c, t_m):
            f["tasa_centro"], f["tasa_minimos_cuadrados"] = a, b
            fa = "    -" if np.isnan(a) else f"{a:5.2f}"
            fb = "    -" if np.isnan(b) else f"{b:5.2f}"
            reportar(f"  {tipo:<16s} {f['N']:4d} {f['nodos']:6d} {f['u_centro']:10.2f} {f['kappa_centro']:10.3e} {fa}"
                     f" {f['kappa_minimos_cuadrados']:10.3e} {fb} {f['kappa_valor_propio']:11.3e}"
                     f" {f['residuo_spsolve']:8.1e}")
        reportar()
    singular = [f for f in filas if not np.isfinite(f["u_centro"]) or f["residuo_spsolve"] > 1e-6]
    reportar("  K_II es invertible en todas las mallas (residuo de spsolve pequeño): la difusión"
             " numérica no es nula." if not singular else
             "  ATENCIÓN: K_II singular o mal condicionada en: "
             + ", ".join(f"{f['malla']} N = {f['N']}" for f in singular))
    reportar()
    guardar_csv("08_sovinec_convergencia.csv", filas)

    # --- 3. Evolución temporal hacia el equilibrio (malla base) ---
    nodes, elements, bnd = malla("estructurada", N_base)
    N_ss, *_ = estacionario(nodes, elements, bnd, 0.0)
    k_base = 1.0 / valor_centro(nodes, elements, N_ss)
    t_eq = 5.0 / (2.0 * np.pi ** 2 * k_base)
    T_ev = t_eq
    dt_ev = T_ev / pasos_evolucion
    k_centro = int(np.argmin(np.hypot(nodes[:, 0], nodes[:, 1])))
    registro = {"t": [], "u": []}

    def al_paso(k, t, N):
        registro["t"].append(t)
        registro["u"].append(N[k_centro])

    D_func = fields.funcion_tensor(B, D_par, 0.0)
    res = timestep.simular(nodes, elements, bnd, D_func, lambda x, y: 0 * x, T_ev, dt_ev, theta=1.0, Q=Q,
                           bc="dirichlet", t_guardar=[T_ev], al_paso=al_paso)
    t_ev = np.array(registro["t"])
    razon = np.array(registro["u"]) / N_ss[k_centro]
    dif_final = np.max(np.abs(res["N"][-1] - N_ss)) / np.max(np.abs(N_ss))
    reportar(f"3. EVOLUCIÓN TEMPORAL (malla estructurada {N_base} x {N_base}, theta = 1, u0 = 0)")
    reportar(f"  kappa_num = {k_base:.4e}; tiempo de equilibrio estimado 5 / (2 pi^2 kappa_num) = {t_eq:.1f}")
    reportar(f"  dt = {dt_ev:.4g}, {pasos_evolucion} pasos hasta t = {T_ev:.1f}")
    reportar(f"  u_h(0,0,t) / u_h,estacionario(0,0) en t = {T_ev:.1f}: {razon[-1]:.5f}"
             f" (esperado ~ 1 - exp(-5) = {1 - np.exp(-5):.5f})")
    reportar(f"  max |N(t = {T_ev:.1f}) - N_estacionario| / max |N_estacionario| = {dif_final:.2e}")
    reportar()

    # --- Comparación con el artículo y nota ---
    reportar("4. COMPARACIÓN CON EL ARTÍCULO")
    for tipo in tipos:
        sel = [f for f in filas if f["malla"] == tipo]
        for clave, nombre in (("kappa_centro", "centro"), ("kappa_minimos_cuadrados", "mín. cuad.")):
            pend = -np.polyfit(np.log([f["N"] for f in sel]), np.log([f[clave] for f in sel]), 1)[0]
            reportar(f"  {tipo:<16s} {nombre:<11s}: kappa_num ~ N^-{pend:.2f}")
    f17 = [f for f in filas if f["malla"] == "estructurada" and f["N"] == 17]
    if f17:
        reportar(f"  Con 16 x 16 celdas (N = 17) en la malla estructurada: kappa_num / kappa_par ="
                 f" {f17[0]['kappa_centro']:.2e} (centro)")
    reportar("  Pakmor et al. (Fig. 10) obtienen kappa_num / kappa_par ~ N^-1.85, por debajo de 1e-2")
    reportar("  con 16^2 celdas. La comparación es de tendencia: los métodos numéricos son distintos.")
    reportar()
    reportar("  Relación con la Parte C de 12_sensibilidad.py: allí, con un anillo de B circular, la")
    reportar("  difusión perpendicular numérica no tenía una definición única (la varianza radial y el")
    reportar("  ajuste del perfil daban valores distintos). Aquí el problema está construido para que")
    reportar("  el estado estacionario sea psi / kappa con cualquier kappa perpendicular, así que con")
    reportar("  D_perp = 0 el equilibrio mide directamente kappa_num, y los tres estimadores (centro,")
    reportar("  mínimos cuadrados y valor propio) se pueden comparar entre sí.")
    reportar()

    # --- Figura 1: campo B sobre psi ---
    fig, ax = plt.subplots(figsize=(6.4, 5.6), constrained_layout=True)
    cf = ax.contourf(X, Y, psi(X, Y), levels=30, cmap="viridis")
    fig.colorbar(cf, ax=ax, label=r"$\psi$ = cos($\pi$x) cos($\pi$y)  (Q = 2$\pi^2\psi$)")
    xs = np.linspace(-0.5, 0.5, 60)
    Xs, Ys = np.meshgrid(xs, xs)
    ax.streamplot(Xs, Ys, *B(Xs, Ys), color="white", linewidth=0.7, density=1.2, arrowsize=0.8)
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-0.5, 0.5)
    ax.set_aspect("equal")
    ax.set_xlabel("x (adimensional)")
    ax.set_ylabel("y (adimensional)")
    ax.set_title(r"Líneas de B sobre $\psi$ (y la fuente Q $\propto \psi$)")
    ax.grid(False)
    viz.guardar(fig, RESULTADOS / "08_sovinec_campo")

    # --- Figura 2: mapas del estado estacionario (u_h kappa_num) ---
    N_m = [N for N in N_mapas if N in N_lista]
    fig, axes = plt.subplots(1, len(N_m) + 1, figsize=(4.2 * (len(N_m) + 1), 4.2), constrained_layout=True)
    niveles = np.linspace(-0.05, 1.05, 23)
    for ax, N_lado in zip(axes, N_m):
        nodes, elements, N, k_c = estados[("estructurada", N_lado)]
        cf = ax.tricontourf(nodes[:, 0], nodes[:, 1], elements, N * k_c, levels=niveles, cmap="viridis",
                            extend="both")
        ax.set_title(rf"N = {N_lado}: $u_h\,\kappa_{{num}}$ ($\kappa_{{num}}$ = {k_c:.1e})", fontsize=10)
    axes[-1].contourf(X, Y, psi(X, Y), levels=niveles, cmap="viridis", extend="both")
    axes[-1].set_title(r"$\psi$ (exacta)")
    for ax in axes:
        ax.set_aspect("equal")
        ax.set_xlabel("x (adimensional)")
        ax.set_ylabel("y (adimensional)")
        ax.grid(False)
    fig.colorbar(cf, ax=axes.tolist(), shrink=0.85, label=r"$u_h\,\kappa_{num}$")
    fig.suptitle(r"Estado estacionario con $D_\perp$ = 0 (mallas estructuradas)")
    viz.guardar(fig, RESULTADOS / "08_sovinec_mapas")

    # --- Figura 3: convergencia de kappa_num ---
    fig, ax = plt.subplots(figsize=(6.8, 5.0), constrained_layout=True)
    for tipo in tipos:
        sel = [f for f in filas if f["malla"] == tipo]
        Ns = [f["N"] for f in sel]
        ax.loglog(Ns, [f["kappa_centro"] for f in sel], "o-", color=COLOR_TIPO[tipo], markersize=7,
                  label=f"{tipo}, 1 / u_h(0,0)")
        ax.loglog(Ns, [f["kappa_minimos_cuadrados"] for f in sel], "s--", color=COLOR_TIPO[tipo],
                  markersize=6, markerfacecolor="white", label=f"{tipo}, mínimos cuadrados")
    Nr = np.array([N_lista[0], N_lista[-1]], dtype=float)
    e0 = [f["kappa_centro"] for f in filas if f["malla"] == "estructurada"][0]
    ax.loglog(Nr, 1.3 * e0 * (Nr / Nr[0]) ** -1.85, color="#6b7280", linestyle=":", label=r"$\propto N^{-1.85}$")
    ax.loglog(Nr, 0.7 * e0 * (Nr / Nr[0]) ** -2.0, color="#6b7280", linestyle="-.", label=r"$\propto N^{-2}$")
    ax.set_xlabel("N (nodos por lado)")
    ax.set_ylabel(r"$\kappa_{\perp,num} / \kappa_\parallel$")
    ax.set_title("Prueba de Sovinec: difusión perpendicular numérica")
    ax.legend(fontsize=8)
    viz.guardar(fig, RESULTADOS / "08_sovinec_convergencia")

    # --- Figura 4: evolución temporal ---
    fig, ax = plt.subplots(figsize=(7, 4.4), constrained_layout=True)
    ax.plot(t_ev, razon, color="#0072B2", label="FEM, theta = 1")
    ax.plot(t_ev, 1.0 - np.exp(-2.0 * np.pi ** 2 * k_base * t_ev), color="#111827", linestyle="--",
            linewidth=1.2, label=r"$1 - e^{-2\pi^2 \kappa_{num} t}$")
    ax.axhline(1.0, color="#9ca3af", linewidth=0.8)
    ax.set_xlabel("t (adimensional)")
    ax.set_ylabel(r"$u_h(0,0,t) / u_{h,est}(0,0)$")
    ax.set_title(f"Evolución hacia el equilibrio (malla estructurada {N_base} x {N_base})")
    ax.legend()
    viz.guardar(fig, RESULTADOS / "08_sovinec_evolucion")

    (RESULTADOS / "08_sovinec.txt").write_text("\n".join(lineas) + "\n", encoding="utf-8")
    print("Guardado en " + RESULTADOS.name + "/: 08_sovinec.txt, 08_sovinec_calibracion.csv,"
          " 08_sovinec_convergencia.csv y figuras 08_sovinec_campo/mapas/convergencia/evolucion")
    return 0


if __name__ == "__main__":
    sys.exit(main())
