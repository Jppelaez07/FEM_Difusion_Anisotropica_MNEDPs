"""Animaciones de los resultados (requiere correr antes scripts/10_casos.py).

1. Una animación por caso (results/casos/*.npz): escala fija, isolíneas y líneas de B.
2. Comparación 2x2 de los casos 1, 2, 3 y 4b, sincronizada.
3. Crank-Nicolson frente a Euler implícito con la gaussiana estrecha de 05_estabilidad.
4. Anillo con B circular (parte C de 12_sensibilidad): h = 0.4 frente a h = 0.05, 1000:1.
Unos 10 s a 20 fps y 1080 px de ancho. Guarda en mp4 (H.264, ffmpeg de imageio-ffmpeg);
si falla, en gif.
Uso: python scripts/20_animaciones.py [casos] [comparacion] [cn] [anillo]   (sin argumentos, todo)
"""

import importlib.util
import sys
import time
from pathlib import Path

import imageio_ffmpeg
import matplotlib
matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, Normalize, SymLogNorm

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from src import casos, fields, mesh, postproc, rutas, timestep, viz  # noqa: E402

plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()

FPS, CUADROS = 20, 200                  # 10 s
ANCHO_PULGADAS, DPI = 10.8, 100         # 1080 px
MAPA = "inferno"
DECADAS = 3
ISOLINEAS = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]
N_REJILLA = 360
DIR_ANIM = rutas.RESULTADOS / "animaciones"
COLOR_ISO = "#d1d5db"
COLOR_B = "#9ca3af"


def cargar_script(nombre):
    """Importa un script de scripts/ (solo sus parámetros; main no se ejecuta)."""
    spec = importlib.util.spec_from_file_location(nombre, RAIZ / "scripts" / f"{nombre}.py")
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


def guardar_animacion(fig, actualizar, n_cuadros, nombre):
    """Guarda en mp4 (H.264); si falla, en gif como respaldo."""
    t0 = time.time()
    anim = animation.FuncAnimation(fig, actualizar, frames=n_cuadros, blit=False)
    ruta = DIR_ANIM / f"{nombre}.mp4"
    try:
        escritor = animation.FFMpegWriter(
            fps=FPS, codec="libx264",
            extra_args=["-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p"])
        anim.save(ruta, writer=escritor, dpi=DPI)
    except Exception as error:                      # sin ffmpeg o error al codificar
        print(f"  mp4 falló ({error}); se guarda en gif", flush=True)
        ruta.unlink(missing_ok=True)
        ruta = DIR_ANIM / f"{nombre}.gif"
        anim.save(ruta, writer=animation.PillowWriter(fps=FPS), dpi=DPI)
    plt.close(fig)
    print(f"  {ruta.name}: {ruta.stat().st_size / 1e6:.2f} MB ({time.time() - t0:.0f} s)", flush=True)
    return ruta


def interpolar_tiempo(t_datos, valores, t):
    """Interpolación lineal en el tiempo entre instantes guardados."""
    k = int(np.clip(np.searchsorted(t_datos, t) - 1, 0, len(t_datos) - 2))
    a = (t - t_datos[k]) / (t_datos[k + 1] - t_datos[k])
    return (1 - a) * valores[k] + a * valores[k + 1]


def lineas_de_campo(ax, c, extent):
    """Líneas de B en gris tenue (no se dibujan en el caso isotrópico)."""
    if c["D_par"] == c["D_perp"]:
        return
    x = np.linspace(extent[0], extent[1], 60)
    y = np.linspace(extent[2], extent[3], 60)
    X, Y = np.meshgrid(x, y)
    Bx, By = casos.construir_campo(c)(X, Y)
    ax.streamplot(X, Y, Bx, By, color=COLOR_B, linewidth=0.6, density=1.1, arrowsize=0.01,
                  broken_streamlines=False, zorder=3)


def parametros(c):
    campo = c["campo"]
    if campo["tipo"] == "uniforme":
        txt_b = f"B uniforme a {campo['alpha_grados']:g}°"
    elif campo["tipo"] == "sinusoidal":
        txt_b = rf"B = (1, {campo['A']:g} cos(2$\pi$x/{campo['lam']:g}))"
    else:
        txt_b = f"B circular en ({campo['xc']:g}, {campo['yc']:g})"
    fuente = "fuente continua" if c["fuente"]["t_apagado"] is None else \
        f"fuente hasta t = {c['fuente']['t_apagado']:g}"
    return (rf"$D_\parallel$ = {c['D_par']:g}, $D_\perp$ = {c['D_perp']:g}, {txt_b}, "
            f"{c['bc'].capitalize()}, {fuente}")


class PanelCaso:
    """Mapa de un caso sobre una rejilla regular, con isolíneas y líneas de B."""

    def __init__(self, ax, d, norma, titulo):
        c = d["config"]
        self.ax, self.d, self.norma = ax, d, norma
        Lx, Ly = c["dominio"]["Lx"], c["dominio"]["Ly"]
        self.xg, self.yg = np.linspace(0, Lx, N_REJILLA), np.linspace(0, Ly, N_REJILLA)
        self.P = viz.matriz_rejilla(d["nodes"], d["elements"], self.xg, self.yg)
        self.extent = (0, Lx, 0, Ly)
        self.img = ax.imshow(np.full((N_REJILLA, N_REJILLA), norma.vmin), origin="lower",
                             extent=self.extent, cmap=MAPA, norm=norma, interpolation="bilinear")
        lineas_de_campo(ax, c, self.extent)
        f = c["fuente"]
        ax.plot(f["xs"], f["ys"], marker="+", color="#22d3ee", markersize=9, mew=2, zorder=4)
        ax.set_title(titulo, fontsize=11)
        ax.set_xlabel("x (adimensional)")
        ax.set_ylabel("y (adimensional)")
        ax.grid(False)
        self.contornos = None

    def dibujar(self, t):
        N = interpolar_tiempo(self.d["t_guardados"], self.d["N_guardados"], t)
        Z = (self.P @ N).reshape(N_REJILLA, N_REJILLA)
        self.img.set_data(np.maximum(Z, self.norma.vmin))
        if self.contornos is not None:
            self.contornos.remove()
        niveles = [v for v in ISOLINEAS if self.norma.vmin < v < Z.max()]
        self.contornos = (self.ax.contour(self.xg, self.yg, Z, levels=niveles, colors=COLOR_ISO,
                                          linewidths=0.7, zorder=2) if niveles else None)


def norma_log(datos):
    vmax = max(d["N_guardados"].max() for d in datos)
    return LogNorm(vmin=vmax * 10.0 ** (-DECADAS), vmax=vmax)


def barra_color(fig, panel, axes):
    cb = fig.colorbar(panel.img, ax=axes, shrink=0.85)
    cb.set_label("n (adimensional, escala logarítmica)")
    ticks = [v for v in ISOLINEAS if panel.norma.vmin <= v <= panel.norma.vmax]
    cb.set_ticks(ticks)
    cb.set_ticklabels([f"{v:g}" for v in ticks])


def animacion_caso(nombre):
    d = casos.cargar_caso(nombre)
    c = d["config"]
    fig, ax = plt.subplots(figsize=(ANCHO_PULGADAS, 8.6), constrained_layout=True)
    panel = PanelCaso(ax, d, norma_log([d]), "")
    ax.set_aspect("equal")
    barra_color(fig, panel, ax)
    fig.suptitle(c["titulo"].replace("D_par/D_perp", r"$D_\parallel/D_\perp$") + "\n" + parametros(c),
                 fontsize=12)
    reloj = ax.text(0.02, 0.97, "", transform=ax.transAxes, color="white", fontsize=13,
                    va="top", zorder=5)
    tiempos = np.linspace(0.0, d["t_guardados"][-1], CUADROS)

    def actualizar(k):
        panel.dibujar(tiempos[k])
        reloj.set_text(f"t = {tiempos[k]:6.1f}")

    return guardar_animacion(fig, actualizar, CUADROS, nombre)


def animacion_comparacion():
    nombres = ["caso1", "caso2", "caso3", "caso4b"]
    datos = [casos.cargar_caso(n) for n in nombres]
    norma = norma_log(datos)
    fig, axes = plt.subplots(2, 2, figsize=(ANCHO_PULGADAS, 9.6), constrained_layout=True)
    paneles = []
    for ax, d in zip(axes.ravel(), datos):
        c = d["config"]
        titulo = c["titulo"].replace("D_par/D_perp", r"$D_\parallel/D_\perp$")
        paneles.append(PanelCaso(ax, d, norma, titulo))
        ax.set_aspect("equal")
    barra_color(fig, paneles[0], axes.ravel().tolist())
    reloj = fig.suptitle("", fontsize=13)
    T = datos[0]["t_guardados"][-1]
    tiempos = np.linspace(0.0, T, CUADROS)

    def actualizar(k):
        for p in paneles:
            p.dibujar(tiempos[k])
        reloj.set_text(f"Casos 1, 2, 3 y 4b (fuente continua, Dirichlet)   t = {tiempos[k]:6.1f}")

    return guardar_animacion(fig, actualizar, CUADROS, "comparacion_casos")


def animacion_cn_ei():
    s05 = cargar_script("05_estabilidad")
    xs, ys = s05.xs, s05.ys
    nodes, elements, bnd, _ = mesh.make_mesh(s05.Lx, s05.Ly, s05.h_global_osc, s05.h_osc, xs, ys,
                                             s05.r_ref_osc)
    n_ana = fields.solucion_gaussiana(1.0, xs, ys, s05.sigma0_B, s05.D0)
    t_pasos = np.arange(0.0, s05.T_osc + 1e-9, s05.dt_osc)
    res = {th: timestep.simular(nodes, elements, bnd, s05.D_func, lambda x, y: n_ana(x, y, 0.0),
                                s05.T_osc, s05.dt_osc, theta=th, Q=None, bc="dirichlet",
                                t_guardar=t_pasos)["N"] for th in (0.5, 1.0)}

    ventana = 3.0
    xg = np.linspace(xs - ventana, xs + ventana, N_REJILLA)
    yg = np.linspace(ys - ventana, ys + ventana, N_REJILLA)
    P = viz.matriz_rejilla(nodes, elements, xg, yg)
    s = np.linspace(-ventana, ventana, 601)
    xp, yp = xs + s * np.cos(s05.alpha), ys + s * np.sin(s05.alpha)
    corte = {th: np.array([postproc.evaluar_en_puntos(nodes, elements, N, xp, yp) for N in res[th]])
             for th in res}

    norma = SymLogNorm(linthresh=1e-3, vmin=-1.0, vmax=1.0, base=10)
    fig = plt.figure(figsize=(ANCHO_PULGADAS, 9.0), constrained_layout=True)
    sub = fig.subfigures(2, 1, height_ratios=[1.35, 1.0])
    axes = sub[0].subplots(1, 2)
    ax_c = sub[1].subplots(1, 1)
    nombres = {0.5: r"$\theta$ = 0.5 (Crank-Nicolson)", 1.0: r"$\theta$ = 1 (Euler implícito)"}
    colores = {0.5: "#D55E00", 1.0: "#0072B2"}
    imgs = {}
    for ax, th in zip(axes, (0.5, 1.0)):
        imgs[th] = ax.imshow(np.zeros((N_REJILLA, N_REJILLA)), origin="lower", cmap="RdBu_r",
                             norm=norma, extent=(xg[0], xg[-1], yg[0], yg[-1]))
        ax.plot([xp[0], xp[-1]], [yp[0], yp[-1]], color="#6b7280", linewidth=0.8, linestyle="--")
        ax.set_title(nombres[th])
        ax.set_xlabel("x (adimensional)")
        ax.set_ylabel("y (adimensional)")
        ax.grid(False)
    cb = sub[0].colorbar(imgs[1.0], ax=axes.tolist(), shrink=0.85)
    cb.set_label("n (adimensional; rojo > 0, azul < 0)")
    cb.set_ticks([-1, -0.1, -0.01, 0, 0.01, 0.1, 1])
    cb.set_ticklabels(["-1", "-0.1", "-0.01", "0", "0.01", "0.1", "1"])
    lineas = {th: ax_c.plot(s, corte[th][0], color=colores[th], label=nombres[th])[0] for th in res}
    linea_ex, = ax_c.plot(s, n_ana(xp, yp, 0.0), color="#111827", linestyle="--", linewidth=1.2,
                          label="Analítica")
    ax_c.axhline(0, color="#6b7280", linewidth=0.8)
    ax_c.set_yscale("symlog", linthresh=1e-2)
    ax_c.set_ylim(-1.0, 1.2)
    ax_c.set_xlabel("Distancia al centro sobre la línea de B (adimensional)")
    ax_c.set_ylabel("n (adimensional)")
    ax_c.legend(loc="lower left", fontsize=9)
    titulo = fig.suptitle("", fontsize=12)

    por_paso = CUADROS // len(t_pasos)          # cada paso de tiempo se mantiene fijo

    def actualizar(k):
        j = min(k // por_paso, len(t_pasos) - 1)
        for th in res:
            imgs[th].set_data((P @ res[th][j]).reshape(N_REJILLA, N_REJILLA))
            lineas[th].set_ydata(corte[th][j])
        linea_ex.set_ydata(n_ana(xp, yp, t_pasos[j]))
        titulo.set_text(rf"Gaussiana estrecha ($\sigma_0$ = {s05.sigma0_B:g}) con $\Delta t$ = {s05.dt_osc:g}:"
                        f"   paso {j},  t = {t_pasos[j]:.1f}")

    return guardar_animacion(fig, actualizar, por_paso * len(t_pasos), "cn_vs_euler_implicito")


def animacion_anillo():
    s12 = cargar_script("12_sensibilidad")
    xs, ys = s12.xs, s12.ys
    razon = 1000
    D_func = fields.funcion_tensor(fields.campo_circular(xs, ys), razon * s12.D_perp_C, s12.D_perp_C)
    n0 = lambda x, y: s12.anillo(np.hypot(x - xs, y - ys))
    ventana = 6.5
    xg = np.linspace(xs - ventana, xs + ventana, N_REJILLA)
    yg = np.linspace(ys - ventana, ys + ventana, N_REJILLA)
    Xg, Yg = np.meshgrid(xg, yg)
    Rg = np.hypot(Xg - xs, Yg - ys)
    bordes = np.linspace(2.0, 6.0, 101)
    centros = 0.5 * (bordes[1:] + bordes[:-1])
    clase = np.digitize(Rg.ravel(), bordes) - 1
    valida = (clase >= 0) & (clase < len(centros))
    cuenta = np.bincount(clase[valida], minlength=len(centros))

    mallas = {}
    for h in (0.4, 0.05):
        t0 = time.time()
        nodes, elements, bnd, _ = mesh.malla_uniforme(s12.Lx, s12.Ly, h)
        P = viz.matriz_rejilla(nodes, elements, xg, yg)
        cuadros, tiempos = [], []

        def al_paso(k, t, N, P=P, cuadros=cuadros, tiempos=tiempos):
            cuadros.append((P @ N).astype(np.float32))
            tiempos.append(t)

        timestep.simular(nodes, elements, bnd, D_func, n0, s12.T_C, s12.dt_C, theta=1.0, Q=None,
                         bc="neumann", t_guardar=[0.0], al_paso=al_paso)
        Z = np.array(cuadros)
        perfil = np.array([np.bincount(clase[valida], weights=z[valida], minlength=len(centros)) / cuenta
                           for z in Z])
        mallas[h] = {"Z": Z, "t": np.array(tiempos), "perfil": perfil, "nodos": len(nodes)}
        print(f"  anillo h = {h:g}: {len(nodes)} nodos ({time.time() - t0:.0f} s)", flush=True)

    t = mallas[0.05]["t"]
    r_ref, n_ref = s12.referencia_radial(s12.D_perp_C, t)
    norma = Normalize(vmin=0.0, vmax=1.0)
    fig = plt.figure(figsize=(ANCHO_PULGADAS, 9.2), constrained_layout=True)
    sub = fig.subfigures(2, 1, height_ratios=[1.35, 1.0])
    axes = sub[0].subplots(1, 2)
    ax_p = sub[1].subplots(1, 2, width_ratios=[1.6, 1.0])
    colores = {0.4: "#D55E00", 0.05: "#0072B2"}
    imgs = {}
    th = np.linspace(0, 2 * np.pi, 200)
    for ax, h in zip(axes, (0.4, 0.05)):
        imgs[h] = ax.imshow(mallas[h]["Z"][0].reshape(N_REJILLA, N_REJILLA), origin="lower",
                            cmap=MAPA, norm=norma, extent=(xg[0], xg[-1], yg[0], yg[-1]))
        for r in (3.5, 4.5):
            ax.plot(xs + r * np.cos(th), ys + r * np.sin(th), color="#22d3ee", linewidth=0.6,
                    linestyle=":")
        ax.set_title(f"h = {h:g} ({mallas[h]['nodos']} nodos)")
        ax.set_xlabel("x (adimensional)")
        ax.set_ylabel("y (adimensional)")
        ax.grid(False)
    cb = sub[0].colorbar(imgs[0.05], ax=axes.tolist(), shrink=0.85)
    cb.set_label("n (adimensional)")

    lineas = {h: ax_p[0].plot(centros, mallas[h]["perfil"][0], color=colores[h], label=f"FEM h = {h:g}")[0]
              for h in mallas}
    linea_ref, = ax_p[0].plot(r_ref, n_ref[0], color="#111827", linestyle="--", linewidth=1.2,
                              label="Referencia 1D (exacta)")
    ax_p[0].set_xlim(2.0, 6.0)
    ax_p[0].set_ylim(0.0, 1.05)
    ax_p[0].set_xlabel("r (adimensional)")
    ax_p[0].set_ylabel("n promedio en el ángulo")
    ax_p[0].set_title("Perfil radial")
    ax_p[0].legend(fontsize=9, loc="upper right")

    # Ancho radial (desviación estándar del perfil) en el tiempo
    def ancho(rr, nn):
        w = nn * rr
        m = (w * rr).sum(-1) / w.sum(-1)
        return np.sqrt((w * (rr - m[..., None]) ** 2).sum(-1) / w.sum(-1))
    anchos = {h: ancho(centros, mallas[h]["perfil"]) for h in mallas}
    ancho_ref = ancho(r_ref, n_ref)
    for h in mallas:
        ax_p[1].plot(mallas[h]["t"], anchos[h], color=colores[h], label=f"h = {h:g}")
    ax_p[1].plot(t, ancho_ref, color="#111827", linestyle="--", linewidth=1.2, label="Referencia")
    marca = ax_p[1].axvline(0, color="#6b7280", linewidth=0.8)
    ax_p[1].set_xlabel("t (adimensional)")
    ax_p[1].set_ylabel("Ancho radial del perfil, 2 < r < 6 (adimensional)")
    ax_p[1].set_title("Ensanchamiento del anillo")
    ax_p[1].legend(fontsize=9)
    titulo = fig.suptitle("", fontsize=12)

    idx = np.linspace(0, len(t) - 1, CUADROS).round().astype(int)

    def actualizar(k):
        j = idx[k]
        for h in mallas:
            imgs[h].set_data(mallas[h]["Z"][j].reshape(N_REJILLA, N_REJILLA))
            lineas[h].set_ydata(mallas[h]["perfil"][j])
        linea_ref.set_ydata(n_ref[j])
        marca.set_xdata([t[j], t[j]])
        titulo.set_text(rf"Anillo con B circular, $D_\parallel/D_\perp$ = {razon}, $D_\perp$ = {s12.D_perp_C:g}"
                        f" (Neumann):   t = {t[j]:5.1f}")

    return guardar_animacion(fig, actualizar, CUADROS, "anillo_difusion_numerica")


def main():
    DIR_ANIM.mkdir(parents=True, exist_ok=True)
    viz.estilo()
    plt.rcParams["axes.grid"] = False
    partes = [p.lower() for p in sys.argv[1:]] or ["casos", "comparacion", "cn", "anillo"]
    if "casos" in partes:
        for nombre in casos.CASOS:
            animacion_caso(nombre)
    if "comparacion" in partes:
        animacion_comparacion()
    if "cn" in partes:
        animacion_cn_ei()
    if "anillo" in partes:
        animacion_anillo()
    print("\nTamaños en results/animaciones/:")
    for ruta in sorted(DIR_ANIM.iterdir()):
        print(f"  {ruta.name:<34s} {ruta.stat().st_size / 1e6:6.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
