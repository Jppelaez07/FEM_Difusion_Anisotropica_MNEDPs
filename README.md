# Difusión anisotrópica de rayos cósmicos con elementos finitos

Proyecto del curso **Métodos Numéricos para Ecuaciones en Derivadas Parciales**, Universidad EAFIT.
Autores: Gabriela Sandoval Restrepo, Natalia Gabriela Gómez Martínez y Juan Pablo Peláez Montoya.
Docente: Nicolás Guarín Zapata.

## Descripción

Se resuelve la ecuación de difusión anisotrópica

    dn/dt = div( D(x, y) grad n ) + Q(x, y, t)   en  Omega = [0, Lx] x [0, Ly],

donde `n` es la densidad de rayos cósmicos, `Q` una fuente y
`D = D_perp I + (D_par - D_perp) b b^T` el tensor de difusión, con `b = B/|B|` la
dirección del campo magnético. Donde `|B| = 0` se usa `D = D_perp I`.

Método numérico (ver `docs/Formulacion_matematica.pdf`):

- Elementos finitos triangulares lineales (P1) en mallas no estructuradas generadas con gmsh,
  con refinamiento alrededor de la fuente.
- Matrices locales: `M_e = |J|/24 [[2,1,1],[1,2,1],[1,1,2]]`, `K_e = |J|/2 H^T D H`
  (D en el baricentro) y `F_e = |J|/6 Q` (Q en el baricentro). Ensamblaje vectorizado.
- Condiciones de frontera de Dirichlet homogénea (caso base) o Neumann homogénea.
- Método theta en el tiempo:
  `(M + theta dt K) N_{k+1} = (M - (1 - theta) dt K) N_k + dt (theta F_{k+1} + (1 - theta) F_k)`,
  con Euler implícito (`theta = 1`) como caso base y la matriz factorizada una sola vez.

Todas las simulaciones están en unidades adimensionales.

## Instalación

Probado con Python 3.14 en Windows. Las versiones de las dependencias están fijadas en
`requirements.txt`.

Windows (PowerShell):

```
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Linux o macOS:

```
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

`imageio-ffmpeg` incluye ffmpeg, así que no hace falta instalarlo aparte para las animaciones.

## Cómo correr

Hay dos formas de correr todo el proyecto. Ambas imprimen el tiempo de cada script y un
resumen OK/FALLO, y guardan la salida completa de cada script en `<carpeta>/logs/`.

**Corrida completa** (unos 45 minutos; las animaciones toman la mayor parte). Genera todos los
resultados del informe en `results/`:

```
python scripts/run_all.py
```

**Corrida rápida** (unos 5 minutos), para probar que el código funciona:

```
python scripts/run_all.py --rapido
```

En este modo no se generan las animaciones, `12_sensibilidad.py` omite las mallas de h = 0.05,
`06_pakmor.py` omite la malla de N = 401 y `10_casos.py` usa T = 100 en lugar de 300. Los resultados son reducidos y se guardan en
`results_rapido/`, de modo que nunca sobrescriben los de `results/`. Los scripts que cambian
en este modo lo indican al inicio con "MODO RÁPIDO: resultados reducidos, no usar en el informe".

Las animaciones ya están incluidas en `results/animaciones/`, así que no hace falta generarlas
para verlas.

También se puede correr cada script por separado, desde la carpeta raíz del proyecto:

```
python scripts/03_mms.py
```

`11_figuras_casos.py` y `20_animaciones.py` usan los resultados de `10_casos.py`, que debe
correrse antes. `12_sensibilidad.py` acepta las partes a correr (`A`, `B`, `C`) y
`20_animaciones.py` las animaciones (`casos`, `comparacion`, `cn`, `anillo`); sin argumentos
corren todo.

## Estructura

```
docs/        formulación matemática
src/         código del método
  mesh.py      mallas (gmsh, uniforme, estructurada) y calidad de malla
  fields.py    campos B, tensor D, fuente Q, condición inicial y soluciones exactas
  assembly.py  ensamblaje de M, K y F
  timestep.py  método theta y simulación completa con Dirichlet o Neumann
  postproc.py  masa, momentos, flujo por la frontera, errores L2 y H1
  casos.py     configuración de los escenarios y ejecución de un caso
  viz.py       estilo de las figuras e interpolación a rejillas
  rutas.py     carpeta de resultados y modo rápido
scripts/     verificaciones, escenarios, figuras y animaciones (cada uno corre solo)
results/     resultados completos (incluye las animaciones en results/animaciones/)
results_rapido/  resultados reducidos del modo rápido (no se versiona)
```

## Qué genera cada script

| Script | Contenido | Archivos en `results/` |
|---|---|---|
| `01_malla.py` | Malla no estructurada con refinamiento; calidad; área total, simetría de M y K, K·1 = 0 | `malla.png`, `01_malla.txt` |
| `02_prueba_rapida.py` | Gaussiana con Neumann y Q = 0: conservación de la masa y elongación en la dirección de B | `02_prueba_rapida.png`, `02_prueba_rapida.txt` |
| `03_mms.py` | Solución manufacturada (fuente verificada con sympy): convergencia espacial (L2 ~ h², H1 ~ h) y temporal (theta = 1 ~ dt, theta = 0.5 ~ dt²) | `03_mms_espacial.*`, `03_mms_temporal.*` (csv, png, pdf), `03_mms.txt` |
| `04_gauss.py` | Comparación con la gaussiana anisotrópica analítica: error L2 en el tiempo y cortes a lo largo y a través de B | `04_gauss_error.*`, `04_gauss_cortes.*`, `04_gauss.txt` |
| `05_estabilidad.py` | Paso crítico del esquema explícito (dt_c ~ h²) y oscilaciones de Crank-Nicolson frente a Euler implícito | `05_estabilidad_normas.*`, `05_estabilidad_dtc.*`, `05_oscilaciones.*`, `05_estabilidad.txt` |
| `06_pakmor.py` | Prueba 3.1 de Pakmor et al. (2016), difusión isotrópica de un perfil gaussiano en [-0.5, 0.5]²: caso base 51 x 51 y convergencia del error L1 y L2 en mallas estructuradas y no estructuradas (N = 26 a 401), theta = 0.5 y theta = 1 | `06_pakmor_mapas.*`, `06_pakmor_cortes.*`, `06_pakmor_convergencia.*` (png, pdf), `06_pakmor_caso_base.csv`, `06_pakmor_convergencia.csv`, `06_pakmor.txt` |
| `10_casos.py` | Escenarios 1 a 5: masa, flujo por la frontera, momentos, razón de aspecto, t_99, comparación con la solución exacta de la fuente continua y balance de masa | `casos/<caso>.npz`, `casos/resumen.csv`, `casos/resumen.txt`, `casos/comparacion_exacta.csv` |
| `11_figuras_casos.py` | Figuras de los escenarios | `casos/estados_finales.*`, `casos/masa_flujo.*`, `casos/balance_caso5.*`, `casos/comparacion_exacta.*` |
| `12_sensibilidad.py` | A: barrido de D_par/D_perp. B: difusión perpendicular numérica con B uniforme. C: difusión numérica con B circular frente a una referencia radial 1D | `12_barrido_anisotropia.*`, `12_difusion_numerica.*`, `12_campo_curvo.*`, `12_sensibilidad.txt`, `12_campo_curvo.txt` |
| `20_animaciones.py` | Animación de cada caso, comparación 2x2 de los casos 1, 2, 3 y 4b, Crank-Nicolson frente a Euler implícito y anillo con B circular en dos mallas | `animaciones/*.mp4` |

Las figuras se guardan en PNG y PDF.

## Escenarios (`src/casos.py`)

Dominio [0, 20]², fuente gaussiana continua (Q0 = 1, sigma_Q = 0.5), n0 = 0, Dirichlet,
theta = 1, dt = 0.05.

| Caso | Descripción |
|---|---|
| 1 | Isotrópico, D_par = D_perp = 1 |
| 2 | D_par/D_perp = 10, B uniforme horizontal |
| 3 | D_par/D_perp = 100, B uniforme horizontal |
| 4a | D_par/D_perp = 10, B = (1, cos(2 pi x / 10)) |
| 4b | D_par/D_perp = 100, B circular centrado en (10, 10), fuente en (14, 10) |
| 5 | D_par/D_perp = 10, B a 30°, fuente encendida hasta t = 5, con Dirichlet y con Neumann |

Los valores de cada caso están en `src/casos.py`: `COMUN` contiene la configuración común y
`CASOS` lo que cambia en cada uno.
