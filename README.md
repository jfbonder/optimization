# Agujeros óptimos para la constante de traza de Sobolev / primer autovalor de Steklov

Este repositorio contiene:

1. **Versiones revisadas (corregidas) de tres trabajos publicados** sobre la optimización de la mejor constante de la inmersión de traza de Sobolev en dominios con agujeros, con su fuente LaTeX y el PDF compilado.
2. **Una implementación en Python** (paquete `steklov_holes`) del cálculo numérico de agujeros óptimos: el algoritmo del trabajo original y dos algoritmos nuevos.
3. **Scripts de experimentos** que reproducen todas las figuras, tablas y películas del trabajo revisado de *Annali di Matematica*.
4. Los **PDF originales** (versiones finales de los autores) que sirvieron de base.

---

## Índice

- [1. Los trabajos](#1-los-trabajos)
- [2. Resumen de las correcciones](#2-resumen-de-las-correcciones)
- [3. El problema y los métodos numéricos](#3-el-problema-y-los-métodos-numéricos)
- [4. Resultados numéricos](#4-resultados-numéricos)
- [5. ¿Qué método resultó más eficiente?](#5-qué-método-resultó-más-eficiente)
- [6. Estructura del repositorio](#6-estructura-del-repositorio)
- [7. Instalación y uso](#7-instalación-y-uso)
- [8. Referencia de la API](#8-referencia-de-la-api)
- [9. Cómo reproducir todo](#9-cómo-reproducir-todo)
- [10. Limitaciones conocidas y trabajo futuro](#10-limitaciones-conocidas-y-trabajo-futuro)

---

## 1. Los trabajos

| Carpeta | Trabajo | Referencia publicada | Original |
|---|---|---|---|
| `paper1_sobolev_trace_holes/` | J. Fernández Bonder, J. D. Rossi, N. Wolanski, *On the best Sobolev trace constant and extremals in domains with holes* | Bull. Sci. Math. **130** (2006), no. 7, 565–579. MR2261964 | `bon-ros-wol5.pdf` (manuscrito con el título *Behavior of the best Sobolev trace constant and extremals in domains with holes*) |
| `paper3_free_boundary_regularity/` | J. Fernández Bonder, J. D. Rossi, N. Wolanski, *Regularity of the free boundary in an optimization problem related to the best Sobolev trace constant* | SIAM J. Control Optim. **44** (2006), no. 5, 1614–1635 | `bonder.pdf` |
| `paper2_steklov_shape_derivative/` | J. Fernández Bonder, P. Groisman, J. D. Rossi, *Optimization of the first Steklov eigenvalue in domains with holes: a shape derivative approach* | Ann. Mat. Pura Appl. (4) **186** (2007), no. 2, 341–358. DOI 10.1007/s10231-006-0009-y. MR2295124 | `aunum-final.pdf` |

Cada carpeta contiene `main.tex` (fuente reconstruida a partir del PDF), `main.pdf` (compilado) y `CHANGES.md`. **Al comienzo de la Introducción de cada trabajo revisado hay una subsección "Differences with the published version"** que detalla exactamente qué cambia respecto del texto publicado. `CHANGES.md` reproduce esa lista.

Notas sobre las referencias:
- Del paper de Bull. Sci. Math. no pude confirmar el DOI. El número (no. 7) aparece en una sola fuente; conviene verificarlo.
- Para el paper de SICON, algunas fuentes indican 2005 y otras 2006 (vol. 44, no. 5); tampoco pude ver el DOI.
- Desde este entorno no había acceso a Crossref ni a los sitios de las editoriales, así que las referencias se verificaron con búsquedas web.

## 2. Resumen de las correcciones

Ver los `CHANGES.md` y las secciones "Differences with the published version" para el detalle completo. Lo esencial:

**Bull. Sci. Math. (2006)**
- *Teorema 1.6 (continuidad respecto de la distancia de Hausdorff).* Es falso para $1<p\le N$. Contraejemplo: conjuntos finitos de puntos (capacidad nula) que convergen a una bola (capacidad positiva). Ahora el teorema enuncia (a) semicontinuidad superior para todo $p$ y (b) continuidad y convergencia de extremales si $p>N$ o si los agujeros son no crecientes ($A\subset A_n$).
- *Capacidad.* Con la capacidad homogénea, todo acotado tiene capacidad nula si $p=N$, así que el Teorema 1.5 fallaba en $p=N$. Se usa la capacidad de Sobolev $\|\phi\|^p_{W^{1,p}}$, que tiene los mismos conjuntos nulos si $p<N$.
- *Teorema 1.4 (simetría).* El conjunto de ceros de $u^*$ son casquetes centrados en $-re_N$. El agujero simétrico se define como $\{u^*=0\}$.
- Además, detalles menores de la prueba de existencia y erratas.

**SIAM J. Control Optim. (2006)**
- *Condición de signo.* Las clases admisibles no exigían $v\ge0$, aunque las pruebas lo usan. Sin esa condición, $v=-u_\emptyset$ minimiza $J_\varepsilon$ en $K_1$ con $|\{v>0\}|=0$, y los Teoremas 5.2 y 5.3 fallan tal como están enunciados. Agregando $v\ge0$ (equivalente a medir $|\{v\ne0\}|$) todas las pruebas funcionan. Es una **aclaración de hipótesis**; la teoría de regularidad no cambia.
- Constantes en el Lema 2.2 ($\min(1,\varepsilon)$), $u\le C\rho_k$ (no $\rho_k^N$) en el Teorema 2.5, y signos y notación en la Sección 4.

**Ann. Mat. Pura Appl. (2007)**
- *Teorema 2(1)(b).* Afirmaba que la bola centrada nunca es óptima. La prueba necesita $\lambda_1^2+\frac{N-1}{R}\lambda_1>1$, pero el Corolario 2.13 sólo da la versión con $r$. La desigualdad necesaria vale **si y sólo si** $R<R^*(r)$, donde $R^*(r)$ es el único mínimo de $\lambda_1(\cdot,r)$.
  - Para $R<R^*(r)$ la bola centrada es un máximo local estricto entre sus trasladadas, y por lo tanto no es óptima.
  - Para $R>R^*(r)$ es un **mínimo local estricto** entre sus trasladadas. Numéricamente parece ser el óptimo global (Conjetura 2.8).
- *Fórmula explícita* $\lambda_1(R,r)=\dfrac{K_\nu(r)I_{\nu+1}(R)+I_\nu(r)K_{\nu+1}(R)}{K_\nu(r)I_\nu(R)-I_\nu(r)K_\nu(R)}$, con $\nu=\frac{N-2}{2}$. El original decía que no se podía calcular explícitamente.
- *Segunda variación en la bola centrada (nuevo).* Vale $\lambda_1(A_\varepsilon)=\lambda_1(B_r)+\varepsilon^2\varphi'(r)^2\sum_{\ell\ge1}c_\ell h_\ell^2+o(\varepsilon^2)$, con $c_\ell=-\mu_\ell-\frac{N-1}{r}$ explícito en funciones de Bessel. Además $c_1$ tiene el signo de $\partial_R\lambda_1$ (prueba vía el wronskiano de $\varphi'$ y del modo $\ell=1$), y $c_\ell\sim(\ell-1)/r$.
- *Prueba del Lema 2.7.* Se apoyaba en la continuidad Hausdorff de [BRW], que es falsa en general. Ahora tiene una prueba directa.
- *Prueba del Teorema 3.* La construcción de competidores discretos ahora se hace con $(u_\alpha-\delta)^+$ y la estabilidad $W^{1,p}$ del interpolante.
- *La Sección 3 (numérica) está reescrita*: ver abajo.

## 3. El problema y los métodos numéricos

Dado $\Omega\subset\mathbb R^2$ y $0<\alpha<|\Omega|$, se busca un agujero $A\subset\Omega$ con $|A|=\alpha$ que minimice
$$\lambda_1(A)=\min_{u=0\text{ en }A}\frac{\int_\Omega|\nabla u|^2+u^2}{\int_{\partial\Omega}u^2},$$
que es el primer autovalor de $-\Delta u+u=0$ en $\Omega\setminus A$, $\partial_\nu u=\lambda u$ en $\partial\Omega$, $u=0$ en $A$.

**Discretización.** Elementos finitos P1. El problema discreto es $\mathbf A\mathbf u=\lambda\mathbf B\mathbf u$, con $\mathbf B$ (masa de borde) semidefinida. Se calcula el mayor $\mu=1/\lambda$ de $\mathbf B\mathbf u=\mu\mathbf A\mathbf u$ con Lanczos (ARPACK) y una factorización LU de $\mathbf A$. Con unos $10^4$ nodos, cada resolución lleva centésimas de segundo; el original usaba `fmincon`. Los agujeros discretos son uniones de triángulos (la clase $\mathcal O_h$ del paper): $|A_h|\ge\alpha$ y al sacar el último triángulo queda $<\alpha$. Por eso $|A_h|-\alpha=O(h^2)$.

**Algoritmo 1: intercambio de triángulos (el del paper original).** Usa la derivada de forma $\lambda_1'(A)[V]=-\int_{\partial A}(\partial_\nu u)^2\langle V,\nu\rangle$. En cada paso saca del agujero los triángulos de la interfaz con $|\partial_\nu u_h|$ grande y agrega los de $|\partial_\nu u_h|$ chico. Se implementó con las mejoras que sugería el propio paper: varios intercambios al principio y cada vez menos ($m_k=\max(1,10-\lfloor k/20\rfloor)$), y una lista tabú contra ciclos. Su comportamiento:
- no es monótono;
- mueve la interfaz de a una capa de triángulos;
- no sale de configuraciones críticas (por ejemplo, el disco centrado).

**Algoritmo 2: reordenamiento penalizado (nuevo).** Se relaja la condición $u=0$ en $A$ con una penalización:
$$\lambda_\beta(\chi)=\min\frac{\int|\nabla v|^2+v^2+\beta\int\chi v^2}{\int_{\partial\Omega}v^2}.$$
Es el análogo discreto de la relajación de [BRW 2006] y de la penalización de [BRW SICON]. En cada paso el nuevo agujero son los triángulos con menor $\frac1{|T|}\int_T u_k^2$. Propiedades (Proposición 3.2 del trabajo revisado):
- $\lambda_\beta(\chi_k)$ es **monótona no creciente** en $k$;
- $\lambda_\beta\uparrow\lambda_{1,h}$ cuando $\beta\to\infty$; se usa continuación en $\beta$ de $10$ a $10^5$;
- converge en pocas iteraciones y puede cambiar la topología del agujero;
- puede quedar en mínimos locales, así que se prueba desde varios agujeros iniciales (`multistart`).

**Algoritmo 3: flujo de gradiente de forma con malla móvil y refinamiento adaptativo (nuevo).** Es la idea de "mover los triángulos" y "refinar cerca del agujero". Con $g=(\partial_\nu u)^2$ y la velocidad normal $V=g-\bar g$, que conserva el volumen, se tiene $\lambda'=-\int(g-\bar g)^2\le0$. Cada paso:
1. Se mueven los nodos de la interfaz con $(\tau V+\delta)\nu$. La corrección $\delta=(|A_h|-\alpha)/|\partial A_h|$ restaura el volumen y $\tau$ se elige con *backtracking*. Los puntos de contacto con $\partial\Omega$ se deslizan tangencialmente.
2. El desplazamiento se extiende armónicamente a toda la malla.
3. Cuando la calidad de la malla se deteriora, se remalla (DistMesh) con tamaño $h_{\rm fine}$ cerca de la interfaz suavizada, creciendo hasta $h$ lejos de ella.

**Estrategia recomendada:** Algoritmo 2 (con varios inicios) para la topología, seguido del Algoritmo 3 para resolver la frontera libre.

## 4. Resultados numéricos

(Valores de $\lambda_{1,h}$; ver `paper2_steklov_shape_derivative/tables/` y `experiments/results/*.json`.) Los Algoritmos 1 y 2 se corren desde **los mismos agujeros iniciales** (3 o 4 por dominio). El Algoritmo 3 se arranca desde el mejor agujero de cada uno.

| Dominio | Alg. 1 (original): mejor valor (tiempo/corrida) | Alg. 2: mejor valor (tiempo/corrida) | Alg. 1 + 3 | Alg. 2 + 3 |
|---|---|---|---|---|
| Disco unidad, $\alpha=\pi/2$ | **1.24772** (1.9 s) | 1.24998 (0.4 s) | **1.24052** | 1.24096 |
| L (tres cuadrados), $\alpha=1/2$ | 0.46808 (2.4 s) | **0.46695** (0.3 s) | **0.46654** | 0.46690 |
| Dos discos con un mango, $\alpha=\pi$ | 0.50410 (2.0 s) | 0.50410 (0.1 s) | 0.50359 | 0.50359 |
| Disco $B_3$, $\alpha=\pi/4$ | 0.82684 (9.3 s) | **0.82675** (2.0 s) | 0.82680 | **0.82670** |

Sensibilidad a la inicialización (detalle por agujero inicial en `experiments/results/compare.log`):
- **L:** desde el cuadrado de la esquina, el original llega a 0.506 y el reordenamiento queda en 0.577. Desde el centro, el original llega a 0.468 y el reordenamiento a 0.578. Desde los otros dos inicios, ambos encuentran la franja (≈0.467).
- **$B_3$:** el reordenamiento llega al disco centrado desde los tres inicios. El original, arrancado cerca del borde, mueve el agujero de a una capa de triángulos y en 300 iteraciones queda lejos del centro (0.836).

Observaciones:
- **Disco unidad:** el agujero óptimo toca $\partial\Omega$ del lado opuesto a donde se concentra la autofunción. Es un "casquete" con frontera libre curva: ni el disco centrado ni el semidisco (Teorema 2), y es esféricamente simétrico, como predice [BRW].
- **L:** el óptimo es una franja en el extremo de uno de los brazos (hay dos, por simetría).
- **Dos discos:** el agujero llena prácticamente uno de los discos.
- **$B_3$ con agujero chico** ($R=3>R^*(0.5)\approx2.30$): el óptimo numérico es el **disco centrado**, lo que es evidencia a favor de la Conjetura 2.8.
- **Convergencia en $h$** (L, $\alpha=1$, el ejemplo de la Tabla 1 original):

  | $h$ | Alg. 2 | Alg. 3 |
  |---|---|---|
  | 0.2 | 0.54588 | 0.53282 |
  | 0.1 | 0.53713 | 0.53188 |
  | 0.05 | 0.53333 | 0.53122 |
  | 0.025 | 0.53206 | 0.53079 |

- **Traslación del disco centrado:** la segunda derivada de $\lambda_1(B_r(te_1))$ calculada por FEM (con malla que se deforma con el agujero) coincide con la fórmula exacta de la Proposición 2.6, por ejemplo $+1.21\cdot10^{-2}$ contra $+1.22\cdot10^{-2}$ para $R=3$, $r=0.5$. El cambio de signo se produce exactamente en $R^*(r)$.
- **Validación:** en la corona $B_1\setminus B_{0.5}$ el FEM converge como $O(h^2)$ al valor exacto 1.6088088…

**Películas** (`movies/`):
- `*_pipelines.mp4` (una por dominio): **las dos cadenas completas desde el mismo agujero inicial**.
  - A la izquierda, Alg. 1 → Alg. 3. Al Algoritmo 1 se lo detiene cuando el mejor $\lambda$ no mejora en 40 iteraciones, y entonces se pasa al Algoritmo 3.
  - En el centro, Alg. 2 → Alg. 3.
  - A la derecha, las dos curvas de $\lambda$ en los mismos ejes, dibujándose a la par; una línea punteada marca el paso al Algoritmo 3.
  - Las genera `experiments/pipeline_movies.py`, y los resultados quedan en `experiments/results/pipeline_*.json`.
- `*_comparison.mp4` (una por dominio): **el método original y el nuevo lado a lado, desde el mismo agujero inicial**. Arriba el original (Alg. 1), abajo el nuevo (Alg. 2 + 3); cada fila muestra el agujero ($u_h=0$ en negro) y la autofunción, y a la derecha las dos curvas de $\lambda$ se dibujan a medida que avanzan las iteraciones.
- `*_original_algorithm.mp4` y `*_rearrangement_and_flow.mp4`: cada método por separado (corrida anterior).
- En la etapa de reordenamiento la curva muestra el autovalor duro $\lambda_{1,h}$, que puede subir en los primeros pasos: lo que es monótono es el penalizado $\lambda_\beta$.
- `experiments/make_movies.py` regenera las películas a partir de los cuadros guardados.

## 5. ¿Qué método resultó más eficiente?

**Nota.** Una primera versión de mi implementación del algoritmo original lo hacía detenerse antes de tiempo: cuando la interfaz anterior no ofrecía suficientes triángulos para recuperar la medida, el ciclo terminaba (pasaba en la L y en los dos discos). Ya está corregido: ahora se recalcula la interfaz del agujero actual y se agregan triángulos hasta llegar a $\alpha$. Con la corrección, el algoritmo original resulta **competitivo** y las conclusiones cambian respecto de la primera comparación.

| | Tiempo por corrida | Calidad | Comentarios |
|---|---|---|---|
| Alg. 1 (original, corregido) | 2–10 s (300 iteraciones) | dentro del 0.2 % del Alg. 2 (a veces mejor) | No es monótono y oscila. Mueve la interfaz de a una capa por paso: lento para trasladar el agujero lejos ($B_3$) y no sale del disco centrado. En la L es **más robusto** frente a la inicialización. |
| Alg. 2 (reordenamiento) | **0.1–3 s** | dentro del 0.2 % del Alg. 1 (a veces mejor) | Monótono en $\lambda_\beta$; converge en ≲20 iteraciones y cambia la topología en un paso. Es **5 a 20 veces más rápido**, pero más sensible al agujero inicial (en la L queda en 0.577 desde 2 de 4 inicios). |
| Alg. 3 (flujo de forma) | 1–3 min | **el mejor** (0.01–0.7 % menos) | Frontera libre suave y ajustada a la malla, con refinamiento cerca del agujero. Su resultado final casi no depende de qué algoritmo discreto le dio el agujero inicial. |

Conclusión:
- En tiempo, **el más eficiente es el reordenamiento penalizado (Alg. 2)**.
- En robustez no domina: en la L, el algoritmo original (corregido) encuentra buenas configuraciones desde más inicios.
- Como ambos métodos discretos son baratos, la estrategia recomendada es correr **los dos desde unos pocos agujeros iniciales y pasar el mejor resultado al flujo de forma (Alg. 3)**, que da la frontera libre con precisión y los valores más bajos.

## 6. Estructura del repositorio

```
├── README.md                         este archivo
├── requirements.txt
├── aunum-final.pdf  bon-ros-wol5.pdf  bonder.pdf     versiones finales originales
├── paper1_sobolev_trace_holes/       Bull. Sci. Math. revisado (main.tex, main.pdf, CHANGES.md)
├── paper2_steklov_shape_derivative/  Ann. Mat. Pura Appl. revisado
│   ├── figures/                      figuras generadas por experiments/
│   └── tables/                       tablas LaTeX generadas por experiments/make_tables.py
├── paper3_free_boundary_regularity/  SIAM J. Control Optim. revisado
├── steklov_holes/                    paquete Python
│   ├── mesh.py        mallas (DistMesh, disco, corona, L, dos discos con mango, refinamiento)
│   ├── fem.py         ensamblado P1: rigidez, masa, masa de borde, gradientes por elemento
│   ├── eigen.py       primer autopar (agujero duro, Dirichlet, penalizado; chequeo denso)
│   ├── radial.py      fórmulas exactas: λ1(R,r), R*(r), Θ''(0), coeficientes c_ℓ
│   ├── algorithms.py  Algoritmos 1, 2 y 3, multistart, búsqueda local, remallado
│   └── plotting.py    figuras (agujero binario + autofunción) y películas
├── experiments/
│   ├── translation.py       traslación del disco centrado (Tabla y Figura)
│   ├── second_variation.py  diagrama de regímenes en el plano (r, R)
│   ├── optimal_holes.py     configuración de los dominios (y corrida individual de cada método)
│   ├── compare_methods.py   original vs nuevo desde los mismos inicios: tabla, figuras y películas comparativas
│   ├── make_movies.py       regenera las películas a partir de los cuadros guardados
│   ├── pipeline_movies.py   películas de las cadenas Alg. 1 → 3 y Alg. 2 → 3 desde el mismo inicio
│   ├── convergence.py       convergencia en h (L, α = 1)
│   ├── make_tables.py       tablas LaTeX a partir de los JSON
│   └── results/             resultados en JSON y logs
├── movies/                           películas MP4 de las iteraciones
└── tests/test_steklov_holes.py       tests (pytest)
```

## 7. Instalación y uso

Requisitos: Python ≥ 3.10, `numpy`, `scipy`, `matplotlib`. Para las películas MP4 hace falta `imageio-ffmpeg` (trae su propio ffmpeg); sin él se pueden generar GIF. Para compilar los trabajos se usa una distribución TeX con `latexmk`.

```bash
pip install -r requirements.txt
```

Ejemplo mínimo:

```python
import numpy as np
import steklov_holes as sh
from steklov_holes import algorithms as al, plotting as pl
from steklov_holes.mesh import d_circle

alpha = np.pi / 2
mesh = sh.disk(0.04)                       # disco unidad, h = 0.04
S = sh.P1System(mesh)                      # matrices P1

# Algoritmo 2 desde varios agujeros iniciales
res = al.multistart(S, alpha, centers=[(0.25, 0.1), (0, 0), (-0.5, 0)], record=True)
print(res.lam, res.area)

# Algoritmo 3 (malla móvil + remallado adaptativo)
fd = lambda p: d_circle(p, (0, 0), 1.0)
flow = al.shape_gradient_flow(mesh, res.hole, alpha, smooth=25, record=True,
                              remesh_args=dict(fd=fd, bbox=((-1, -1), (1, 1)),
                                               h=0.04, h_fine=0.015))
pl.plot_result(flow, "disco.png")
pl.make_movie(res.info["frames"] + flow.info["frames"], "disco.mp4", fps=4)
```

## 8. Referencia de la API

**`steklov_holes.mesh`**
- `disk(h, radius, center)`, `annulus(h, R, r, hole_center)`, `lshape(h)`, `two_disks_with_handle(h, ...)`: generan mallas `Mesh`.
- `distmesh2d(fd, h0, bbox, pfix, fh)`: mallador general a partir de una función distancia con signo (`d_circle`, `d_rectangle`, `d_union`, `d_difference`).
- `Mesh`: atributos `p`, `t`, `bedges`, `edges`, `e2t`, `areas`, `centroids`, `h`; métodos `refine()` y `deform(D)`.

**`steklov_holes.fem.P1System(mesh, steklov_edges=None)`**
- Matrices `K` (rigidez), `M` (masa), `A = K + M` y `B` (masa de borde).
- Métodos `weighted_mass(w)`, `element_l2sq(u)` y `element_grad(u)`.

**`steklov_holes.eigen`**
- `eig_hard(S, hole)`: $\lambda_{1,h}(A_h)$ y $u_h$ normalizada.
- `eig_dirichlet(S, fixed)`: con nodos de Dirichlet dados.
- `eig_penalized(S, chi, beta)`: $\lambda_\beta(\chi)$.
- `dense_check(S, fixed)`: referencia densa vía el complemento de Schur.

**`steklov_holes.radial`**
- `lambda1(R, r, N)`, `dlambda_dR`, `R_star(r, N)`, `theta2`.
- `c_ell(ell, R, r, N)`, `mu_ell`, `translation_second_derivative(R, r, N)`, `R_crit`.

**`steklov_holes.algorithms`**
- `swap_algorithm(S, hole0, alpha, n_iter, n_swap, tabu)`: Algoritmo 1.
- `rearrangement(S, hole0, alpha, betas)`: Algoritmo 2.
- `multistart(S, alpha, centers, betas_list)`: el Algoritmo 2 desde varios inicios.
- `local_search(S, hole, alpha)`: pulido monótono con evaluaciones exactas.
- `shape_gradient_flow(mesh, hole, alpha, step, smooth, remesh_args, ...)`: Algoritmo 3.
- `remesh(mesh, hole, fd, bbox, h, h_fine, corners)`, `initial_hole(mesh, alpha, center, shape)`, `interface_edges(mesh, hole)`.
- Todos aceptan `record=True` para guardar cuadros (`result.info["frames"]`).
- Todos devuelven `Result` con los campos `mesh`, `hole`, `lam`, `u`, `history`, `areas`, `info`.

**`steklov_holes.plotting`**
- `plot_result(res, fname)`, `plot_holes_panel(states, titles, fname)`.
- `make_movie(frames, fname, fps)`: genera MP4 o GIF según la extensión.
- `make_comparison_movie(frames_a, frames_b, fname)`: película lado a lado de dos métodos, con las dos curvas de $\lambda$.

## 9. Cómo reproducir todo

```bash
cd experiments
```

```bash
python3 translation.py         # ~1 min
```

```bash
python3 second_variation.py    # segundos
```

```bash
python3 compare_methods.py     # ~15 min, incluye películas comparativas
```

```bash
python3 convergence.py         # ~4 min
```

```bash
python3 make_tables.py
```

```bash
cd ../paper2_steklov_shape_derivative && latexmk -pdf main.tex
```

Los tests (rápidos, < 5 s):

```bash
python3 -m pytest -q tests
```

Los tests verifican:
- la fórmula de Bessel contra la EDO radial;
- que $c_1$ se anula exactamente en $R^*(r)$;
- la convergencia $O(h^2)$ del FEM;
- el resolvedor iterativo contra el denso;
- la monotonía del reordenamiento y el límite $\beta\to\infty$;
- la clase $\mathcal O_h$;
- el signo de la segunda derivada de las traslaciones.

## 10. Limitaciones conocidas y trabajo futuro

- **Mínimos locales.** Los Algoritmos 1 y 2 pueden quedar en mínimos locales; correrlos desde varios agujeros iniciales lo resolvió en todos los ejemplos, pero no hay garantía de óptimo global.
- **Artefactos del Algoritmo 3.** Ocasionalmente quedan pequeñas islas de elementos mal clasificados después de remallar. Su efecto en $\lambda$ es despreciable, pero se ven en las figuras.
- **Malla.** El mallador DistMesh es simple; para mallas grandes conviene usar `gmsh`.
- **Posibles mejoras:** un paso de Newton con la segunda variación, regularización por curvatura, una representación por conjuntos de nivel y una extensión a 3D.
- **Problema abierto:** probar la Conjetura 2.8 (optimalidad global de la bola centrada para $R>R^*(r)$). La estrategia propuesta (asintótica de agujeros chicos, *blow-up* y problema sobredeterminado exterior, y unicidad local por la no degeneración $c_\ell>0$) está esbozada en el trabajo revisado.
