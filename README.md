# Perfiles educativos de los hogares hondureños con una Restricted Boltzmann Machine

Aprendizaje **no supervisado** sobre datos categóricos de la **EPHPM 2025** (INE, Honduras): cada hogar se describe con variables educativas, se codifica en binario, se comprime con un **RBM apilado** (72 → 40 → 20 dimensiones), se agrupa con **KMeans** y se visualiza con **t-SNE**.

El repositorio está pensado para *leerse como un tutorial*: el notebook alterna código, teoría ("Guía de estudio") e interpretación de cada figura.

> **Método.** Inspirado en Deng, L. (2022), *Unsupervised Categorical Learning of Objects and Feature Dimension Reduction Using a Brain-Inspired Algorithm*, Towards Data Science.
> **Datos.** Microdatos del INE, **no incluidos** (ver [`data/README.md`](data/README.md)).

---

## 1. La idea en un minuto

Una encuesta de hogares tiene una fila por *persona*, pero la pregunta de interés es sobre el *hogar*. El flujo es:

```
Personas (28,714 filas)
   │  feature engineering: jefe del hogar + agregados de todos los miembros
   ▼
Hogares (7,870 filas × variables categóricas)
   │  one-hot encoding
   ▼
Matriz binaria X  (hogares × ~72 indicadores 0/1)
   │  RBM capa 1  →  H1 (40)
   │  RBM capa 2  →  H2 (20)      ← representación latente "comprimida"
   ▼
KMeans sobre H2 (k elegido por silhouette)   +   t-SNE para visualizar
   ▼
Clusters de hogares con perfiles educativos distintos
```

**¿Por qué un RBM?** Es un modelo probabilístico de dos capas (visible/oculta) que aprende qué indicadores *co-ocurren*. Sus unidades ocultas funcionan como detectores de patrones y sus activaciones sirven como representación compacta. Es un buen vehículo didáctico para entender energía, muestreo de Gibbs y representaciones latentes antes de pasar a autoencoders modernos.

## 2. Estructura del repositorio

```
.
├── notebooks/
│   └── EPHPM2025_RBM_Analysis.ipynb   # El análisis completo, con guías de estudio
├── docs/
│   └── EXTENDING.md                   # Cómo extender el proyecto (con esqueletos de código)
├── scripts/
│   └── make_synthetic_data.py         # Datos sintéticos para prueba de humo
├── data/                              # Aquí va el CSV del INE (ignorado por git)
├── figures/                           # Las figuras se guardan aquí al ejecutar
├── outputs/                           # CSV de resultados (ignorado por git)
└── requirements.txt
```

## 3. Cómo ejecutarlo

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# con microdatos reales:  data/EPHPM2025.csv
jupyter lab notebooks/EPHPM2025_RBM_Analysis.ipynb
```

Sin microdatos, para verificar que todo corre:

```bash
python scripts/make_synthetic_data.py --out data/EPHPM2025_synthetic.csv
EPHPM_CSV=data/EPHPM2025_synthetic.csv jupyter lab notebooks/EPHPM2025_RBM_Analysis.ipynb
```

Tiempo aproximado: unos minutos (t-SNE es la parte más lenta).

## 4. Ruta de lectura sugerida

| Sección del notebook | Qué aprenderás |
|---|---|
| 1. Carga | Estructura persona-dentro-de-hogar; por qué `NPER` no es el tamaño del hogar |
| 2. Feature engineering | Del nivel persona al nivel hogar: variables del jefe vs. agregados |
| 3. One-hot | Por qué el `BernoulliRBM` exige entradas en [0, 1] |
| 4. RBM | Función de energía, entrenamiento, apilamiento, lectura de pesos |
| 5. KMeans | Selección de *k* con silhouette |
| 6–7. t-SNE y figuras | Qué sí y qué no se puede interpretar de un t-SNE |
| 9. Extensiones | Hoja de ruta (detallada en [`docs/EXTENDING.md`](docs/EXTENDING.md)) |

## 5. Salidas

Al ejecutar se generan en `figures/`:

- `fig1_tsne_comparacion.png` — t-SNE de datos brutos vs. RBM capa 1 vs. RBM capa 2
- `fig2_clusters_kmeans.png` — clusters KMeans junto al nivel educativo del jefe
- `fig3_perfiles_cluster.png` — heatmap de perfiles y tamaños de cluster
- `fig4_silhouette.png` — selección de *k*

y en `outputs/ephpm2025_cluster_labels.csv` la etiqueta de cluster y coordenadas t-SNE por hogar.

## 6. Limitaciones (léelas antes de citar resultados)

1. **Sin factores de expansión.** La EPHPM es un muestreo complejo; sin `FACTOR`, el análisis describe la *muestra*, no a Honduras.
2. **Validación circular.** `nivel` entra al RBM y luego se usa para colorear; que los clusters "coincidan" con el nivel educativo es una prueba de consistencia interna, no una validación externa. La validación fuerte es cruzar con variables que el modelo nunca vio (`Pobreza`, `nbi`).
3. **Falta una línea base.** No hay comparación con PCA/SVD sobre el mismo input, así que no está demostrado que el RBM aporte más que una reducción lineal.
4. **t-SNE es visual, no métrico.** Distancias absolutas y tamaños de cúmulos no son interpretables.
5. **Silhouette favorece pocos clusters** y no mide utilidad sustantiva.
6. **Codificación de `ED03` y `ED15`.** Verifica en el diccionario del INE qué significa el código `1` (¿"Sí"?) antes de interpretar `hh_has_dropout` y `hh_has_illiterate`.
7. Las interpretaciones escritas en el notebook corresponden a la corrida original del autor; con otra semilla, versión de scikit-learn o datos, **revisa tus propios gráficos**.

## 7. Cómo extender el proyecto

Resumen (detalle, orden y código en [`docs/EXTENDING.md`](docs/EXTENDING.md)):

1. **Rigor:** línea base PCA/SVD, métricas cuantitativas de la representación, estabilidad de clusters.
2. **Validez externa:** cruzar con `Pobreza`/`nbi`, usar `FACTOR`, análisis por departamento.
3. **Más datos:** multi-bloque (vivienda, servicios, TIC, empleo) y comparación EPHPM 2024 vs. 2025.
4. **Otros modelos:** VAE, DBN de 3 capas, clustering alternativo (GMM, HDBSCAN, k-modes).
5. **Ingeniería:** pasar el notebook a paquete (`src/`), añadir pruebas y CI.

## 8. Referencias

- Goodfellow, Bengio, Courville. *Deep Learning*, cap. 20 (RBM y DBN).
- Hinton, G. (2012). *A Practical Guide to Training Restricted Boltzmann Machines.*
- van der Maaten & Hinton (2008). *Visualizing Data using t-SNE.*
- Wattenberg et al. *How to Use t-SNE Effectively* (distill.pub).
- Documentación de scikit-learn: `BernoulliRBM`, `TSNE`, `KMeans`.
- INE Honduras. Metodología y diccionario de variables de la EPHPM.

## 9. Licencia y uso de datos

*(Añade aquí la licencia del código, p. ej. MIT.)* Los microdatos de la EPHPM pertenecen al INE y no se redistribuyen en este repositorio.
