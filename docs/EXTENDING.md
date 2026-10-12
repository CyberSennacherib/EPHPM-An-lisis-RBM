# Cómo extender el proyecto

Las extensiones están ordenadas por **valor ÷ esfuerzo**. Cada una indica qué pregunta responde, qué variables del notebook usa y un esqueleto de código. Los fragmentos asumen que ya corriste el notebook (existen `X_bin`, `H`, `H2`, `hh`, `cluster_labels`, `Z0/Z1/Z2`, `RANDOM`).

---

## Nivel 1 — Rigor metodológico (hazlo primero)

### 1. ¿El RBM aporta algo frente a una reducción lineal?
**Pregunta:** el notebook muestra que t-SNE "separa mejor" tras el RBM, pero no compara con una línea base ni cuantifica.

```python
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import trustworthiness
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# Línea base lineal con la misma dimensión que H2
svd = TruncatedSVD(n_components=H2.shape[1], random_state=RANDOM)
H_svd = svd.fit_transform(X_bin.values)

# (a) ¿Se preservan las vecindades del espacio original?
for name, rep in {'SVD-20': H_svd, 'RBM-40': H, 'RBM-20': H2}.items():
    print(name, round(trustworthiness(X_bin.values, rep, n_neighbors=15), 4))

# (b) ¿Qué tan bien predice una regresión logística el nivel del jefe?
#     (cuidado: `nivel` entró al RBM → es un chequeo de información retenida,
#      no una validación externa)
y = hh['nivel_num'].astype(int).values
for name, rep in {'SVD-20': H_svd, 'RBM-20': H2}.items():
    acc = cross_val_score(LogisticRegression(max_iter=1000), rep, y, cv=5).mean()
    print(name, round(acc, 3))
```
Si el RBM no supera claramente al SVD, esa es una conclusión legítima (y publicable en el README).

### 2. Estabilidad de los clusters
**Pregunta:** ¿los clusters sobreviven a otra semilla del RBM y del KMeans?

```python
from sklearn.metrics import adjusted_rand_score

def pipeline(seed, k):
    r1 = BernoulliRBM(n_components=N_HIDDEN,  learning_rate=LR, n_iter=N_ITER,
                      batch_size=BATCH, random_state=seed).fit(X_bin.values)
    h1 = r1.transform(X_bin.values)
    r2 = BernoulliRBM(n_components=N_HIDDEN2, learning_rate=LR, n_iter=N_ITER,
                      batch_size=BATCH, random_state=seed).fit(h1)
    h2 = r2.transform(h1)
    return KMeans(n_clusters=k, n_init=15, random_state=seed).fit_predict(h2)

runs = [pipeline(s, best_k) for s in range(5)]
aris = [adjusted_rand_score(runs[i], runs[j])
        for i in range(5) for j in range(i + 1, 5)]
print(f'ARI medio entre semillas: {np.mean(aris):.3f}')   # ≈1 → estable
```

### 3. Criterios alternativos para elegir *k*
Calinski-Harabasz, Davies-Bouldin y el *gap statistic* complementan al silhouette. Compara también con clustering sobre `X_bin` o `H_svd` para ver cuánto depende el resultado de la representación.

---

## Nivel 2 — Validez sustantiva (validez de los resultados)

### 4. Factores de expansión (`FACTOR`)
La EPHPM es un muestreo complejo; sin ponderar, el análisis describe la muestra.

```python
w = (head.set_index('HOGAR')['FACTOR']
         .reindex(hh['HOGAR']).astype(float).values)
hh['factor'] = w

# Distribución poblacional de clusters (no muestral)
pop_share = hh.groupby('cluster')['factor'].sum() / hh['factor'].sum()

# Estadísticas ponderadas por cluster
hh.groupby('cluster').apply(
    lambda g: pd.Series({
        'nivel_medio': np.average(g['nivel_num'], weights=g['factor']),
        'anos_prom':   np.average(g['hh_mean_anosest'], weights=g['factor']),
    }))

# KMeans ponderado (opcional)
KMeans(n_clusters=best_k, n_init=15, random_state=RANDOM).fit(H2, sample_weight=w)
```
Para varianzas e intervalos de confianza correctos haría falta además considerar estratos y UPM (diseño muestral), p. ej. con réplicas bootstrap.

### 5. Validación externa con `Pobreza` y `nbi`
Estas variables **no** entran al modelo, así que cruzarlas sí es una prueba independiente.

```python
val = hh.merge(df[['HOGAR', 'Pobreza', 'nbi']].drop_duplicates('HOGAR'),
               on='HOGAR', how='left')
print(pd.crosstab(val['cluster'], val['Pobreza'], normalize='index').round(3))

def cramers_v(x, y):
    from scipy.stats import chi2_contingency
    ct = pd.crosstab(x, y)
    chi2 = chi2_contingency(ct)[0]
    return np.sqrt(chi2 / (ct.values.sum() * (min(ct.shape) - 1)))

for var in ['Pobreza', 'nbi', 'DEPTO', 'DOMINIO']:
    print(var, round(cramers_v(val['cluster'].astype(str), val[var].astype(str)), 3))
```

### 6. Geografía
```python
geo = hh.groupby(['DEPTO', 'cluster']).size().unstack(fill_value=0)
sns.heatmap(geo.div(geo.sum(axis=1), axis=0), cmap='YlOrRd')
```
Con un shapefile departamental (`geopandas`) puedes pasar a mapas coropléticos.

### 7. Revisar la codificación de `ED03` / `ED15`
Confirma en el diccionario del INE qué significa `1` y ajusta `hh_has_dropout` / `hh_has_illiterate`. Documenta la decisión en el notebook.

---

## Nivel 3 — Más datos

### 8. Análisis multi-bloque
Añade vivienda (paredes, piso, techo), servicios (agua, saneamiento, electricidad), TIC y empleo. Patrón: una función `build_block_features(df, block)` por bloque, cada una devolviendo un DataFrame a nivel hogar, y concatenar los one-hot. Al crecer la dimensión, considera `n_components` mayores y regularización (`learning_rate` menor, más épocas).

### 9. Comparación temporal 2024 vs. 2025
Refactoriza el pipeline en una función `fit_pipeline(df) -> (hh, X_bin, H2, labels)` y córrelo sobre cada año. Los clusters no son comparables por número de etiqueta: ajusta el modelo de un año y **asigna** el otro (`rbm.transform` + `km.predict`), o alinea clusters con el algoritmo húngaro sobre sus centroides.

---

## Nivel 4 — Otros modelos

| Alternativa | Qué aporta | Nota |
|---|---|---|
| **VAE** (PyTorch) | Espacio latente probabilístico, generación de hogares sintéticos | Entrada binaria → salida con `BCE` |
| **DBN de 3 capas** | Más abstracción | Ver esqueleto en la Sección 9 del notebook |
| **Autoencoder + embeddings por variable** | Maneja categorías sin one-hot disperso | Útil si crecen las categorías |
| **GMM / HDBSCAN / k-modes / LCA** | Clustering que respeta la naturaleza categórica | El **análisis de clases latentes** es el estándar en ciencias sociales: buen contrapunto interpretativo |
| **MCA** (análisis de correspondencias múltiples) | Reducción lineal específica para categóricas | Mejor línea base que SVD sobre one-hot |

---

## Nivel 5 — Ingeniería del repositorio

1. **Modularizar:** mover `build_household_features`, `encode_binary`, `fit_stacked_rbm`, `select_k` a `src/ephpm_rbm/` y dejar el notebook como narrativa que las importa.
2. **Pruebas:** con `pytest` y los datos sintéticos de `scripts/make_synthetic_data.py` (formas de las matrices, que `X_bin` sea 0/1, que cada hogar tenga un único jefe).
3. **CI:** un workflow de GitHub Actions que ejecute el notebook con datos sintéticos (`jupyter nbconvert --execute`) para detectar roturas de código.
4. **Reproducibilidad:** fijar versiones (`pip freeze` o `uv lock`) y registrar en el README la versión de scikit-learn usada en la corrida de referencia.
5. **Notebook con salidas:** tras correr con datos reales, guarda el notebook ejecutado y las figuras agregadas (sin tablas a nivel de hogar) para que GitHub renderice resultados.
