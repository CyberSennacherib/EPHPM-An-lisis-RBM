# Datos

Este directorio está vacío a propósito: **los microdatos de la EPHPM 2025 no se incluyen en el repositorio**.
Pertenecen al Instituto Nacional de Estadística (INE) de Honduras y se rigen por sus condiciones de uso.

## Opción A — Datos reales
1. Obtén el CSV de la EPHPM 2025 desde el INE.
2. Guárdalo como `data/EPHPM2025.csv` (o define la variable de entorno `EPHPM_CSV` con otra ruta).

El notebook acepta el archivo con o sin la segunda fila de descripciones largas (la detecta solo).

### Columnas que usa el notebook
| Columna | Uso |
|---|---|
| `HOGAR` | Identificador del hogar |
| `RELA_J` | Relación con el jefe (1 = jefe/a) |
| `NPER` | Número de orden de la persona en el hogar (no es el tamaño del hogar) |
| `SEXO`, `DOMINIO`, `DEPTO` | Características del jefe y ubicación |
| `ED02`, `ED03`, `ED08`, `ED09`, `ED15` | Variables educativas (asistencia, nivel, área, alfabetismo) |
| `nivel`, `anosest` | Nivel educativo (1–5, 9 = Ninguno) y años de estudio |

## Opción B — Datos sintéticos (solo para probar que el código corre)
```bash
python scripts/make_synthetic_data.py --out data/EPHPM2025_synthetic.csv
EPHPM_CSV=data/EPHPM2025_synthetic.csv jupyter lab
```
Los resultados sintéticos **no tienen significado sustantivo**.
