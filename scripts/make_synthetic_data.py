"""Genera un CSV SINTÉTICO con el mismo esquema que EPHPM2025.csv.

Sirve únicamente como prueba de humo (smoke test): verifica que el notebook
corre de punta a punta sin necesidad de los microdatos reales del INE.
Los clusters que salgan de estos datos NO significan nada sustantivo.

Uso:
    python scripts/make_synthetic_data.py --out data/EPHPM2025_synthetic.csv
"""
import argparse
import numpy as np
import pandas as pd


def make(n_households: int = 1500, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for hogar in range(1, n_households + 1):
        size = int(np.clip(rng.poisson(3.5) + 1, 1, 12))
        dominio = rng.integers(1, 4)
        depto = rng.integers(1, 19)
        # "capital educativo" latente del hogar, para que haya algo de estructura
        latent = rng.normal()
        for p in range(1, size + 1):
            asiste = int(rng.random() < 0.25)
            ever = 1 if asiste else int(rng.random() < 0.9)
            nivel = int(np.clip(np.round(2 + latent + rng.normal(0, 1)), 1, 5))
            if ever == 2 or rng.random() < 0.08:
                nivel = 9  # 9 = Ninguno
            anos = 0 if nivel == 9 else int(np.clip(nivel * 3 + rng.normal(0, 2), 0, 22))
            rows.append(dict(
                HOGAR=hogar, RELA_J=1 if p == 1 else int(rng.integers(2, 8)), NPER=p,
                SEXO=int(rng.integers(1, 3)), DOMINIO=dominio, DEPTO=depto,
                ED02=1 if asiste else 2, ED03=ever, ED08=int(rng.integers(1, 10)),
                ED09=int(rng.integers(1, 8)), ED15=1 if rng.random() < 0.85 else 2,
                nivel=nivel, anosest=anos,
            ))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/EPHPM2025_synthetic.csv")
    ap.add_argument("--n", type=int, default=1500)
    a = ap.parse_args()
    make(a.n).to_csv(a.out, index=False)
    print(f"Escrito {a.out}")
