"""Ejemplo mínimo de un pipeline reproducible."""
import json
from pathlib import Path
 
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
 
SEED = 42  # semilla fija para que el resultado sea siempre el mismo
 
 
def main():
    X, y = load_breast_cancer(return_X_y=True)
    modelo = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    f1 = cross_val_score(modelo, X, y, cv=cv, scoring="f1_macro")
 
    resultados = {
        "f1_macro_media": round(float(f1.mean()), 4),
        "f1_macro_de": round(float(f1.std()), 4),
    }
    Path("results").mkdir(exist_ok=True)
    Path("results/metricas.json").write_text(
        json.dumps(resultados, indent=2), encoding="utf-8"
    )
    print(resultados)
 
 
if __name__ == "__main__":
    main()
