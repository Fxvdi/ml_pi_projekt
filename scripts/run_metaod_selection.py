# /// script
# requires-python = "==3.10.*"
# dependencies = [
#     "scikit-learn==1.0.2",
#     "numpy==1.23.5",
#     "scipy==1.9.3",
#     "pandas==1.5.3",
#     "numba==0.56.4",
#     "pyod",
#     "liac-arff",
#     "joblib",
# ]
# ///
"""MetaOD Detektor-Selektion im Legacy-Stack.

MetaODs vortrainierte Modelle (``trained_models/``) wurden mit scikit-learn
0.22.1 gepickelt. Der moderne Projekt-Stack (sklearn 1.9, numpy 2.x) kann die
gepickelten RandomForest-Trees nicht laden (geändertes C-Struct-Layout).
Deshalb läuft die MetaOD-Selektion isoliert in einem alten Stack, den uv über
die PEP-723-Metadaten oben automatisch bereitstellt:

    uv run --isolated scripts/run_metaod_selection.py <input.npy> <output.json>

``input.npy``  : (n_samples, n_features) Feature-Matrix (bei uns skaliert).
``output.json``: Ranking als Liste von MetaOD-Empfehlungsstrings, z.B.
                 "OCSVM (0.1, 'poly')", bestes zuerst.
"""
import json
import os
import sys

import numpy as np
from joblib import load

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VENDORED_METAOD = os.path.join(PROJECT_ROOT, "external")
MODELS_DIR = os.path.join(PROJECT_ROOT, "trained_models")

sys.path.insert(0, VENDORED_METAOD)  # vendored MetaOD source (pure python)

from metaod.models.gen_meta_features import generate_meta_features

# MetaOD bündelt zwei trainierte Meta-Learner zu einem Ensemble.
ENSEMBLE = ["train_0.joblib", "train_2.joblib"]


def select_models(X: np.ndarray, n_selection: int) -> list[str]:
    """Repliziert metaod.predict_metaod.select_model mit sklearn-Kompatfix.

    Der gepickelte MinMaxScaler stammt aus sklearn 0.22.1; das Attribut ``clip``
    (ab sklearn 1.0) fehlt und wird nachgesetzt, sonst schlägt transform fehl.
    """
    meta_scalar = load(os.path.join(MODELS_DIR, "meta_scalar.joblib"))
    if not hasattr(meta_scalar, "clip"):
        meta_scalar.clip = False

    meta_X, _ = generate_meta_features(X)
    meta_X = np.nan_to_num(meta_X, nan=0)
    meta_X = meta_scalar.transform(np.asarray(meta_X).reshape(1, -1)).astype(float)

    model_list = list(load(os.path.join(MODELS_DIR, "model_list.joblib")))

    scores = np.zeros([len(ENSEMBLE), len(model_list)])
    for i, model_file in enumerate(ENSEMBLE):
        clf = load(os.path.join(MODELS_DIR, model_file))
        scores[i, ] = clf.predict(meta_X)

    combined = np.average(scores, axis=0)
    top_idx = np.flip(np.argsort(combined))[:n_selection]
    return [str(model_list[i]) for i in top_idx]


def main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit(
            "Usage: run_metaod_selection.py <input.npy> <output.json> [n_selection]"
        )
    input_path, output_path = sys.argv[1], sys.argv[2]
    n_selection = int(sys.argv[3]) if len(sys.argv) > 3 else 20

    X = np.load(input_path)
    ranking = select_models(X, n_selection)

    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(ranking, fh, indent=2)

    print(f"MetaOD-Ranking ({len(ranking)}) -> {output_path}")
    for rank, name in enumerate(ranking, 1):
        print(f"  {rank:2}. {name}")


if __name__ == "__main__":
    main()
