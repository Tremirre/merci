import numpy as np
from sklearn.datasets import load_breast_cancer, load_wine, make_classification


def get_datasets_dict() -> dict[str, tuple[np.ndarray]]:
    breast_cancer_data = load_breast_cancer(return_X_y=True)
    wine_x, wine_y = load_wine(return_X_y=True)
    wine_y[wine_y >= 7] = 1  # good
    wine_y[wine_y < 7] = 0  # bad
    random_binary_data = make_classification(n_classes=2)

    return {
        "BreastCancer": breast_cancer_data,
        "WineQuality": (wine_x, wine_y),
        "RandomBinary": random_binary_data,
    }
