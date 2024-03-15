import numpy as np
import structlog
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import merci
from datasets import get_datasets_dict
from models import get_classifiers_dict

_logger = structlog.get_logger()


def merge_arrays(arr1: np.ndarray, arr2: np.ndarray) -> np.ndarray:
    return np.concatenate([arr1, arr2])


def run_pipeline() -> None:
    datasets = get_datasets_dict()
    classifiers = get_classifiers_dict()
    for data_name, dataset in datasets.items():
        x, y = dataset
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.8, random_state=42
        )
        for clf_name, clf in classifiers.items():
            _logger.info(f"Running {clf_name} on {data_name} dataset")
            clf = clf()
            clf.fit(x_train, y_train)
            y_pred = clf.predict(x_test)
            accuracy = accuracy_score(y_test, y_pred)
            evaluator = merci.evaluate.TransductiveEvaluator(
                clf,
                (x_train, y_train),
                (x_test, y_test),
                lambda a, b: (merge_arrays(a[0], b[0]), merge_arrays(a[1], b[1])),
            )
            reliability = evaluator.evaluate()
            _logger.info("Metrics", Reliability=reliability, Accuracy=accuracy)


def main() -> None:
    run_pipeline()


if __name__ == "__main__":
    main()
