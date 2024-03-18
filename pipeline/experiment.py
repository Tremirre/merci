import numpy as np
import pandas as pd
import structlog
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from config import MerciExperimentConfig
from datasets import get_datasets_dict
from models import get_classifiers_dict

_logger = structlog.get_logger()


class MerciExperiment:
    """
    Class for running multiple ml trainings to test merci evaluator
    """

    def __init__(self, cfg: MerciExperimentConfig) -> None:
        """
        :param cfg: merci experiment config
        """
        self.cfg: MerciExperimentConfig = cfg

    @staticmethod
    def merge_arrays(arr1: np.ndarray, arr2: np.ndarray) -> np.ndarray:
        return np.concatenate([arr1, arr2])

    def run_pipeline(self) -> None:
        datasets = get_datasets_dict()
        classifiers = get_classifiers_dict()
        results_dict = {}
        for data_name, dataset in datasets.items():
            x, y = dataset
            x_train, x_test, y_train, y_test = train_test_split(
                x, y, train_size=self.cfg.train_to_test_ratio, random_state=42
            )
            results_dict[data_name] = {}
            for clf_name, clf in classifiers.items():
                _logger.info(f"Running {clf_name} on {data_name} dataset")
                clf = clf()
                clf.fit(x_train, y_train)
                y_pred = clf.predict(x_test)
                accuracy = accuracy_score(y_test, y_pred)
                evaluator = self.cfg.evaluator_type(
                    clf,
                    (x_train, y_train),
                    (x_test, y_test),
                    lambda a, b: (
                        self.merge_arrays(a[0], b[0]),
                        self.merge_arrays(a[1], b[1]),
                    ),
                )
                reliability = evaluator.evaluate()
                _logger.info("Metrics", Reliability=reliability, Accuracy=accuracy)
                results_dict[data_name][clf_name] = [accuracy, reliability]
        results_df = (
            pd.DataFrame.from_dict(results_dict, orient="index").stack().to_frame()
        )
        results_df = pd.DataFrame(
            results_df[0].values.tolist(),
            index=results_df.index,
            columns=["accuracy", "reliability"],
        )
        results_df.to_csv(f"{self.cfg.output_path}/results.csv")
