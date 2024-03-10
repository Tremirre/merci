import abc
import typing

import numpy as np

import merci.types
import merci.exceptions
import merci.measures


class ModelEvaluator(abc.ABC):
    """
    Base class for model evaluators.
    Validates the dataset and provides a common interface for evaluation.
    """

    def __init__(
        self,
        model: merci.types.Classifier,
        train_dataset: merci.types.Dataset,
        test_dataset: merci.types.Dataset,
    ) -> None:
        """
        Initialize the model evaluator.

        :param model: Classification model to evaluate
        :param train_dataset: A tuple containing input and target data for training
        :param test_dataset: A tuple containing input and target data for testing
        """
        self.validate_dataset(train_dataset)
        self.validate_dataset(test_dataset)

        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

    def validate_dataset(self, dataset: merci.types.Dataset) -> None:
        """
        Validate the input dataset.

        :param dataset: A tuple containing input and target data
        :raises merci.exceptions.InvalidDataset: If input and target data have different lengths
        :raises merci.exceptions.InvalidDataset: If input data is empty
        """

        X, y = dataset
        if len(X) != len(y):
            raise merci.exceptions.InvalidDataset(
                "Input and target data must have the same length"
            )
        if not len(X):
            raise merci.exceptions.InvalidDataset("Input data must not be empty")

        all_target_numeric = all(isinstance(yi, np.number) for yi in y)
        all_target_distribution = all(
            isinstance(yi, np.ndarray) and np.isclose(np.sum(yi), 1) and np.all(yi >= 0)
            for yi in y
        )
        if not (all_target_numeric or all_target_distribution):
            raise merci.exceptions.InvalidDataset(
                "Target data must be numeric or a probability distribution"
            )

    @abc.abstractmethod
    def evaluate(self) -> float:
        """
        Evaluate the model using the provided dataset.
        """
        pass


class TransductiveEvaluator(ModelEvaluator):
    """
    Transductive model evaluator.
    """

    Merger = typing.Callable[
        [merci.types.Dataset, merci.types.Dataset], merci.types.Dataset
    ]

    def __init__(
        self,
        model: merci.types.Classifier,
        train_dataset: merci.types.Dataset,
        test_dataset: merci.types.Dataset,
        merger: Merger = lambda train, test: (train[0] + test[0], train[1] + test[1]),
    ) -> None:
        """
        Initialize the transductive model evaluator.

        :param model: Classification model to evaluate
        :param train_dataset: A tuple containing input and target data for training
        :param test_dataset: A tuple containing input and target data for testing
        :param merger: A function to merge the train and test datasets, defaults to concatenation
        """
        super().__init__(model, train_dataset, test_dataset)
        self.merger = merger

    def evaluate(self) -> float:
        """
        Evaluate the model using the provided dataset.

        :return: The accuracy of the model on the dataset
        """
        X_train, y_train = self.train_dataset
        X_test, _ = self.test_dataset

        y_test = self.model.predict(X_test)
        merged_dataset = self.merger((X_train, y_train), (X_test, y_test))
        self.model.fit(*merged_dataset)

        y_transductive = np.array(self.model.predict(X_test))
        y_test = np.array(y_test)

        score = merci.measures.reliability_estimation(y_test, y_transductive)
        return score
