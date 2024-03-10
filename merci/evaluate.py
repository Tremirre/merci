import abc
import merci.types
import merci.exceptions


class ModelEvaluator(abc.ABC):
    """
    Base class for model evaluators.
    Validates the dataset and provides a common interface for evaluation.
    """

    def __init__(
        self,
        model: merci.types.BinaryClassifier,
        dataset: merci.types.Dataset,
    ) -> None:
        self._validate_dataset(dataset)
        self.model = model
        self.dataset = dataset

    @staticmethod
    def _validate_dataset(dataset: merci.types.Dataset) -> None:
        """
        Validate the input dataset.

        :param dataset: A tuple containing input and target data
        :raises merci.exceptions.InvalidDataset: If input and target data have different lengths
        :raises merci.exceptions.InvalidDataset: If input data is empty
        :raises merci.exceptions.InvalidDataset: If target data is not binary
        """

        X, y = dataset
        if len(X) != len(y):
            raise merci.exceptions.InvalidDataset(
                "Input and target data must have the same length"
            )
        if not X:
            raise merci.exceptions.InvalidDataset("Input data must not be empty")

        try:
            are_all_binary = all(0 <= yi <= 1 for yi in y)
        except TypeError as e:
            raise merci.exceptions.InvalidDataset(
                "Target data must be binary, with values in [0, 1]"
            ) from e

        if not are_all_binary:
            raise merci.exceptions.InvalidDataset(
                "Target data must be binary, with values in [0, 1]"
            )

    @abc.abstractmethod
    def evaluate(self) -> float:
        """
        Evaluate the model using the provided dataset.
        """
        pass
