import typing

InputData = typing.TypeVar("InputData")
TargetData = typing.TypeVar("TargetData")

Dataset = typing.Tuple[InputData, TargetData]


class BinaryClassifier(typing.Protocol):
    def fit(self, X: InputData, y: TargetData) -> None:
        pass

    def predict(self, X: InputData) -> TargetData:
        pass
