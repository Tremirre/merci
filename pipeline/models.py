from sklearn.base import BaseEstimator
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


def get_classifiers_dict() -> dict[str, BaseEstimator]:
    return {
        "AdaBoost": AdaBoostClassifier,
        "NearestNeighbors": KNeighborsClassifier,
        "DecisionTree": DecisionTreeClassifier,
    }
