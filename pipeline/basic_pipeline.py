import structlog

from datasets import get_datasets_dict
from models import get_classifiers_dict

_logger = structlog.get_logger()


def run_pipeline() -> None:
    datasets = get_datasets_dict()
    classifiers = get_classifiers_dict()
    for data_name, dataset in datasets.items():
        for clf_name, clf in classifiers.items():
            _logger.info(f"Running {clf_name} on {data_name} dataset")


def main() -> None:
    run_pipeline()


if __name__ == "__main__":
    main()
