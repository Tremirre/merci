from dataclasses import dataclass
import structlog
from pathlib import Path

import merci

_logger = structlog.get_logger()


@dataclass
class MerciExperimentConfig:
    """
    Class with ml training parameters
    """

    evaluator_type: merci.ModelEvaluator
    train_to_test_ratio: float
    output_path: Path

    def log_self(self) -> None:
        """
        Log current config
        """
        _logger.info(f"Running with following config: {self}")
