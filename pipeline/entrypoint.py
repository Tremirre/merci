from experiment import MerciExperiment
from options import parse_args


def main() -> None:
    cfg = parse_args()
    experiment = MerciExperiment(cfg=cfg)
    experiment.run_pipeline()


if __name__ == "__main__":
    main()
