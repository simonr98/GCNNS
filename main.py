from cw2.cw_data import cw_logging
from cw2 import experiment, cluster_work, cw_error
from Experiment import Experiment


class CustomExperiment(experiment.AbstractExperiment):

    def initialize(self, config: dict, rep: int, logger: cw_logging.LoggerArray) -> None:
        pass

    def run(self, config: dict, rep: int, logger: cw_logging.LoggerArray) -> None:
        my_config = config.get("params")

        exp = Experiment(my_config)
        exp.run_experiment()

    def finalize(self, surrender: cw_error.ExperimentSurrender = None, crash: bool = False):
        pass


if __name__ == "__main__":
    cw = cluster_work.ClusterWork(CustomExperiment)
    cw.run()