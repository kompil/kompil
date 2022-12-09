from kompil.profile.bench import Benchmark
from kompil.metrics.metrics import factory as metrics_factory


class Criteria:
    @staticmethod
    def from_keywords(**kwargs) -> "Criteria":
        data = dict()
        for key, target_value in kwargs.items():
            submetric, metric = key.split("_")
            if metric not in data:
                data[metric] = dict()
            data[metric][submetric] = target_value
        return Criteria(data)

    def __init__(self, data):
        self.__counstraints = data

    @property
    def active(self):
        return self.__counstraints is not None

    def reached(self, benchmark: Benchmark) -> bool:
        """
        Return True if benchmark reaches the target quality criteria.
        """
        if self.__counstraints is None:
            return False
        for metric, sub in self.__counstraints.items():
            higher_is_better = metrics_factory().get(metric)[0].HIGHER_IS_BETTER
            benched_metric = benchmark.get(metric).to_dict()
            for submetric, target_value in sub.items():
                benched_value = benched_metric[submetric]
                if higher_is_better and benched_value < target_value:
                    return False
                if not higher_is_better and benched_value > target_value:
                    return False
        return True

    def to_dict(self) -> dict:
        return self.__counstraints

    @staticmethod
    def from_dict(data) -> "Criteria":
        return Criteria(data)
