import os
import csv
import json
import math

from typing import List, Union, Optional, Dict, Any, Tuple

import kompil.utils.git as git

from kompil.utils.time import now, to_str
from kompil.utils.numbers import to_scale
from kompil.profile.bench import Benchmark
from kompil.profile.criteria import Criteria

# Report version is to ensure compatibility. If the version don't match, either we can migrate the
# report to the new format, either we can checkout to the commit where the version was still active
# Note: This version represents only how the report is formatted and not how the read/write is
# implemented
_REPORT_VERSION = 0

SHORT_REPORT = """
time: {timestamp}
name: {name}
model:
{model}
total parameters: {param_count}
benchmark:
{benchmark}
"""


def build_git_info() -> Dict[str, str]:
    """
    Tool to build the git dictionary info for the report
    """
    current_commit = git.current_commit()
    closest_main_commit = git.find_closest_main_commit()

    return {
        "current_commit": current_commit,
        "closest_main_commit": closest_main_commit,
        "diff_from_main_branch": git.diff(base_commit=closest_main_commit),
        "diff": git.diff(),
    }


class Curve:
    __TYPE_TO_STR = {
        int: "int",
        float: "float",
        str: "str",
    }
    __STR_TO_TYPE = {
        "int": int,
        "float": float,
        "str": str,
    }

    def __init__(self, *headers: List[Union[str, type]]):
        self.__headers = headers
        self.__rows = []

    def append(self, *values: List[float]):
        assert len(values) == len(self.__headers)
        self.__rows.append(values)

    def __len__(self):
        return len(self.__rows)

    def __getitem__(self, idx: int) -> List[float]:
        return self.__rows[idx]

    @property
    def headers(self):
        return self.__headers

    def write_csv(self, *file_path: List[str]):
        with open(os.path.join(*file_path), "w") as csvfile:
            writer = csv.writer(csvfile, delimiter=";", quotechar="|", quoting=csv.QUOTE_MINIMAL)
            writer.writerow([name for name, _ in self.__headers])
            writer.writerow([Curve.__TYPE_TO_STR[typ] for _, typ in self.__headers])
            for values in iter(self):
                writer.writerow(values)

    @staticmethod
    def read_csv(*file_path: List[str]):
        with open(os.path.join(*file_path)) as csvfile:
            reader = csv.reader(csvfile, delimiter=";", quotechar="|", quoting=csv.QUOTE_MINIMAL)

            rows = list(reader)

            row_names = rows[0]
            row_types = rows[1]

            header = [(row_names[i], row_types[i]) for i in range(len(row_names))]
            curve = Curve(*header)

            for row in rows[2:]:
                row = [curve.__STR_TO_TYPE[row_types[i]](value) for i, value in enumerate(row)]
                curve.append(*row)

            return curve


class EncodingReport:
    def __init__(
        self,
        timestamp: str,
        model_name: str,
        model_params: dict,
        model_nb_parameters: int,
        model_extra: List[Union[float, int]],
        model_pattern: str,
        training_params: dict,
        criteria: Criteria,
        status: str,
        compute_time: float,
        benchmark: Optional[Benchmark],
        epochs: int,
        mac_address: str,
        command_line: str,
        original_video_path: str,
        metric_curves: Dict[str, Curve],
        learning_rate_curve: Curve,
        model_resume: str,
        git_info: Dict[str, str],
        batch_size: int,
        video_name: str,
        take_best: bool,
        quality_metric: str,
        pruning: Union[List[Tuple[int, float]], None] = None,
        fine_tuning: Union[int, None] = None,
        grad_clipping: Union[float, None] = None,
        batch_acc: Union[int, None] = None,
    ):
        self.timestamp = timestamp
        self.model_name = model_name
        self.model_params = model_params
        self.model_nb_parameters = model_nb_parameters
        self.model_extra = model_extra
        self.model_pattern = model_pattern
        self.training_params = training_params
        self.criteria = criteria
        self.status = status
        self.compute_time = compute_time
        self.benchmark = benchmark
        self.epochs = epochs
        self.mac_address = mac_address
        self.command_line = command_line
        self.original_video_path = original_video_path
        self.metric_curves = metric_curves
        self.learning_rate_curve = learning_rate_curve
        self.model_resume = model_resume
        self.git_info = git_info
        self.batch_size = batch_size
        self.video_name = video_name
        self.take_best = take_best
        self.pruning = pruning
        self.fine_tuning = fine_tuning
        self.grad_clipping = grad_clipping
        self.batch_acc = batch_acc
        self.quality_metric = quality_metric

    def to_data(self):
        return {
            "report_version": _REPORT_VERSION,
            "timestamp": self.timestamp,
            "model": {
                "name": self.model_name,
                "params": self.model_params,
                "nb_parameters": self.model_nb_parameters,
                "pattern": self.model_pattern,
                "extra": self.model_extra,
            },
            "training": {
                "params": self.training_params,
                "criteria": self.criteria.to_dict(),
                "batch_size": self.batch_size,
                "video": self.video_name,
                "pruning": self.pruning,
                "fine_tuning": self.fine_tuning,
                "grad_clipping": self.grad_clipping,
                "batch_acc": self.batch_acc,
                "take_best": self.take_best,
                "quality_metric": self.quality_metric,
            },
            "results": {
                "status": self.status,
                "total_compute_time": self.compute_time,
                "benchmark": self.benchmark.to_dict() if self.benchmark else None,
                "epochs": self.epochs,
            },
            "local": {
                "mac_address": self.mac_address,
                "command_line": self.command_line,
                "original_video_path": self.original_video_path,
            },
        }

    @property
    def short(self) -> str:
        return SHORT_REPORT.format(
            timestamp=self.timestamp,
            name=self.model_name,
            model=self.model_resume,
            param_count=self.model_nb_parameters,
            benchmark=self.benchmark.to_table() if self.benchmark else None,
        )

    def __build_folder_path(self):

        ymd = to_str(now(), reduction="ymd")
        hms = to_str(now(), reduction="hms")

        params = to_scale(self.model_nb_parameters)
        compute_time = math.trunc(self.compute_time)

        return os.path.join(
            "reports",
            ymd,
            f"{hms}_{self.model_name}_{params}_{self.epochs}e_{compute_time}s",
        )

    def save_in(self, build_folder: str):
        assert build_folder
        folder_path = os.path.join(build_folder, self.__build_folder_path())

        # Make the report folder
        os.makedirs(folder_path, exist_ok=True)

        self.save_to(folder_path)

        return folder_path

    def save_to(self, folder_path: str):

        # Write data
        with open(os.path.join(folder_path, "data.json"), "w+") as f:
            f.write(json.dumps(self.to_data(), sort_keys=True, indent=4))

        # Write short report
        with open(os.path.join(folder_path, "short.txt"), "w+") as f:
            f.write(self.short)

        # Write curves
        metrics_folder = os.path.join(folder_path, "metrics")
        os.makedirs(metrics_folder, exist_ok=True)
        for curve_name, curve in self.metric_curves.items():
            curve.write_csv(metrics_folder, f"{curve_name}.csv")
        self.learning_rate_curve.write_csv(folder_path, "learning_rate.csv")

        # Write versioning info
        if self.git_info["diff_from_main_branch"]:
            with open(os.path.join(folder_path, "diff_from_main_branch.txt"), "w+") as f:
                f.write(self.git_info["diff_from_main_branch"])

        if self.git_info["diff"]:
            with open(os.path.join(folder_path, "diff.txt"), "w+") as f:
                f.write(self.git_info["diff"])

        versioning_info = {
            "current_commit": self.git_info["current_commit"],
            "closest_main_commit": self.git_info["closest_main_commit"],
        }
        with open(os.path.join(folder_path, "versioning.json"), "w+") as f:
            f.write(json.dumps(versioning_info, sort_keys=True, indent=4))
        return folder_path

    @staticmethod
    def read(folder_path: str):
        # Utils
        def get_old(_data: dict, _key: str):
            class NoKey:
                pass

            value = _data.get(_key, NoKey)
            if value == NoKey:
                print("WARNING: old report version")
                value = None
            return value

        # Check for existing folder
        if not os.path.exists(folder_path) or not os.path.isdir(folder_path):
            raise FileNotFoundError(f"No reports found at {folder_path}")

        # Read the data file
        with open(os.path.join(folder_path, "data.json")) as datafile:
            data = json.load(datafile)

        # First check the report version
        if "report_version" not in data or data["report_version"] != _REPORT_VERSION:
            raise RuntimeError(
                "Report is not a the local version, please checkout to an older commit."
            )

        timestamp = data["timestamp"]

        model_name = data["model"]["name"]
        model_params = data["model"]["params"]
        model_nb_parameters = data["model"]["nb_parameters"]
        model_extra = data["model"]["extra"]
        model_pattern = data["model"]["pattern"]

        training_params = data["training"]["params"]
        data_training = data["training"]
        batch_size = data_training.get("batch_size", None)
        video_name = data_training.get("video", None)
        criteria = Criteria.from_dict(data_training["criteria"])

        pruning = data_training.get("pruning", None)
        fine_tuning = data_training.get("fine_tuning", None)
        if fine_tuning is None:
            fine_tuning = data_training.get("fine_tune", None)
            if fine_tuning is not None:
                print("WARNING: old report version")
        take_best = data_training.get("take_best", "unknown")
        quality_metric = data_training.get("quality_metric", "unknown")
        batch_acc = get_old(data_training, "batch_acc")
        grad_clipping = get_old(data_training, "grad_clipping")

        _data_res = data["results"]
        status = _data_res["status"]
        compute_time = _data_res["total_compute_time"]
        _benchmark_dict = _data_res["benchmark"] if _data_res["benchmark"] is not None else None
        benchmark = Benchmark.from_dict(_benchmark_dict)
        epochs = _data_res["epochs"]

        mac_address = data["local"]["mac_address"]
        command_line = data["local"]["command_line"]
        original_video_path = data["local"]["original_video_path"]

        # Short will not be read, should not be used anyway
        model_resume = ""

        # Read curves
        learning_rate_curve = Curve.read_csv(folder_path, "learning_rate.csv")
        metrics_folder = os.path.join(folder_path, "metrics")
        metric_curves = {}
        if not os.path.exists(metrics_folder):
            # This is for retrocompatibility purposes
            metric_curves["psnr"] = Curve.read_csv(folder_path, "psnr.csv")
            metric_curves["ssim"] = Curve.read_csv(folder_path, "ssim.csv")
            if os.path.exists(os.path.join(folder_path, "vmaf.csv")):
                metric_curves["vmaf"] = Curve.read_csv(folder_path, "vmaf.csv")
        else:
            for file_name in os.listdir(metrics_folder):
                metric_name, metric_ext = os.path.splitext(file_name)
                if metric_ext != ".csv":
                    continue
                metric_curves[metric_name] = Curve.read_csv(metrics_folder, file_name)

        # Read git info
        diff_path = os.path.join(folder_path, "diff.txt")
        diff_main_path = os.path.join(folder_path, "diff_from_main_branch.txt")

        if os.path.exists(diff_main_path):
            with open(diff_main_path) as f:
                _diff_from_main_branch = f.read()
        else:
            _diff_from_main_branch = ""

        if os.path.exists(diff_path):
            with open(diff_path) as f:
                _diff = f.read()
        else:
            _diff = ""

        with open(os.path.join(folder_path, "versioning.json")) as datafile:
            _versioning_info = json.load(datafile)

        git_info = {
            "diff_from_main_branch": _diff_from_main_branch,
            "diff": _diff,
            **_versioning_info,
        }

        # Build report
        return EncodingReport(
            timestamp=timestamp,
            model_name=model_name,
            model_params=model_params,
            model_nb_parameters=model_nb_parameters,
            model_extra=model_extra,
            model_pattern=model_pattern,
            training_params=training_params,
            criteria=criteria,
            status=status,
            compute_time=compute_time,
            benchmark=benchmark,
            epochs=epochs,
            mac_address=mac_address,
            command_line=command_line,
            original_video_path=original_video_path,
            metric_curves=metric_curves,
            learning_rate_curve=learning_rate_curve,
            model_resume=model_resume,
            git_info=git_info,
            batch_size=batch_size,
            quality_metric=quality_metric,
            video_name=video_name,
            pruning=pruning,
            fine_tuning=fine_tuning,
            take_best=take_best,
            batch_acc=batch_acc,
            grad_clipping=grad_clipping,
        )
