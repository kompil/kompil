#!/bin/env python3

import os
import json
import copy
import shutil
import argparse
import argcomplete
from typing import List, Dict, Any, Union, Optional

from kompil.cli_defaults import EncodingDefaults
from sota.constants import *


KOMPIL_PY_SCRIPT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
    "kompil.py",
)
ELEMENTS_PATHS = os.path.join(os.path.dirname(os.path.realpath(__file__)), "elements")

Variables = Union[Dict[str, Any], None]


def shell(args: str):
    assert args
    print("Running >>> ", args)
    os.system(args)


def kompil(args: str):
    assert args

    print("Running >>> kompil", args)

    os.system(f"python3 {KOMPIL_PY_SCRIPT} {args}")


def __add_single_opt(data: dict, name: str, cli: str):
    val = data.get(name, None)
    if val is None:
        return str()
    return f"{cli} {val} "


def __add_list_opt(data: dict, name: str, cli: str):
    val = data.get(name, None)
    if val is None:
        return str()
    opt_list = " ".join(str(elem) for elem in val)
    return f"{cli} {opt_list} "


def __add_boolean_opt(data: dict, name: str, cli: str):
    val = data.get(name, False)
    if not val:
        return str()
    return cli + " "


def __extract_params(data: dict, exclude: List[str], variables: Variables = None):
    params = []
    for key, value in data.items():
        if key in exclude:
            continue
        if variables is not None and isinstance(value, str):
            value = variables.get(value, value)
        params.append(key)
        params.append(str(value))
    return params


def __add_loss_opt(data: dict, variables: Variables):
    if "loss" not in data:
        return str()

    if isinstance(data["loss"], str):
        return f"--loss {data['loss']} "

    options = str()
    data_loss = data["loss"]

    if "name" in data_loss:
        options += f"--loss {data_loss['name']} "

    params = __extract_params(data_loss, ["name"], variables)
    if params:
        options += f"--params-loss {' '.join(params)} "

    return options


def __add_scheduler_opt(data: dict):
    if "scheduler" not in data:
        return str()

    if isinstance(data["scheduler"], str):
        return f"--scheduler {data['scheduler']} "

    options = str()
    data_scheduler = data["scheduler"]

    if "name" in data_scheduler:
        options += f"--scheduler {data_scheduler['name']} "
    if "learning_rate" in data_scheduler:
        options += f"--learning-rate {data_scheduler['learning_rate']} "

    params = __extract_params(data_scheduler, ["name", "learning_rate"])
    if params:
        options += f"--params-scheduler {' '.join(params)} "

    return options


def dict_to_encode_options(data: dict, variables: Variables):
    options = str()
    data_tr = data.get("training", {})
    # Standards
    options += __add_single_opt(data, "resolution", "--resolution")
    options += __add_single_opt(data, "colorspace", "--colorspace")
    # Training std
    options += __add_single_opt(data_tr, "precision", "--precision")
    options += __add_single_opt(data_tr, "batch_size", "--batch-size")
    options += __add_single_opt(data_tr, "max_epochs", "--autoquit-epoch")
    options += __add_single_opt(data_tr, "optimizer", "--optimizer")
    options += __add_single_opt(data_tr, "fine_tuning", "--fine-tuning")
    options += __add_single_opt(data_tr, "gradient_clipping", "--gradient-clipping")
    options += __add_single_opt(data_tr, "accumulate_batches", "--accumulate-batches")
    options += __add_boolean_opt(data_tr, "take_best", "--take-best")
    options += __add_loss_opt(data_tr, variables)
    options += __add_scheduler_opt(data_tr)
    # Topology
    data_topo = data_tr.get("topology", None)
    if data_topo is not None:
        options += __add_single_opt(data_topo, "builder", "--topology-builder")
        options += __add_list_opt(data_topo, "parameters", "-x")
    # Metrics
    options += __add_single_opt(data, "quality_metric", "--quality-metric")
    quality_metric = data.get("quality_metric", None)
    eval_metrics = set(EncodingDefaults.EVAL_METRICS)
    if quality_metric is not None:
        eval_metrics.add(quality_metric)
    options += "--eval-metrics " + " ".join(str(elem) for elem in eval_metrics) + " "
    return options


def run_maskmakers(data: dict, tmp_folder: str):
    options_gen = str()
    options_gen += __add_single_opt(data, "resolution", "--resolution")
    options_gen += __add_single_opt(data, "colorspace", "--colorspace")

    video = data["video"]

    variables = {}

    for data_mm in data.get("maskmaker", []):
        options = copy.copy(options_gen)

        params = __extract_params(data_mm, ["id", "type"])
        if params:
            options += f"--params {' '.join(params)} "

        mask_id = data_mm["id"]
        mask_type = data_mm["type"]

        path = os.path.join(tmp_folder, f"{mask_id}.pth")

        kompil(f"maskmaker build {mask_type} {video} {options} -o {path}")

        variables[f"$mask:{mask_id}"] = path

    return variables


def run_encode(data: dict, report_path: str, mock: bool, no_models_on_ram: bool):
    # To options
    video = data["video"]
    # Maskmakers
    tmp_mask_folders = os.path.join("/dev/shm", "kompil_masks")
    if os.path.exists(tmp_mask_folders):
        shutil.rmtree(tmp_mask_folders)
    os.makedirs(tmp_mask_folders, exist_ok=True)
    variables = run_maskmakers(data, tmp_mask_folders)
    # Mock
    if mock:
        data["training"]["max_epochs"] = 3
    # Encode
    encode_options = dict_to_encode_options(data, variables)
    if no_models_on_ram:
        encode_options += " --no-models-on-ram"
    kompil(f"encode {video} {encode_options} --report-path {report_path}")
    # Eval
    kompil(f"eval {report_path}/{FILENAME_MODEL} {video} -o {report_path}/{FILENAME_EVAL}")


def run_quantization(data: dict, report_path: str):
    quantization = data.get("quantization", dict(blacklist=None, method="cypher"))
    method = quantization.get("method")
    video = data["video"]
    model_path = f"{report_path}/{FILENAME_MODEL}"
    qmodel_path = f"{report_path}/{FILENAME_QUANTIZED_MODEL}"

    kompil(f"quantize {model_path} {qmodel_path} -m {method}")
    kompil(f"eval {qmodel_path} {video} -o {report_path}/{FILENAME_EVAL_QUANT} --device cpu --fp32")


def run_packer(data: dict, report_path: str):
    packer = data.get("packer", "foo")
    model_path = f"{report_path}/{FILENAME_QUANTIZED_MODEL}"
    kompil(f"packer pack {model_path} {report_path}/packed_model.pth -p {packer}")
    shell(f"7z a {report_path}/{FILENAME_PACKED_MODEL} {report_path}/packed_model.pth")
    shell(f"rm {report_path}/packed_model.pth")

    with open(os.path.join(report_path, FILENAME_PACKER_INFO), "w+") as f:
        f.write(json.dumps(packer, indent=4))


def _build_report_path(data: dict, suffix: Optional[str] = None) -> str:
    folder = data["group"]
    run_name = data["name"]
    if suffix is not None:
        run_name += suffix
    return os.path.join(SOTA_DATA_PATH, folder, run_name)


def run_one(
    fpath: str, mock: bool = False, suffix: Optional[str] = None, no_models_on_ram: bool = False
):
    # Open file
    with open(fpath, "r") as f:
        data = json.load(f)
    # Run
    report_path = _build_report_path(data, suffix)
    run_encode(data, report_path, mock, no_models_on_ram)
    run_quantization(data, report_path)
    run_packer(data, report_path)


def main():
    parser = argparse.ArgumentParser(description="Entry point for every kompil commands.")
    parser.set_defaults(func=lambda _: parser.print_help())
    parser.add_argument("file", type=str)
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run with only 3 epochs for test purposes",
    )
    parser.add_argument("--suffix", type=str, help="Register the run name adding a suffix")
    parser.add_argument("--no-models-on-ram", action="store_true")
    argcomplete.autocomplete(parser)
    args = parser.parse_args()

    run_one(os.path.join(ELEMENTS_PATHS, args.file), args.mock, args.suffix, args.no_models_on_ram)


if __name__ == "__main__":
    main()
