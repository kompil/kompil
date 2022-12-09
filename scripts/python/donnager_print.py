import os
import json
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Tuple

from kompil.profile.report import EncodingReport


def _list_reports(folder: str):
    for file_name in os.listdir(folder):
        full_path_1 = os.path.join(folder, file_name)
        if not os.path.isdir(full_path_1):
            continue
        for file_name_2 in os.listdir(full_path_1):
            full_path_2 = os.path.join(full_path_1, file_name_2)
            yield full_path_2


def read_report(folder_path: str):

    report = EncodingReport.read(folder_path)

    if report.model_name != "avatar_1080p_20s":
        return None

    nb_parameters = report.model_nb_parameters
    extra = report.model_extra
    psnr_mean = report.benchmark.psnr.mean

    return nb_parameters, extra, psnr_mean


def read_all(folder_path: str):
    data = []
    for report_folder in _list_reports(folder_path):
        item = read_report(report_folder)
        if item is None:
            continue
        data.append(item)
    return data


def hyperparams_figure(data):
    n_hp = len(data[0][1])

    # Prep data
    colnames = [*[f"lin{i}" for i in range(1, n_hp - 8)], *[f"deconv{i}" for i in range(1, 10)]]

    dimensions = []
    for i in range(n_hp):
        values = [item[1][i] for item in data]
        dimensions.append(dict(range=[0, max(values)], label=colnames[i], values=values))

    weight = [item[0] for item in data]
    dimensions.append(dict(range=[min(weight), max(weight)], label="weight", values=weight))

    psnr_mean = [item[2] for item in data]
    dimensions.append(
        dict(range=[min(psnr_mean), max(psnr_mean)], label="psnr_mean", values=psnr_mean)
    )

    return go.Parcoords(dimensions=dimensions)


g = hyperparams_figure(read_all("build/report_bench"))
fig = go.Figure(data=[g])
fig.show()
