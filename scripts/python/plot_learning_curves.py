import csv
import os
import shutil
import fnmatch
import json
import sys
import numpy as np
import plotly.graph_objects as go


def reports_folders(path: str):
    all_files = os.listdir(path)
    folders = fnmatch.filter(all_files, "report_*")

    for foldername in folders:
        if not os.path.isdir(os.path.join(path, foldername)):
            continue
        yield os.path.join(path, foldername)


def _read_curve(report_folder: str):
    epochs = []
    psnr_min = []
    psnr_max = []
    psnr_mean = []
    l1_min = []
    l1_max = []
    l1_mean = []

    file_path = os.path.join(report_folder, "learning_curve.csv")
    with open(file_path, "r") as csvfile:
        spamreader = csv.reader(csvfile, delimiter=";")

        first = True
        for row in spamreader:
            if first:
                first = False
                continue
            epochs.append(int(row[0]))
            psnr_min.append(float(row[1]))
            psnr_max.append(float(row[2]))
            psnr_mean.append(float(row[3]))
            l1_min.append(float(row[4]))
            l1_max.append(float(row[5]))
            l1_mean.append(float(row[6]))

    return epochs, psnr_min, psnr_max, psnr_mean, l1_min, l1_max, l1_mean


def _l1_to_scatters(epochs, psnr_min, psnr_max, psnr_mean, l1_min, l1_max, l1_mean):
    return [
        go.Scatter(name="l1_min", x=epochs, y=l1_min, line=go.Line(color="blue")),
        go.Scatter(name="l1_max", x=epochs, y=l1_max, line=go.Line(color="magenta")),
        go.Scatter(name="l1_mean", x=epochs, y=l1_mean, line=go.Line(color="cyan")),
    ]


def _psnr_to_scatters(epochs, psnr_min, psnr_max, psnr_mean, l1_min, l1_max, l1_mean):
    return [
        go.Scatter(name="psnr_min", x=epochs, y=psnr_min, line=go.Line(color="green")),
        go.Scatter(name="psnr_max", x=epochs, y=psnr_max, line=go.Line(color="gray")),
        go.Scatter(name="psnr_mean", x=epochs, y=psnr_mean, line=go.Line(color="orange")),
    ]


def plot(path_to_reports: str):
    assert path_to_reports

    for report_folder in reports_folders(path_to_reports):

        curves = _read_curve(report_folder)

        scatters = _psnr_to_scatters(*curves)

        layout = go.Layout(
            scene=dict(
                xaxis=dict(title="Epochs"),
                yaxis=dict(title="PSNR"),
            )
        )

        fig = go.Figure(data=scatters, layout=layout)
        fig.update_layout(title=f"PSNR {report_folder}")
        fig.show()


def main(argv):
    plot(argv[0])


if __name__ == "__main__":
    main(sys.argv[1:])
