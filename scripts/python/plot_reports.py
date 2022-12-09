import plotly.graph_objects as go
import os
import shutil
import fnmatch
import json
import sys


def _get_reports(path: str):
    all_files = os.listdir(path)
    folders = fnmatch.filter(all_files, "report_*")
    reports_json = []

    for foldername in folders:
        if not os.path.isdir(os.path.join(path, foldername)):
            continue
        file_path = os.path.join(path, foldername, "data.json")
        with open(file_path, "r+") as f:
            json_data = json.load(f)
        reports_json.append(json_data)

    return reports_json


def _explode_report(report_json: str):
    model = report_json["model"]
    topology_layers = model["topology"]
    note = report_json["note"]

    nb_nodes = next(layer for layer in topology_layers if layer["type"] == "linear")["output_size"]
    nb_dense_layers = sum(map(lambda layer: layer["type"] == "linear", topology_layers))
    min_loss = note["loss"]["min"]
    min_psnr = note["psnr"]["min"]
    duration = model["nb_frames"]

    return nb_nodes, nb_dense_layers, min_loss, min_psnr, duration


def plot(path_to_reports: str):
    assert path_to_reports

    SYMBOLS = ["square", "x", "circle", "diamond"]

    reports = _get_reports(path_to_reports)
    runs_by_durations = {}
    scatters = []

    # Retrieve each report info and store them by video duration
    for report in reports:
        nb_nodes, nb_layers, min_loss, min_psnr, duration = _explode_report(report)

        if not duration in runs_by_durations:
            run = {}
            run["psnr"] = [min_psnr]
            run["loss"] = [min_loss]
            run["nodes"] = [nb_nodes]
            run["layers"] = [nb_layers]
            runs_by_durations[duration] = run
        else:
            run = runs_by_durations[duration]
            run["psnr"].append(min_psnr)
            run["loss"].append(min_loss)
            run["nodes"].append(nb_nodes)
            run["layers"].append(nb_layers)

    symbol_idx = 0
    for duration_key, duration_val in runs_by_durations.items():
        duration_psnr = duration_val["psnr"]
        duration_layers = duration_val["layers"]
        duration_nodes = duration_val["nodes"]
        scatter = go.Scatter3d(
            name=f"Duration : {duration_key} frames",
            x=duration_layers,
            y=duration_nodes,
            z=duration_psnr,
            mode="markers",
            marker=dict(
                size=8,
                colorscale="speed",  # choose a colorscale
                opacity=0.8,
                symbol=SYMBOLS[symbol_idx],
            ),
        )
        scatters.append(scatter)
        symbol_idx += 1

    layout = go.Layout(
        scene=dict(
            xaxis=dict(title="Nb layers"),
            yaxis=dict(title="Nb nodes"),
            zaxis=dict(title="Min PSNR"),
        )
    )

    fig = go.Figure(data=scatters, layout=layout)
    fig.update_layout(title="PSNR according layers/nodes")
    fig.show()


def main(argv):
    plot(argv[0])


if __name__ == "__main__":
    main(sys.argv[1:])
