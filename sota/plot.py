#!/bin/env python3

import os
import json
import shutil
import contextlib
import plotly.io
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs
from airium import Airium
from typing import List, Dict, Tuple, Any, Iterable

from kompil.profile.report import EncodingReport, Curve as ReportCurve
from kompil.utils.numbers import to_scale
from sota.constants import *


COLLAPSIBLE_SCRIPT = """
var coll = document.getElementsByClassName("collapsible");
var i;

for (i = 0; i < coll.length; i++) {
  coll[i].addEventListener("click", function() {
    this.classList.toggle("active");
    var content = this.nextElementSibling;
    if (content.style.display === "block") {
      content.style.display = "none";
    } else {
      content.style.display = "block";
    }
  });
}
"""
COLLAPSIBLE_STYLE = """
.collapsible {
  background-color: #777;
  color: white;
  cursor: pointer;
  padding: 12px;
  width: 100%;
  border: none;
  text-align: left;
  outline: none;
  font-size: 15px;
}

.active, .collapsible:hover {
  background-color: #555;
}

.content {
  padding: 10px 10px;
  display: none;
  overflow: hidden;
  background-color: #ffffff;
  width: 100%;
}
"""
TOGSEL_SCRIPT = """
function togselSet(id, btns, divs) {

  function togselSetBtn(btn) {
    var elem = document.getElementById(btn);
    elem.classList.add("togsel_on");
    elem.classList.remove("togsel_off");
  }

  function togselUnsetBtn(btn) {
    var elem = document.getElementById(btn);
    elem.classList.remove("togsel_on");
    elem.classList.add("togsel_off");
  }

  function togselSetDiv(div) {
    var elem = document.getElementById(div);
    elem.style.display = "block";
  }

  function togselUnsetDiv(div) {
    var elem = document.getElementById(div);
    elem.style.display = "none";
  }

  for (i = 0; i < btns.length; i++) {
      togselUnsetBtn(btns[i]);
  }

  for (i = 0; i < divs.length; i++) {
      togselUnsetDiv(divs[i]);
  }
  togselSetBtn(btns[id]);
  togselSetDiv(divs[id]);
}

"""
TOGSEL_STYLE = """
.togsel_on, .togsel_off {
  color: white;
  cursor: pointer;
  padding: 12px;
  border: none;
  text-align: center;
  font-size: 15px;
}

.togsel_on {
  background-color: #152054;
}

.togsel_off {
  background-color: #777;
}

.togsel_off:hover {
  background-color: #213487;
}
"""


__TOGSEL_ID = 0


def togsel(a, *args):
    global __TOGSEL_ID
    baseid = int(__TOGSEL_ID)
    __TOGSEL_ID += 1

    btns = [f"togsel_btn_{baseid}_{sel}" for sel in args]
    divs = [f"togsel_div_{baseid}_{sel}" for sel in args]

    # Buttons
    for i, sel in enumerate(args):
        onclick_text = f"togselSet({i}, {btns}, {divs})"
        with a.button(onclick=onclick_text, id=btns[i], **{"class": "togsel_off"}):
            a(sel)

    # Divs
    for i, sel in enumerate(args):
        with a.div(id=divs[i], style="display:none"):
            yield a, sel


class Standards:
    class Std:
        class Video:
            class Quality:
                @staticmethod
                def fint_ext(path: str, extensions: List[str]) -> str:
                    for fname in os.listdir(path):
                        for ext in extensions:
                            if fname.endswith("." + ext):
                                return os.path.join(path, fname)
                    raise RuntimeError(f"{ext} extension not found in {path}")

                def __init__(self, name: str, path: str) -> None:
                    self.name = name
                    self.path = path
                    # File weight
                    exts = ["mp4", "webm"]
                    encoded_path = self.fint_ext(path, exts)
                    self.weight = os.path.getsize(encoded_path)
                    # Eval data
                    eval_path = self.fint_ext(path, ["json"])
                    with open(eval_path, "r") as f:
                        self.eval = json.load(f)

            def __init__(self, name: str, path: str):
                self.name = name
                self.qualities = []
                for quality_name in os.listdir(path):
                    quality_path = os.path.join(path, quality_name)
                    self.qualities.append(self.Quality(quality_name, quality_path))
                # Sort by weight
                self.qualities.sort(key=lambda q: q.weight)

        def __init__(self, name: str, path: str):
            self.name = name
            self.video_runs = {}
            for video_name in os.listdir(path):
                self.video_runs[video_name] = self.Video(video_name, os.path.join(path, video_name))

        def __getitem__(self, name: str) -> Video:
            return self.video_runs[name]

    def __init__(self, dpath: str):
        self.standard_runs = {}
        for standard in os.listdir(dpath):
            self.standard_runs[standard] = self.Std(standard, os.path.join(dpath, standard))

    def __getitem__(self, name: str) -> Std:
        return self.standard_runs[name]


class Report:
    def __init__(self, dpath: str, group: str, name: str):
        self.group = group
        self.name = name
        with open(os.path.join(dpath, FILENAME_EVAL_QUANT), "r") as f:
            self.eval_quant = json.load(f)
        with open(os.path.join(dpath, FILENAME_PACKER_INFO), "r") as f:
            self.packer_info = json.load(f)
        with open(os.path.join(dpath, FILENAME_EVAL), "r") as f:
            self.eval = json.load(f)
        self.file_size = os.path.getsize(os.path.join(dpath, FILENAME_PACKED_MODEL))
        self.encoding = EncodingReport.read(dpath)


def find_reports() -> Iterable[str]:
    for group in os.listdir(SOTA_DATA_PATH):
        group_dpath = os.path.join(SOTA_DATA_PATH, group)
        if not os.path.isdir(group_dpath):
            continue
        for name in os.listdir(group_dpath):
            dpath = os.path.join(group_dpath, name)
            if not os.path.isdir(dpath):
                continue
            yield dpath, group, name


def load_reports() -> List[Report]:
    return [Report(dpath, group, name) for dpath, group, name in find_reports()]


def check_report_compat(reports: List[Report]):
    # Utils
    def test_similar(name: str, getval):
        val = getval(reports[0].encoding)
        for report in reports:
            assert val == getval(
                report.encoding
            ), f"{name} variable is not the same for every reports"

    # Test data is present
    for report in reports:
        assert report.eval, "Report must be evaluated before used here"
        assert "vmaf" in report.eval, "Report evaluation must have VMAF option enabled"

    test_similar("video", lambda r: r.video_name)


def sort_reports_by_params(reports: List[Report]):
    reports.sort(key=lambda r: r.encoding.model_nb_parameters)


def sort_reports_by_groups(reports: List[Report]) -> Dict[str, Report]:
    output = dict()
    for report in reports:
        if report.group not in output:
            output[report.group] = list()
        output[report.group].append(report)
    return output


def get_reports_differencies(reports: List[Report]):
    # Utils
    def _is_similar(_getval):
        val = _getval(reports[0])
        for report in reports[1:]:
            if val != _getval(report):
                return False
        return True

    hyperparameters = [
        ("epochs", lambda r: r.encoding.epochs),
        ("video", lambda r: r.encoding.video_name),
        ("scheduler", lambda r: r.encoding.training_params["scheduler"]),
        ("scheduler params", lambda r: r.encoding.training_params["scheduler_params"]),
        ("loss", lambda r: r.encoding.training_params["loss"]),
        ("loss params", lambda r: r.encoding.training_params["loss_params"]),
        ("optimizer", lambda r: r.encoding.training_params["optimizer"]),
        ("learning rate", lambda r: r.encoding.training_params["optimizer_params"]["lr"]),
        ("batch size", lambda r: r.encoding.batch_size),
        ("video", lambda r: r.encoding.video_name),
        ("colorspace", lambda r: r.encoding.model_params["colorspace"]),
        ("fine tuning", lambda r: r.encoding.fine_tuning),
        ("quality metric", lambda r: r.encoding.quality_metric),
        ("packer", lambda r: r.packer_info),
        ("gradient clipping", lambda r: r.encoding.grad_clipping),
        ("accumulate batches", lambda r: r.encoding.batch_acc),
    ]

    # Check some data are similar
    similar = []
    unsimilar = []
    for parameter, get_val in hyperparameters:
        if _is_similar(get_val):
            similar.append((parameter, get_val))
        else:
            unsimilar.append((parameter, get_val))

    return similar, unsimilar


@contextlib.contextmanager
def html_page(a: Airium, reports_by_groups: Dict[str, List[Report]], sel_group=None) -> Airium:
    fbg = "background-color: #CCCCCC"

    a("<!DOCTYPE html>")
    with a.html(lang="pl"):
        with a.head():
            a.meta(charset="utf-8")
            a.title(_t="Kompil encoding: State Of The Art")
            with a.style():
                a(COLLAPSIBLE_STYLE)
                a(TOGSEL_STYLE)
            with a.script(type="text/javascript"):
                a(get_plotlyjs())

        with a.body():
            with a.table(style="width:100%"):
                with a.tr():
                    with a.td(valign="top", style="min-width:150px"):

                        with a.a(href=f"index.html"):
                            with a.font(style=fbg if sel_group is None else ""):
                                a("Index")
                        a(" | ")

                        for group in reports_by_groups.keys():
                            with a.a(href=f"video_{group}.html"):
                                with a.font(style=fbg if sel_group == group else ""):
                                    a(group)
                            a(" | ")

                with a.tr():
                    with a.td(valign="top"):
                        yield

            with a.script():
                a(COLLAPSIBLE_SCRIPT)
                a(TOGSEL_SCRIPT)


@contextlib.contextmanager
def js_collapsible_button(a, title):
    with a.button(**{"type": "button", "class": "collapsible"}):
        a(title)
    with a.div(**{"class": "content"}):
        yield a


def metric_per_params_figure(reports: List[Report], metric: str) -> go.Figure:

    nb_param = [report.encoding.model_nb_parameters for report in reports]
    min_metric = [report.eval[metric]["min"] for report in reports]
    avg_metric = [report.eval[metric]["mean"] for report in reports]
    max_metric = [report.eval[metric]["max"] for report in reports]

    fig = go.Figure()

    fig.update_layout(
        autosize=False,
        width=480,
        height=320 + 60,
        margin=dict(l=10, r=10, b=30, t=30, pad=0),
        hovermode="x",
        xaxis=dict(range=[0, max(nb_param) * 1.2]),
        yaxis=dict(range=[min(min_metric) * 0.8, max(max_metric) * 1.2]),
    )

    fig.add_trace(go.Scatter(x=nb_param, y=min_metric, mode="lines+markers", name="min " + metric))
    fig.add_trace(go.Scatter(x=nb_param, y=avg_metric, mode="lines+markers", name="avg " + metric))
    fig.add_trace(go.Scatter(x=nb_param, y=max_metric, mode="lines+markers", name="max " + metric))
    return fig


def metric_per_mbytes_figure(
    reports: List[Report],
    metric: str,
    standards: Standards,
    group: str,
) -> go.Figure:

    nb_param = [report.file_size / 1024 / 1024 for report in reports]
    min_metric = [report.eval_quant[metric]["min"] for report in reports]
    avg_metric = [report.eval_quant[metric]["mean"] for report in reports]
    max_metric = [report.eval_quant[metric]["max"] for report in reports]

    fig = go.Figure()

    fig.update_layout(
        autosize=False,
        width=480,
        height=320 + 60,
        margin=dict(l=10, r=10, b=30, t=30, pad=0),
        hovermode="x",
        xaxis=dict(range=[0, max(nb_param) * 1.2]),
        yaxis=dict(range=[min(min_metric) * 0.8, max(max_metric) * 1.2]),
    )

    options = dict(mode="lines+markers")
    options_hide = dict(mode="lines+markers", visible="legendonly")

    fig.add_trace(go.Scatter(x=nb_param, y=min_metric, name="min " + metric, **options_hide))
    fig.add_trace(go.Scatter(x=nb_param, y=avg_metric, name="avg " + metric, **options))
    fig.add_trace(go.Scatter(x=nb_param, y=max_metric, name="max " + metric, **options_hide))

    # AVC
    avc_list = standards["AVC"][group].qualities
    avc_weight = [quality.weight / 1024 / 1024 for quality in avc_list]
    avc_metric = [quality.eval[metric]["mean"] for quality in avc_list]
    fig.add_trace(go.Scatter(x=avc_weight, y=avc_metric, name="avg AVC", **options_hide))

    # VP9
    vp9_list = standards["VP9"][group].qualities
    vp9_weight = [quality.weight / 1024 / 1024 for quality in vp9_list]
    vp9_metric = [quality.eval[metric]["mean"] for quality in vp9_list]
    fig.add_trace(go.Scatter(x=vp9_weight, y=vp9_metric, name="avg VP9", **options_hide))

    return fig


def figure_metric_per_frame_compare(reports: List[Report], metric: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        margin=dict(l=0, r=0, b=30, t=30, pad=0),
        hovermode="x",
        height=350,
        title=dict(text=f"<b>{metric}/frames"),
    )
    for i, report in enumerate(reports):
        data = report.eval[metric]["data"]
        fig.add_trace(go.Scatter(x=list(range(len(data))), y=data, mode="lines", name=report.name))
    return fig


def figure_quant_metric_per_frame_compare(reports: List[Report], metric: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        margin=dict(l=0, r=0, b=30, t=30, pad=0),
        hovermode="x",
        height=350,
        title=dict(text=f"<b>{metric}/frames"),
    )
    for i, report in enumerate(reports):
        data = report.eval_quant[metric]["data"]
        fig.add_trace(go.Scatter(x=list(range(len(data))), y=data, mode="lines", name=report.name))
    return fig


def figure_metric_per_epoch_compare(reports: List[Report], metric: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        margin=dict(l=0, r=0, b=30, t=30, pad=0),
        hovermode="x",
        height=350,
        title=dict(text=f"<b>{metric}/epochs"),
    )

    for i, report in enumerate(reports):
        curve = report.encoding.metric_curves[metric]
        curve_t = [[row[h] for row in curve] for h in range(len(curve.headers))]

        fig.add_trace(
            go.Scatter(
                x=curve_t[0],
                y=curve_t[3],
                showlegend=True,
                name=f"{report.name} avg",
                visible=True if i == 0 else "legendonly",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=curve_t[0] + curve_t[0][::-1],
                y=curve_t[1] + curve_t[2][::-1],
                fill="toself",
                line_color="rgba(255,255,255,0)",
                showlegend=True,
                name=f"{report.name} min/max",
                visible="legendonly",
            )
        )
    return fig


def video_diff_hp(a: Airium, reports: List[Report], unsimilar) -> Airium:

    tdstyle = "padding:5px;"

    with a.table(border=1, style="min-width:300px;border-collapse:collapse;text-align:center"):

        with a.tr():
            a.th(_t="", style=tdstyle)
            a.th(_t=f"parameters", style=tdstyle)
            for key, _ in unsimilar:
                a.th(_t=f"{key}", style=tdstyle)

        for report in reports:
            with a.tr():
                a.th(_t=report.name, style=tdstyle)
                a.td(_t=to_scale(report.encoding.model_nb_parameters), style=tdstyle)
                for _, getval_cb in unsimilar:
                    a.td(_t=str(getval_cb(report)), style=tdstyle)

    return a


def video_common_hp(a: Airium, reports: List[Report], similar) -> Airium:

    for key, getval in similar:
        a(f"<b>{key}:</b> {getval(reports[0])}<br>")

    return a


def _all_eval_metrics(reports: List[Report]) -> List[str]:
    metrics = set()
    for report in reports:
        for metric, _ in report.eval.items():
            metrics.add(metric)
    return list(metrics)


def _all_curved_metrics(reports: List[Report]) -> List[str]:
    metrics = set()
    for report in reports:
        for metric, _ in report.encoding.metric_curves.items():
            metrics.add(metric)
    return list(metrics)


def video_page(a: Airium, reports: List[Report], standards: Standards, group: str) -> Airium:
    first_report = reports[0].encoding
    eval_metrics = _all_eval_metrics(reports)
    eval_metrics.sort()
    curved_metrics = _all_curved_metrics(reports)
    curved_metrics.sort()

    with a.h2():
        a("Comparaison for video " + first_report.video_name)

    similar, unsimilar = get_reports_differencies(reports)

    with js_collapsible_button(a, "Hyperparameters"):
        with a.h3():
            a("Common")
        video_common_hp(a, reports, similar)
        with a.h3():
            a("Differents")
        video_diff_hp(a, reports, unsimilar)

    a.br()
    a.br()

    with js_collapsible_button(a, "Training"):
        for a, metric in togsel(a, *curved_metrics):
            a.br()
            a(to_html_figure(figure_metric_per_epoch_compare(reports, metric)))

    a.br()
    a.br()

    with js_collapsible_button(a, "Post-training results"):

        for a, metric in togsel(a, *eval_metrics):
            a.br()

            with a.table(border=1, style="border-collapse: collapse"):
                with a.tr():
                    a.th(_t=f"{metric}/number of parameters")
                    a.th(_t=f"{metric}/frame")
                with a.tr():
                    with a.td():
                        a(to_html_figure(metric_per_params_figure(reports, metric)))
                    with a.td():
                        a(to_html_figure(figure_metric_per_frame_compare(reports, metric)))

    a.br()
    a.br()

    with js_collapsible_button(a, "Post-packer results"):

        for a, metric in togsel(a, *eval_metrics):
            a.br()

            with a.table(border=1, style="border-collapse: collapse"):
                with a.tr():
                    a.th(_t=f"{metric}/MB")
                    a.th(_t=f"{metric}/frame")
                with a.tr():
                    with a.td():
                        a(
                            to_html_figure(
                                metric_per_mbytes_figure(reports, metric, standards, group)
                            )
                        )
                    with a.td():
                        a(to_html_figure(figure_quant_metric_per_frame_compare(reports, metric)))

    return a


def to_html_figure(fig):
    html = plotly.io.to_html(
        fig,
        config={"displaylogo": False},
        full_html=False,
        include_plotlyjs=False,
    )
    return html


def main():
    all_reports = load_reports()
    standards = Standards(SOTA_STD_PATH)

    if os.path.exists(SOTA_HTML_PATH):
        shutil.rmtree(SOTA_HTML_PATH)
    os.makedirs(SOTA_HTML_PATH, exist_ok=True)

    reports_by_groups = sort_reports_by_groups(all_reports)

    with open(os.path.join(SOTA_HTML_PATH, "index.html"), mode="w+") as f:
        a = Airium()
        with html_page(a, reports_by_groups):
            a("Index")
        f.write(str(a))
        print("Written index in", f.name)

    for group, reports in reports_by_groups.items():
        sort_reports_by_params(reports)
        check_report_compat(reports)
        with open(os.path.join(SOTA_HTML_PATH, f"video_{group}.html"), mode="w+") as f:
            a = Airium()
            with html_page(a, reports_by_groups, sel_group=group):
                video_page(a, reports, standards, group)
            f.write(str(a))


if __name__ == "__main__":
    main()
