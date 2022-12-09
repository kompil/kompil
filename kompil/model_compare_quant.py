import torch
import contextlib
import itertools
import math
import plotly.io
import plotly.express as px

from airium import Airium
from dataclasses import dataclass
from os.path import getsize
from plotly.offline import get_plotlyjs
from typing import List, Dict, Tuple, Union
from torch.nn import Module
from torch.nn.quantized import FloatFunctional, QFunctional
from torch.quantization.stubs import QuantWrapper

from kompil.nn.models.model import VideoNet, model_load
from kompil.nn.layers.quantized import Quantize, DeQuantize


__COLLAPSIBLE_SCRIPT = """
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
}"""
__COLLAPSIBLE_STYLE = """
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

__PADDING_CSS = "padding:5px;"
__BACKGROUND_CSS = "background-color:"
__HIGHLIGHT_CSS = f"{__BACKGROUND_CSS}#a9a7764d;"

__MAX_DISPLAYED = 10
__MAX_SAMPLED_HIST = 5000

__PER_CHANNEL_CONFS = [
    torch.per_channel_affine,
    torch.per_channel_symmetric,
    torch.per_channel_affine_float_qparams,
]


@dataclass
class Layer:
    input: torch.Tensor
    output: torch.Tensor
    name: str
    module: Module

    def __init__(self, input, output, module, name):
        self.input = input
        self.output = output
        self.module = module
        self.name = name

    @property
    def weight(self):
        return _get_weight(self.module)


def _val_to_colorhex(val: float) -> str:
    base = int(0x53D625)
    scaled = val * 50000
    scaled = 128 if math.isnan(scaled) else min(int(scaled), 128)
    val = scaled * 0x00FF00
    colorhex = hex(base + int(val))
    return colorhex


def _background(val: float) -> str:
    colorhex = _val_to_colorhex(val)
    colorstr = colorhex.replace("0x", "#")
    return f"{colorstr};"


def _round(val) -> float:
    val = float(val)
    return round(val, 5) if val is not None and (val > 1e-5 or val < -1e-5) else "0"


def _x_or_round(val) -> Union[float, str]:
    return _round(val) if val is not None else "X"


def _figure_to_html(fig):
    html = plotly.io.to_html(
        fig,
        config={"displaylogo": False},
        full_html=False,
        include_plotlyjs=False,
    )
    return html


def _addition_quant_per_tensor(tensor1: torch.Tensor, tensor2: torch.Tensor) -> torch.Tensor:
    from torch._ops import ops

    assert tensor1.qscheme() == tensor2.qscheme()

    scale, zp = _get_scale_and_zp(tensor1, keep_size=False)

    qadd = ops.quantized.add(tensor1, tensor2, scale.item(), zp.long().item())

    return qadd


def _flatten_module(
    module: torch.nn.Module, this_name: str
) -> Tuple[List[torch.nn.Module], List[str]]:
    flatt_children = []
    flatt_children_names = []

    if not isinstance(module, (QuantWrapper, torch.nn.Sequential)):
        flatt_children.append(module)
        flatt_children_names.append(this_name)

    children = list(module.children())
    if children == []:
        return flatt_children, flatt_children_names

    # Add every children
    for child_name, child in module.named_children():
        # But do not considere Quantized specific layers
        if isinstance(
            child,
            (torch.nn.quantized.modules.linear.LinearPackedParams, FloatFunctional, QFunctional),
        ):
            continue
        child_full_name = this_name + "." + child_name
        flatt_child, flatt_child_names = _flatten_module(child, child_full_name)
        flatt_children.extend(flatt_child)
        flatt_children_names.extend(flatt_child_names)

    return flatt_children, flatt_children_names


# TODO: This should be like this but the quantization methods prevent to define a relation
# between modules because the topology change.
# def _read_weights(module: torch.nn.Module, this_name: str) -> Dict[str, torch.nn.Module]:
#     weights = {}

#     this_weight = _get_weight(module)

#     if this_weight is not None:
#         weights[this_name] = this_weight

#     for child_name, child in module.named_children():
#         child_full_name = this_name + "." + child_name
#         child_weights = _read_weights(child, child_full_name)
#         weights.update(child_weights)

#     return weights


def _get_weight(module: Module) -> torch.Tensor:
    if hasattr(module, "weight"):
        if callable(module.weight):
            return module.weight()
        else:
            return module.weight
    # Because of module pish / oops !
    # TODO: remove after reworking pish
    elif hasattr(module, "_params"):
        return module._params

    return None


def _sample_tensor(tensor: torch.Tensor, max_sample: int) -> Tuple[List, List]:
    flat = torch.flatten(tensor)
    if tensor.numel() < max_sample:
        split = 1
    else:
        split = max(1, tensor.numel() // max_sample)
    idx = torch.arange(0, tensor.numel() - 1, split)
    crop = flat.index_select(dim=0, index=idx)

    return crop.tolist(), idx.tolist()


def _analyse_tensor(
    tensor: torch.Tensor,
) -> Tuple[torch.Tensor, float, float, float, float]:
    values = torch.flatten(tensor)
    min = torch.min(tensor).item()
    max = torch.max(tensor).item()
    std = torch.std(tensor).item() if tensor.dtype in [torch.half, torch.float] else None
    mean = torch.mean(tensor).item() if tensor.dtype in [torch.half, torch.float] else None

    return values, min, max, mean, std


def _compare_tensors(tensors: List[torch.Tensor]) -> Dict:
    comp = {}

    for idx, t in enumerate(tensors):
        diff_dict = {}

        for sidx, pair in enumerate(itertools.combinations(tensors, 2)):
            if not any([(t == t_).all() for t_ in pair]):
                continue

            _, min, max, mean, _ = _analyse_tensor(torch.abs(pair[0] - pair[1]))

            diff_dict[sidx] = {"mean": mean, "min": min, "max": max}

        values, min, max, mean, std = _analyse_tensor(t)
        comp[idx] = {
            "values": values,
            "min": min,
            "max": max,
            "mean": mean,
            "std": std,
            "diff": diff_dict,
        }

    return comp


@contextlib.contextmanager
def _generate_html_button(a, title, collapsible=False):
    if collapsible:
        with a.button(**{"type": "button", "class": "collapsible"}):
            a(title)
    else:
        with a.button(**{"type": "button"}):
            a(title)
    with a.div(**{"class": "content"}):
        yield a


@contextlib.contextmanager
def _generate_html_page(a: Airium) -> Airium:
    a("<!DOCTYPE html>")
    with a.html(lang="en"):
        with a.head():
            a.meta(charset="utf-8")
            a.title(_t="Model and quantized model comparison")
            with a.style():
                a(__COLLAPSIBLE_STYLE)
            with a.script(type="text/javascript"):
                a(get_plotlyjs())

        with a.body():
            yield

            with a.script():
                a(__COLLAPSIBLE_SCRIPT)

    return a


def _get_scale_and_zp(
    tensor: torch.Tensor, keep_size: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    if not tensor.is_quantized:
        return None, None

    if hasattr(tensor, "q_scale") and tensor.qscheme() in [
        torch.per_tensor_affine,
        torch.per_tensor_symmetric,
    ]:
        if keep_size:
            zero_point = torch.Tensor([tensor.q_zero_point()] * tensor.numel())
            scale = torch.Tensor([tensor.q_scale()] * tensor.numel())
        else:
            zero_point = torch.Tensor([tensor.q_zero_point()])
            scale = torch.Tensor([tensor.q_scale()])
    elif hasattr(tensor, "q_per_channel_scales") and tensor.qscheme() in __PER_CHANNEL_CONFS:
        if keep_size:
            axe = tensor.q_per_channel_axis()
            zero_points_per_chan = tensor.q_per_channel_zero_points().tolist()
            zero_point = torch.as_tensor(
                [
                    x
                    for zp in zero_points_per_chan
                    for x in itertools.repeat(zp, tensor.numel() // tensor.shape[axe])
                ]
            )

            scale_per_chan = tensor.q_per_channel_scales().tolist()
            scale = torch.as_tensor(
                [
                    x
                    for sc in scale_per_chan
                    for x in itertools.repeat(sc, tensor.numel() // tensor.shape[axe])
                ]
            )
        else:
            zero_point = tensor.q_per_channel_zero_points()
            scale = tensor.q_per_channel_scales()
    elif hasattr(tensor, "scale"):
        if keep_size:
            zero_point = torch.Tensor([tensor.zero_point] * tensor.numel())
            scale = torch.Tensor([tensor.scale] * tensor.numel())
        else:
            zero_point = torch.Tensor([tensor.zero_point])
            scale = torch.Tensor([tensor.scale])
    else:
        zero_point, scale = None, None

    return (scale, zero_point)


def _compute_quant_correction(
    ovals: torch.Tensor, qvals: torch.Tensor, keep_qparams: bool
) -> torch.Tensor:
    # Error compute
    error = ovals - qvals.dequantize()

    if not keep_qparams:
        from torch.quantization.observer import MinMaxObserver

        # Will record min and max for further forward
        obs = MinMaxObserver(
            dtype=torch.quint8,
            qscheme=qvals.qscheme(),
            reduce_range=False,
            quant_min=0,
            quant_max=255,
        )
        # Surprise, here the forward
        obs(error)
        # Get the resulted scale + zp
        scale, zp = obs.calculate_qparams()
    else:
        scale, zp = _get_scale_and_zp(qvals, keep_size=False)

    #  Can now quantize the error tensor
    if qvals.qscheme() in __PER_CHANNEL_CONFS:
        qcorr = torch.quantize_per_channel(
            error, scale, zp, dtype=qvals.dtype, axis=qvals.q_per_channel_axis()
        )
    else:
        qcorr = torch.quantize_per_tensor(error, scale, zp, dtype=qvals.dtype)

    return qcorr


def _add_error_and_correction_cells_html(
    a: Airium,
    oval: float,
    qcorr: int,
    dqcorr: float,
    fix_dqval: float,
    corr_scale: float,
    corr_zp: int,
) -> Airium:
    error_fixed_dq = oval - fix_dqval
    error_fixed_dq_abs = abs(error_fixed_dq)

    a.td(_t=str(qcorr), style=f"{__PADDING_CSS}")
    a.td(_t=str(int(corr_zp)), style=f"{__PADDING_CSS}")
    a.td(_t=_round(corr_scale), style=f"{__PADDING_CSS}")
    a.td(_t=str(int(corr_zp - qcorr)), style=f"{__PADDING_CSS}")
    a.td(_t=_round(dqcorr), style=f"{__PADDING_CSS}")
    a.td(_t=_round(fix_dqval), style=f"{__PADDING_CSS}{__HIGHLIGHT_CSS}")
    a.td(
        _t=_round(error_fixed_dq),
        style=f"{__PADDING_CSS}{__BACKGROUND_CSS}{_background(error_fixed_dq_abs)}",
    )

    return a


def _add_global_comparison_tab_html(a: Airium, global_diff_dict: dict, text: str = "") -> Airium:
    with a.table(
        border=1,
        style="float:left;min-width:300px;border-collapse:collapse;text-align:center;margin-right:30px;",
    ):
        for key in ["Min", "Max", "Mean"]:
            low_key = key.lower()
            with a.tr():
                a.th(_t=f"{text} {key} error", style=__PADDING_CSS)
                a.td(
                    _t=_round(global_diff_dict[low_key]),
                    style=f"{__PADDING_CSS}{__BACKGROUND_CSS}{_background(global_diff_dict[low_key])}",
                )

    return a


def _add_empty_cell_html(a: Airium, repeat: int = 1, text: str = "...") -> Airium:
    assert repeat >= 1

    a.th(_t=text, style=__PADDING_CSS)

    for _ in range(repeat - 1):
        a.td(_t=text, style=__PADDING_CSS)

    return a


def _add_quant_comparison_tab_html(
    a: Airium,
    comparison: dict,
    scale: torch.Tensor,
    zp: torch.Tensor,
    cscale: torch.Tensor,
    czp: torch.Tensor,
    max_sample: int,
    text: str,
) -> Airium:
    original_data = comparison[0]
    dequantized_data = comparison[1]
    quantized_data = comparison[2]

    otensor_list, sample_idx = _sample_tensor(original_data["values"], max_sample=max_sample)
    dqtensor_list, _ = _sample_tensor(dequantized_data["values"], max_sample=max_sample)
    qtensor_list, _ = _sample_tensor(quantized_data["values"], max_sample=max_sample)
    scale_list, _ = _sample_tensor(scale, max_sample=max_sample)
    zp_list, _ = _sample_tensor(zp, max_sample=max_sample)

    if cscale is not None:
        dequantized_correction = comparison[3]
        quantized_correction = comparison[4]
        dq_data_p_corr = comparison[5]
        cscale_list, _ = _sample_tensor(cscale, max_sample=max_sample)
        czp_list, _ = _sample_tensor(czp, max_sample=max_sample)
        qcorr_list, _ = _sample_tensor(quantized_correction["values"], max_sample=max_sample)
        dqcorr_list, _ = _sample_tensor(dequantized_correction["values"], max_sample=max_sample)
        fix_list, _ = _sample_tensor(dq_data_p_corr["values"], max_sample=max_sample)

    assert len(otensor_list) == len(dqtensor_list)

    with a.table(
        border=1,
        style="min-width:300px;border-collapse:collapse;text-align:center;margin-right:30px;",
    ):
        with a.tr():
            a.th(_t="", style=__PADDING_CSS)
            # Values
            a.th(_t=f"Original", style=__PADDING_CSS)
            a.th(_t=f"Quantized<br>Qt | ZP | Sc", style=__PADDING_CSS, colspan="3")
            a.th(_t=f"Dequantized<br>(Qt - ZP) x Sc", style=__PADDING_CSS)

            # Error and correction
            a.th(_t=f"Error<br>Og - Dq", style=__PADDING_CSS)

            if cscale is not None:
                a.th(_t=f"Qt Correction<br>CQt | CZP | CSc | Δ", style=__PADDING_CSS, colspan="4")
                a.th(_t=f"Dq Correction<br>(CQt - CZP) x CSc", style=__PADDING_CSS)
                a.th(_t=f"Fixed Dq<br>Dq + DqCorr", style=__PADDING_CSS)
                a.th(_t=f"Error Fixed Dq<br>Og - FxDq", style=__PADDING_CSS)

        for idx in range(len(otensor_list)):
            with a.tr():
                a.th(_t=f"{text} n° {str(sample_idx[idx])}", style=__PADDING_CSS)
                a.td(_t=_round(otensor_list[idx]), style=f"{__PADDING_CSS}{__HIGHLIGHT_CSS}")
                a.td(_t=str(qtensor_list[idx]), style=__PADDING_CSS)

                a.td(_t=str(int(zp_list[idx])), style=__PADDING_CSS)
                a.td(_t=_round(scale_list[idx]), style=__PADDING_CSS)
                a.td(_t=_round(dqtensor_list[idx]), style=f"{__PADDING_CSS}{__HIGHLIGHT_CSS}")

                # First pass Dq error
                error = otensor_list[idx] - dqtensor_list[idx]
                error_abs = abs(error)
                a.td(
                    _t=_round(error),
                    style=f"{__PADDING_CSS}{__BACKGROUND_CSS}{_background(error_abs)}",
                )

                if cscale is not None:
                    _add_error_and_correction_cells_html(
                        a,
                        otensor_list[idx],
                        qcorr_list[idx],
                        dqcorr_list[idx],
                        fix_list[idx],
                        cscale_list[idx],
                        czp_list[idx],
                    )

        with a.tr():
            _add_empty_cell_html(a, 14 if cscale is not None else 7)

        low_text = text.lower()

        for key in ["Min", "Max", "Mean", "Std"]:
            low_key = key.lower()

            with a.tr():
                a.th(_t=f"{key} {low_text}", style=__PADDING_CSS)
                a.td(_t=_round(original_data[low_key]), style=__PADDING_CSS)
                a.td(_t=_x_or_round(quantized_data[low_key]), style=__PADDING_CSS)
                a.td(_t="X")
                a.td(_t="X")
                a.td(_t=_round(dequantized_data[low_key]), style=__PADDING_CSS)
                a.td(_t="X")

                if cscale is not None:
                    a.td(_t=_x_or_round(quantized_correction[low_key]), style=__PADDING_CSS)
                    a.td(_t="X")
                    a.td(_t="X")
                    a.td(_t="X")
                    a.td(_t=_round(dequantized_correction[low_key]), style=__PADDING_CSS)
                    a.td(_t=_round(dq_data_p_corr[low_key]), style=__PADDING_CSS)
                    a.td(_t="X")

    return a


def _add_comparison_tab_html(a: Airium, comparison: dict, max_sample: int, text: str) -> Airium:
    original_data = comparison[0]
    dequantized_data = comparison[1]

    otensor_list, sample_idx = _sample_tensor(original_data["values"], max_sample=max_sample)
    dqtensor_list, _ = _sample_tensor(dequantized_data["values"], max_sample=max_sample)

    assert len(otensor_list) == len(dqtensor_list)

    with a.table(
        border=1,
        style="min-width:300px;border-collapse:collapse;text-align:center;margin-right:30px;",
    ):
        with a.tr():
            a.th(_t="", style=__PADDING_CSS)
            a.th(_t=f"Original", style=__PADDING_CSS)
            a.th(_t=f"Dequantized", style=__PADDING_CSS)
            a.th(_t=f"Error", style=__PADDING_CSS)

        for idx in range(len(otensor_list)):
            with a.tr():
                a.th(_t=f"{text} n° {str(sample_idx[idx])}", style=__PADDING_CSS)
                a.td(_t=_round(otensor_list[idx]), style=__PADDING_CSS)
                a.td(_t=_round(dqtensor_list[idx]), style=__PADDING_CSS)
                diff = abs(dqtensor_list[idx] - otensor_list[idx])
                a.td(_t=_round(diff), style=f"{__PADDING_CSS}{__BACKGROUND_CSS}{_background(diff)}")

        with a.tr():
            _add_empty_cell_html(a, 4)

        low_text = text.lower()

        for key in ["Min", "Max", "Mean", "Std"]:
            low_key = key.lower()

            with a.tr():
                a.th(_t=f"{key} {low_text}", style=__PADDING_CSS)
                a.td(_t=_round(original_data[low_key]), style=__PADDING_CSS)
                a.td(_t=_round(dequantized_data[low_key]), style=__PADDING_CSS)
                diff = abs(dequantized_data[low_key] - original_data[low_key])
                a.td(_t=_round(diff), style=f"{__PADDING_CSS}{__BACKGROUND_CSS}{_background(diff)}")

    return a


def _add_histogram_html(a: Airium, tensor: torch.Tensor, text: str) -> Airium:
    data, _ = _sample_tensor(tensor, __MAX_SAMPLED_HIST)
    fig = px.histogram(data, title=text)
    html = _figure_to_html(fig)
    a(html)

    return a


def _add_comparison_html(
    a: Airium,
    original_tensor: torch.Tensor,
    quant_tensor: torch.Tensor,
    apply_corr: bool,
    quant_hist: bool,
    text: str,
) -> Airium:
    if quant_tensor.is_quantized:
        comp_list = [
            original_tensor,
            quant_tensor.dequantize(),
            quant_tensor.int_repr(),
        ]

        if apply_corr:
            keep_qparams_as_correction = True

            qcorr_tensor = _compute_quant_correction(
                original_tensor, quant_tensor, keep_qparams=keep_qparams_as_correction
            )
            cscale, czeropoint = _get_scale_and_zp(qcorr_tensor)
            comp_list.extend(
                [
                    qcorr_tensor.dequantize(),
                    qcorr_tensor.int_repr(),
                    _addition_quant_per_tensor(quant_tensor, qcorr_tensor).dequantize(),
                ]
            )

        comparison = _compare_tensors(comp_list)
        scale, zeropoint = _get_scale_and_zp(quant_tensor)

        with a.span(style="display: inline-grid; grid-template-columns: auto auto;"):
            with a.div():
                with a.div(style="text-align:center;"):
                    a(text)

                a.br()
                _add_tensor_info_html(a, quant_tensor, text)
                a.br()
                if apply_corr:
                    _add_quant_comparison_tab_html(
                        a, comparison, scale, zeropoint, cscale, czeropoint, __MAX_DISPLAYED, text
                    )
                else:
                    _add_quant_comparison_tab_html(
                        a, comparison, scale, zeropoint, None, None, __MAX_DISPLAYED, text
                    )
                a.br()

                with a.span():
                    # Compare original and dequant(quant ouput)
                    _add_global_comparison_tab_html(a, comparison[0]["diff"][0])

                    if apply_corr:
                        # Compare original and dequant(quant output + quant corr)
                        _add_global_comparison_tab_html(a, comparison[0]["diff"][4], "Fixed")

                    a.br()
            if quant_hist:
                with a.div():
                    _add_histogram_html(a, quant_tensor.int_repr(), f"{text} quantized")
                    a.br()

                    if apply_corr and keep_qparams_as_correction:
                        # Flat module and sub from the zeropoint, so we can find the amount of data need to store the correction
                        # This is meaningless if keep_qparams_as_correction is not True
                        _add_histogram_html(
                            a,
                            czeropoint - torch.flatten(qcorr_tensor.int_repr()),
                            f"{text} quantized correction Δ",
                        )
                        a.br()
    else:
        comparison = _compare_tensors([original_tensor, quant_tensor])

        with a.div(style=""):
            with a.div(style="text-align:center;"):
                a(text)

            a.br()
            _add_tensor_info_html(a, quant_tensor, text)
            a.br()
            _add_comparison_tab_html(a, comparison, __MAX_DISPLAYED, text)
            a.br()
            _add_global_comparison_tab_html(a, comparison[0]["diff"][0])
            a.br()

    return a


def _add_overview_html(
    a: Airium, model_path: str, model: VideoNet, qmodel_path: str, qmodel: VideoNet, frame_idx: int
) -> Airium:
    a(f"Forwarded frame index : {frame_idx}")
    a.br()

    with a.table(border=1, style="min-width:300px;border-collapse:collapse;text-align:center"):
        with a.tr():
            a.th(_t="", style=__PADDING_CSS)
            a.th(_t=f"Original", style=__PADDING_CSS)
            a.th(_t=f"Quantized", style=__PADDING_CSS)

        with a.tr():
            a.th(_t="Nb params", style=__PADDING_CSS)
            a.td(_t=str(model.nb_params) + " p", style=__PADDING_CSS)
            a.td(_t=str(qmodel.nb_params) + " p", style=__PADDING_CSS)

        with a.tr():
            a.th(_t="File size", style=__PADDING_CSS)
            a.td(_t=str(getsize(model_path)) + " bytes", style=__PADDING_CSS)
            a.td(_t=str(getsize(qmodel_path)) + " bytes", style=__PADDING_CSS)

    return a


def _add_quant_formula_html(a: Airium) -> Airium:
    with a.div():
        a(f"Original ~= (Qt - ZP) * Sc = DeQt")
        a.br()

    a.br()

    return a


def _add_tensor_info_html(a: Airium, tensor: torch.Tensor, text: str) -> Airium:
    with a.div():
        a(f"Nb {text} : {tensor.numel()}")
        a.br()
        a(f"{text} shape : {tensor.shape}")
        a.br()

        if tensor.is_quantized:
            qscheme = tensor.qscheme()

            if qscheme in __PER_CHANNEL_CONFS:
                a(f"{text} qScheme : {qscheme} ({tensor.q_per_channel_axis()})")
            else:
                a(f"{text} qScheme : {qscheme}")

            a.br()

    a.br()

    return a


def _add_activation_info_html(
    a: Airium, scale: Union[float, torch.Tensor], zero_point: Union[float, torch.Tensor]
) -> Airium:
    with a.div():
        a("Layer quant params :")
        a.br()
        a(f"Scale : {scale.item() if isinstance(scale, torch.Tensor) else scale}")
        a.br()
        a(
            f"Zero point : {zero_point.item() if isinstance(zero_point, torch.Tensor) else zero_point}"
        )
        a.br()

    a.br()

    return a


def _add_layer_info_html(a: Airium, olayer: Layer, qlayer: Layer, text: str) -> Airium:
    with _generate_html_button(a, title=text, collapsible=True):
        if hasattr(qlayer.module, "scale") and qlayer.module.scale is not None:
            _add_activation_info_html(a, qlayer.module.scale, qlayer.module.zero_point)

        # Quant layer info
        # _add_quant_formula_html(a)

        # Compare original weight, input and output vs quantized one
        if olayer.weight is not None and qlayer.weight is not None:
            _add_comparison_html(a, olayer.weight, qlayer.weight, False, False, "Weight")
            a.br()

        if olayer.input is not None and qlayer.output is not None:
            _add_comparison_html(a, olayer.input, qlayer.input, False, False, "Input")
            a.br()

        if olayer.output is not None and qlayer.output is not None:
            _add_comparison_html(a, olayer.output, qlayer.output, True, True, "Output")
            a.br()

    return a


def _add_content_html(a: Airium, model_path: str, qmodel_path: str, frame_idx: int) -> Airium:
    # Load models
    omodel = model_load(model_path).eval()
    qmodel = model_load(qmodel_path).eval()

    # Header info
    _add_overview_html(a, model_path, omodel, qmodel_path, qmodel, frame_idx)

    a.br()
    a.br()

    # Calculate an ordered list of modules
    olistmodules, olistmodulesnames = _flatten_module(omodel, "root")
    qlistmodules, qlistmodulesnames = _flatten_module(qmodel, "root")

    # Read inputs, outputs and weights of listed modules
    olayers = []
    qlayers = []

    def update_ooutput(name: str, l: torch.nn.Module, inp: torch.Tensor, outp: torch.Tensor):
        nonlocal olayers
        assert len(inp) == 1, "Multiple inputs are not handled yet"
        olayers.append(Layer(inp[0], outp, l, name))

    for i, mod in enumerate(olistmodules):
        name = f"{mod._get_name()} ({olistmodulesnames[i]})"
        mod.register_forward_hook(lambda l, inp, outp, n=name: update_ooutput(n, l, inp, outp))

    def update_qoutput(name: str, l: torch.nn.Module, inp: torch.Tensor, outp: torch.Tensor):
        nonlocal qlayers
        assert len(inp) == 1, "Multiple inputs are not handled yet"
        qlayers.append(Layer(inp[0], outp, l, name))

    for i, mod in enumerate(qlistmodules):
        name = f"{mod._get_name()} ({qlistmodulesnames[i]})"
        mod.register_forward_hook(lambda l, inp, outp, n=name: update_qoutput(n, l, inp, outp))

    modinput = torch.Tensor([[frame_idx]])
    omodel(modinput)
    qmodel(modinput)

    # Print comparaison
    for idx in range(len(olayers)):
        # Read the reported values
        olayer = olayers[idx]
        qlayer = qlayers[idx]

        # Define a name representation of the iteration
        name = f"{olayer.name}<>{qlayer.name}"
        qweight = qlayer.weight

        if isinstance(qlayer.module, (Quantize, DeQuantize)):
            _add_layer_info_html(a, olayer, qlayer, f"{name}")
        elif qweight is not None and qweight.is_quantized:
            _add_layer_info_html(a, olayer, qlayer, f"{name} (Fully-quantized)")
        elif qlayer.output.is_quantized:
            _add_layer_info_html(a, olayer, qlayer, f"{name} (Ghostly-quantized)")
        else:
            a(f"{name} not quantized")

        a.br()
        a.br()

    return a


def analyse_model_quant(model_fpath: str, qmodel_fpath: str, frame_idx: int, output_fpath: str):
    assert output_fpath.endswith(".html"), "Output file must be html"

    with open(output_fpath, mode="w+") as f:
        a = Airium()

        with _generate_html_page(a):
            _add_content_html(a, model_fpath, qmodel_fpath, frame_idx)

        f.write(str(a))
        print("Written index in", f.name)
