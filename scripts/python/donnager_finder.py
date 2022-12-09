import os
import copy
import math
import json
import random
import tempfile
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Tuple

from kompil.nn.topology.pattern import pattern_donnager_320p
from kompil.nn.topology.builder import build_topology_from_list
from kompil.utils.numbers import to_scale


NB_FRAMES = 580
FRAME_SHAPE = (3, 320, 480)


class Model:
    def __init__(self, hyperparams: list, name: str = None):
        self.__hyperparams = hyperparams
        self.__name = name
        self.__parameter_count = None

    @property
    def name(self) -> str:
        if self.__name:
            return self.__name
        return str(self.__hyperparams)

    @property
    def hyperparams(self):
        return self.__hyperparams

    @property
    def weight(self):
        if self.__parameter_count is None:
            self.__parameter_count = self.__calculate_weight()
        return self.__parameter_count

    def write_in_file(self, file_path: str):
        topolist = self.__to_topolist()
        with open(file_path, "w") as f:
            f.write(json.dumps(topolist, indent=2))

    def __to_topolist(self) -> list:
        actual_list = [self.__hyperparams[:-9], *self.__hyperparams[-9:]]
        return pattern_donnager_320p(*actual_list)

    def __calculate_weight(self) -> int:
        # Build model
        topolist = self.__to_topolist()
        model, _ = build_topology_from_list(topolist)
        # Calculate weight
        pp = 0
        for p in list(model.parameters()):
            nn = 1
            for s in list(p.size()):
                nn = nn * s
            pp += nn
        return pp


def generate_rand_model(base: list, sort_deconv: bool = True):
    base_in = copy.deepcopy(base)

    # Adjust linears
    linear_values = base_in[:-9]
    for i in range(len(linear_values)):
        value = linear_values[i]
        linear_values[i] = int(random.uniform(a=value * 1 / 5, b=value * 9 / 5))

    linear_values.sort()

    # Adjust deconv
    deconv_values = base_in[-9:]
    for i in range(len(deconv_values)):
        value = deconv_values[i]
        deconv_values[i] = int(random.uniform(a=value * 1 / 2, b=value * 3 / 2))

    if sort_deconv:
        deconv_values.sort(reverse=True)

    # Concat
    concat = [*linear_values, *deconv_values]

    return Model(concat)


def select_by_weight(models: List[Model], weight_range: Tuple[int, int]) -> List[Model]:
    def in_range(model: Model):
        return model.weight >= weight_range[0] and model.weight <= weight_range[1]

    return list(filter(in_range, models))


def weights_figure(models: List[Model]):
    models = sorted(models, key=lambda a: a.weight)

    x = [model.hyperparams for model in models]
    x_str = [model.name for model in models]

    y = [model.weight for model in models]

    return go.Scatter(x=x_str, y=y, mode="markers")


def hyperparams_figure(models: List[Model], highlight: List[int] = []):
    n_hp = len(models[0].hyperparams)

    # Prep data
    colnames = [*[f"lin{i}" for i in range(1, n_hp - 8)], *[f"deconv{i}" for i in range(1, 10)]]

    max_layer = 0
    for i in range(n_hp):
        values = [model.hyperparams[i] for model in models]
        max_layer = max(max_layer, max(values))

    data = []
    for i in range(n_hp):
        values = [model.hyperparams[i] for model in models]
        data.append(dict(range=[0, max_layer], label=colnames[i], values=values))

    weights = [model.weight for model in models]
    data.append(dict(range=[0, max(weights)], label="weight", values=weights))

    if not highlight:
        return go.Parcoords(dimensions=data)

    # Highlight
    highlight_list = [1 if i in highlight else 0 for i in range(len(models))]
    line = dict(color=highlight_list, colorscale=[[0, "gray"], [1, "red"]])

    return go.Parcoords(line=line, dimensions=data)


def write_in_folder(models: List[Model]):
    folder = tempfile.mkdtemp(prefix="donnager_bench_")
    print(f"write bench in {folder}...")

    for i, model in enumerate(models):
        print(f"    - {model.hyperparams}: {to_scale(model.weight)}")
        model.write_in_file(os.path.join(folder, f"bench_{i}.json"))


def print_param_range(min_params: int, max_params: int, sort_deconv: bool):

    base = [100, 256, 384, 384, 384, 384, 384, 300, 256, 128, 64, 16]
    count = 1000

    models = [generate_rand_model(base, sort_deconv) for _ in range(count)]

    models_selected = select_by_weight(models, (min_params, max_params))

    to_bench = random.choices(range(len(models_selected)), k=10)

    models_benched = [models_selected[i] for i in to_bench]

    fig = go.Figure(data=[hyperparams_figure(models_benched)])
    fig.show()

    write_in_folder(models_benched)


def print_random(sort_deconv: bool):

    base = [100, 256, 384, 384, 384, 384, 384, 300, 256, 128, 64, 16]
    count = 1000

    models = [generate_rand_model(base, sort_deconv) for _ in range(count)]

    fig = go.Figure(data=[hyperparams_figure(models)])
    fig.show()


def get_extra_by_params(params: int):
    base = [81, 107, 401, 556, 475, 342, 321, 306, 244, 168, 71, 19]

    actual = [int(val * math.sqrt(params / 8000000)) for val in base]

    return actual


def print_multiple_param_conf():

    for nb_params in range(2000000, 33000000, 1000000):
        print(f"    - {to_scale(nb_params)}P:", *get_extra_by_params(nb_params))


# print_param_range(7800000, 8200000, True)

# print_random(False)

# print_multiple_param_conf()


def print_model_selection():

    topos = [
        [128, 128, 224, 256, 224, 192, 160, 128, 120, 96, 64, 16],  # topo_2MP
        [128, 128, 256, 296, 274, 232, 208, 168, 132, 104, 64, 16],  # topo_3MP
        [128, 192, 288, 512, 306, 264, 224, 176, 140, 128, 64, 16],  # topo_4MP
        [128, 192, 328, 576, 346, 296, 256, 192, 148, 128, 64, 16],  # topo_5MP
        [128, 192, 360, 632, 394, 328, 288, 208, 156, 128, 64, 16],  # topo_6MP
        [128, 256, 392, 664, 426, 360, 320, 224, 172, 128, 64, 16],  # topo_7MP
        [128, 256, 424, 704, 458, 392, 344, 240, 180, 128, 64, 16],  # topo_8MP
        [128, 256, 464, 756, 498, 416, 356, 248, 188, 128, 64, 16],  # topo_9MP
        [128, 256, 504, 808, 538, 440, 372, 256, 192, 128, 64, 16],  # topo_10MP
        [128, 256, 544, 860, 570, 464, 388, 264, 196, 128, 64, 16],  # topo_11MP
        [128, 256, 576, 904, 602, 488, 404, 272, 200, 128, 64, 16],  # topo_12MP
        [128, 256, 608, 944, 634, 512, 420, 280, 204, 128, 64, 16],  # topo_13MP
        [128, 256, 640, 976, 666, 536, 436, 288, 208, 128, 64, 16],  # topo_14MP
        [128, 256, 664, 1016, 698, 560, 448, 296, 212, 128, 64, 16],  # topo_15MP
        [128, 256, 696, 1024, 738, 576, 460, 304, 224, 128, 64, 16],  # topo_16MP
    ]

    for topo in topos:
        model = Model(topo)

        print(f"- {to_scale(model.weight, 1)}P: {' '.join([str(val) for val in topo])}")


print_model_selection()
