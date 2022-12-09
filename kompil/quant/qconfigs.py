from torch.quantization.qconfig import QConfig

from kompil.quant.observers import (
    debug_observer,
    min_max_int8_pt_observer,
    min_max_uint8_pt_observer,
    min_max_int8_pc_observer,
    min_max_uint8_pcf_observer,
)

min_max_uint8_pt_qconfig = QConfig(
    activation=min_max_uint8_pt_observer, weight=min_max_uint8_pt_observer
)

min_max_int8_pt_qconfig = QConfig(
    activation=min_max_int8_pt_observer, weight=min_max_int8_pt_observer
)

min_max_int8_pta_pcw_qconfig = QConfig(
    activation=min_max_int8_pt_observer, weight=min_max_int8_pc_observer
)

min_max_uint8_pta_int8_pcw_qconfig = QConfig(
    activation=min_max_uint8_pt_observer, weight=min_max_int8_pc_observer
)

debug_qconfig = QConfig(activation=debug_observer, weight=min_max_int8_pt_observer)

min_max_int8_pta_uint8_pcfw_qconfig = QConfig(
    activation=min_max_int8_pt_observer, weight=min_max_uint8_pcf_observer
)
