import torch
from typing import List, Optional, Callable
from kompil.nn.models.model import VideoNet
import kompil.nn.layers as layers


def _find_max_values(
    model: VideoNet,
    module: torch.nn.Module,
    dim: int,
    scale_cb: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
):
    max_values: Optional[torch.Tensor] = None

    def _hook(mod, inp, outp: torch.Tensor):
        assert outp.shape[0] == 1, "Only batch sizes 1 are supported yet."
        # Handle pixel_shuffle
        nonlocal scale_cb
        local_values = outp
        if scale_cb is not None:
            local_values = scale_cb(local_values)
        # Compute local max
        nonlocal dim
        local_values = torch.movedim(local_values, dim, 0)
        local_values = local_values.view(local_values.shape[0], -1)
        max_local_values = torch.max(torch.abs(local_values), dim=1).values
        # Calculate global max
        nonlocal max_values
        if max_values is None:
            max_values = max_local_values
            return
        max_lg_values = torch.cat([max_values.unsqueeze(0), max_local_values.unsqueeze(0)], dim=0)
        max_values = torch.max(max_lg_values, dim=0).values

    # Run model with hook
    handle = module.register_forward_hook(_hook)
    model.run_once()
    handle.remove()

    return max_values


def _inverse_pixelshuffle(values: torch.Tensor, pixshuf: Optional[torch.nn.PixelShuffle]):
    if pixshuf is not None:
        pixshuf2 = pixshuf.upscale_factor * pixshuf.upscale_factor
        adapted = torch.nn.functional.interpolate(values.view(1, -1, 1), scale_factor=pixshuf2)
        return adapted.flatten()
    else:
        return values


def _inverse_reshape(values: torch.Tensor, module: layers.Reshape):
    target_shape = module.shape
    upscale_ratio = target_shape[-1] * target_shape[-2]
    adapted = values.repeat_interleave(upscale_ratio)
    return adapted.flatten()


def _adjust_prev_linear(module: torch.nn.Linear, ratio: torch.Tensor):
    weight_shape = module.weight.data.shape
    module.weight.data = torch.mul(
        module.weight.data.view(weight_shape[0], -1), ratio.unsqueeze(0).t()
    ).view(*weight_shape)
    module.bias.data = torch.mul(module.bias.data, ratio)


def _adjust_prev_conv2d(module: torch.nn.Conv2d, ratio: torch.Tensor):
    weight_shape = module.weight.data.shape
    module.weight.data = torch.mul(
        module.weight.data.view(weight_shape[0], -1), ratio.unsqueeze(0).t()
    ).view(*weight_shape)
    module.bias.data = torch.mul(module.bias.data, ratio)


def _adjust_next_linear(module: torch.nn.Linear, ratio: torch.Tensor):
    weight = module.weight.data
    weight = weight.permute((1, 0)).contiguous()
    permuted_shape = weight.shape
    weight = weight.view(permuted_shape[0], -1)
    weight = torch.mul(weight, ratio.unsqueeze(0).t())
    weight = weight.view(permuted_shape).permute((1, 0)).contiguous()
    module.weight.data = weight


def _adjust_next_conv2d(module: torch.nn.Conv2d, ratio: torch.Tensor):
    weight = module.weight.data
    weight = weight.permute((1, 0, 2, 3)).contiguous()
    permuted_shape = weight.shape
    weight = weight.view(permuted_shape[0], -1)
    weight = torch.mul(weight, ratio.unsqueeze(0).t())
    weight = weight.view(permuted_shape).permute((1, 0, 2, 3)).contiguous()
    module.weight.data = weight


def _normalize_conv_sequence(
    model: VideoNet,
    module_prev: torch.nn.Conv2d,
    module_next: torch.nn.Conv2d,
    pixshuf: Optional[torch.nn.PixelShuffle],
):
    def _pixshuf_cb(t: torch.Tensor) -> torch.Tensor:
        nonlocal pixshuf
        if pixshuf is None:
            return t
        return pixshuf(t)

    max_values = _find_max_values(model, module_prev, dim=1, scale_cb=_pixshuf_cb)

    # Find ratios to rescale to [-1;1]
    inv_ratios = 1 / _inverse_pixelshuffle(max_values, pixshuf)
    # Adjust conv prec
    _adjust_prev_conv2d(module=module_prev, ratio=inv_ratios)
    # Adjust conv next
    _adjust_next_conv2d(module=module_next, ratio=max_values)


def _normalize_linear_sequence(
    model: VideoNet, module_prev: torch.nn.Linear, module_next: torch.nn.Linear
):
    max_values = _find_max_values(model, module_prev, dim=1)

    # Find ratios to rescale to [-1;1]
    inv_ratios = 1 / max_values
    # Adjust precedant module
    _adjust_prev_linear(module=module_prev, ratio=inv_ratios)
    # Adjust next module
    _adjust_next_linear(module=module_next, ratio=max_values)


def _normalize_linear_conv_sequence(
    model: VideoNet,
    module_prev: torch.nn.Linear,
    module_next: torch.nn.Conv2d,
    reshape_mod: layers.Reshape,
):
    def _scale_cb(t: torch.Tensor) -> torch.Tensor:
        nonlocal reshape_mod
        if reshape_mod is None:
            return t
        return reshape_mod(t)

    max_values = _find_max_values(model, module_prev, dim=1, scale_cb=_scale_cb)

    # Find ratios to rescale to [-1;1]
    inv_ratios = 1 / _inverse_reshape(max_values, reshape_mod)
    # Adjust precedant module
    _adjust_prev_linear(module=module_prev, ratio=inv_ratios)
    # Adjust next module
    _adjust_next_conv2d(module=module_next, ratio=max_values)


def _recursive_iter_sequences(module: torch.nn.Module) -> List[torch.nn.Sequential]:
    for mod in module.children():
        if isinstance(mod, torch.nn.Sequential):
            yield mod
        else:
            for mod2 in _recursive_iter_sequences(mod):
                yield mod2


__SEQS = []


def register_seq(cls):
    global __SEQS
    __SEQS.append(cls)
    return cls


@register_seq
def _(model, sequence, start) -> bool:
    modules = sequence[start : start + 4]
    if len(modules) < 4:
        return False
    mod0, mod1, mod2, mod3 = modules
    if (
        not isinstance(mod0, torch.nn.Linear)
        or not isinstance(mod1, torch.nn.PReLU)
        or not isinstance(mod2, layers.Reshape)
        or not isinstance(mod3, torch.nn.Conv2d)
    ):
        return False
    print("normalizing linear-conv seq", start, start + 3)
    _normalize_linear_conv_sequence(model, mod0, mod3, mod2)


@register_seq
def _(model, sequence, start) -> bool:
    modules = sequence[start : start + 3]
    if len(modules) < 3:
        return False
    mod0, mod1, mod2 = modules
    if not isinstance(mod0, torch.nn.Linear):
        return False
    if not isinstance(mod1, torch.nn.PReLU):
        return False
    if not isinstance(mod2, torch.nn.Linear):
        return False
    print("normalizing linear seq", start, start + 2)
    _normalize_linear_sequence(model, mod0, mod2)
    return True


@register_seq
def _(model, sequence, start) -> bool:
    modules = sequence[start : start + 2]
    if len(modules) < 2:
        return False
    mod0, mod1 = modules
    if not isinstance(mod0, torch.nn.Conv2d):
        return False
    if not isinstance(mod1, torch.nn.Conv2d):
        return False
    print("normalizing conv seq", start, start + 1)
    _normalize_conv_sequence(model, mod0, mod1, None)
    return True


@register_seq
def _(model, sequence, start) -> bool:
    modules = sequence[start : start + 3]
    if len(modules) < 3:
        return False
    mod0, mod1, mod2 = modules
    if not isinstance(mod0, torch.nn.Conv2d):
        return False
    if not isinstance(mod1, torch.nn.PReLU):
        return False
    if not isinstance(mod2, torch.nn.Conv2d):
        return False
    print("normalizing conv seq", start, start + 2)
    _normalize_conv_sequence(model, mod0, mod2, None)
    return True


@register_seq
def _(model, sequence, start) -> bool:
    modules = sequence[start : start + 4]
    if len(modules) < 4:
        return False
    mod0, mod1, mod2, mod3 = modules
    mod_pixshuf = None
    if not isinstance(mod0, torch.nn.Conv2d) or not isinstance(mod3, torch.nn.Conv2d):
        return False
    if isinstance(mod1, torch.nn.PixelShuffle) and isinstance(mod2, torch.nn.PReLU):
        mod_pixshuf = mod1
    if isinstance(mod1, torch.nn.PReLU) and isinstance(mod2, torch.nn.PixelShuffle):
        mod_pixshuf = mod2
    if mod_pixshuf is None:
        return False
    print("normalizing conv seq", start, start + 3, mod_pixshuf.upscale_factor)
    _normalize_conv_sequence(model, mod0, mod3, mod_pixshuf)
    return False


def normalize_model(model: VideoNet) -> VideoNet:

    new_model = model.clean_clone()

    for sequence in _recursive_iter_sequences(new_model):
        for i in range(len(sequence)):
            for seqfunc in __SEQS:
                if seqfunc(new_model, sequence, i):
                    break

    return new_model
