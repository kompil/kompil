"""
Player inspired from the non-working https://github.com/xjsXjtu/videox-player
"""
import torch
import unidecode
from typing import Dict, Any

from kompil.player.player import Player
from kompil.player.options import PlayerTargetOptions


def multargs_to_dict(name: str, args: dict) -> Dict[int, Any]:
    class __Stop:
        pass

    output = {}
    count = 0
    while True:
        varname = name + str(count)
        varval = args.get(varname, __Stop)
        if varval == __Stop:
            break
        output[count] = varval
        count += 1

    return output


def _args_to_player_opt(**kwargs) -> PlayerTargetOptions:
    # Get device
    cpu = kwargs.get("cpu")
    device = torch.device("cuda") if not cpu else torch.device("cpu")
    # Get dtype
    fp32 = kwargs.get("fp32")
    if cpu and not fp32:
        print("WARNING: cpu only allows for float32, adding the --fp32 option automatically")
        fp32 = True
    dtype = torch.float32 if fp32 else torch.half
    # Get resolution
    resolution = kwargs.get("resolution")
    # Return options
    return PlayerTargetOptions(device=device, dtype=dtype, resolution=resolution)


def play(**kwargs):
    # Get args
    files = kwargs.get("files")
    framerate = kwargs.get("framerate")
    pipe = kwargs.get("pipe")
    decoders = multargs_to_dict("decoder", kwargs)
    filters = multargs_to_dict("filter", kwargs)
    # Get target options
    opt = _args_to_player_opt(**kwargs)
    # Build name
    name = "Player " + " ".join([unidecode.unidecode(file) for file in files])
    # Build player
    player = Player(name, files, opt, framerate, decoders, filters)
    if not pipe:
        player.run()
    else:
        player.pipe()
