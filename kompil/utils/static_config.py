import os
import re
import uuid
import json
import torch
import geocoder
import platform

from kompil.utils.paths import PATH_CONFIG, PATH_CONFIG_HW
from psutil import virtual_memory


def __to_giga(val) -> int:
    return 1024 * 1024 * 1024 * val


def __hardware_info() -> tuple:
    import cpuinfo

    total_gpu_memory = torch.cuda.get_device_properties(0).total_memory
    total_cpu_memory = virtual_memory().total
    gpu_name = torch.cuda.get_device_name(0)
    cpu_name = cpuinfo.get_cpu_info()["brand_raw"]

    # Might be more consistent
    if platform.os.name == "posix":
        total_cpu_memory = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")

    return (
        total_gpu_memory,
        total_cpu_memory,
        gpu_name,
        cpu_name,
    )


def __build_hardware_default_config() -> dict:
    total_gpu_memory, total_cpu_memory, gpu_name, cpu_name = __hardware_info()

    data = {
        "hardware": {
            "gpu": {
                "name": gpu_name,
                "memory": total_gpu_memory,
            },
            "cpu": {
                "name": cpu_name,
            },
            "ram": {
                "memory": total_cpu_memory,
            },
        },
        "user": {
            "gpu_allowed_memory": int(total_gpu_memory - __to_giga(3.0)),
            "cpu_allowed_memory": int(total_cpu_memory - __to_giga(6.0)),
        },
    }
    return data


def __hardware_equivalent(data1: dict, data2: dict):
    return data1["hardware"] == data2["hardware"]


def __update_hardware(data1: dict, data2: dict):
    data1["hardware"] = data2["hardware"]


def __write_hardware_config(data: dict):
    os.makedirs(PATH_CONFIG, exist_ok=True)

    with open(PATH_CONFIG_HW, "w+") as f:
        f.write(json.dumps(data, indent=4, sort_keys=True))


def get_mac_address() -> str:
    return ":".join(re.findall("..", "%012x" % uuid.getnode()))


def get_geolocalization() -> dict:
    loc = geocoder.ip("me")

    data = {"city": loc.city, "latlng": loc.latlng}

    return data


def get_name() -> str:
    return platform.node()


def get_hardware() -> dict:
    # Get default
    default_config = __build_hardware_default_config()
    # If non existing, create the hw config file
    if not os.path.exists(PATH_CONFIG_HW):
        print("NOTE: The hardware config is not defined yet.")
        __write_hardware_config(default_config)
        data = default_config
        print(
            f"A file has been written in {PATH_CONFIG_HW} with default data. "
            "Feel free to modify it later."
        )
    # Read and check the hardware hasn't changed
    with open(PATH_CONFIG_HW, "r") as f:
        data = json.load(f)
    if not __hardware_equivalent(data, default_config):
        print("WARNING: The hardware seems to have changed.")
        res = ""
        yes_answers = ["y", "yes", "ok", "fine"]
        no_answers = ["n", "no", "whatever", "never"]

        while not res.lower() in [*yes_answers, *no_answers]:
            res = input("Do you want to reset the hardware config to new default? [y/n] ")

        if res.lower() in yes_answers:
            __update_hardware(data, default_config)
            __write_hardware_config(data)

    return data


def get_allowed_memory() -> tuple:
    config = get_hardware()["user"]
    max_allowed_ram = config["cpu_allowed_memory"]
    max_allowed_vram = config["gpu_allowed_memory"]

    return max_allowed_ram, max_allowed_vram
