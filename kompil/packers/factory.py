from typing import Callable, Any, Tuple

from kompil.utils.factory import Factory

__FACTORY_PACKER = Factory("packer")
__FACTORY_UNPACKER = Factory("unpacker")


def factory_packer() -> Factory:
    global __FACTORY_PACKER
    return __FACTORY_PACKER


def register_packer(name: str, unpacker: str, unpacker_version: int):
    def decorator(cls):
        nonlocal name
        item = (cls, unpacker, unpacker_version)
        return factory_packer().register_item(name=name, cls=item)

    return decorator


def get_packer(name: str) -> Tuple[Callable[[Any], Any], str]:
    """Get packer function and unpacker name"""
    packer_fct, unpacker_name, unpacker_version = factory_packer().get(name)
    return packer_fct, unpacker_name, unpacker_version


def factory_unpacker() -> Factory:
    global __FACTORY_UNPACKER
    return __FACTORY_UNPACKER


def register_unpacker(name: str, version: int):
    return factory_unpacker().register((name, version))


def get_unpacker(name: str, version: int) -> Callable[[Any], Any]:
    """Get packer function and unpacker name"""
    return factory_unpacker().get((name, version))
