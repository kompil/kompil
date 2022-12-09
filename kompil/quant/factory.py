from kompil.utils.factory import Factory

__FACTORY = Factory("quantizer")


def factory() -> Factory:
    global __FACTORY
    return __FACTORY


def register_quantization(name: str):
    return factory().register(name)
