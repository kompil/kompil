from kompil.utils.factory import Factory

__FACTORY = Factory("corrector")


def factory() -> Factory:
    global __FACTORY
    return __FACTORY


def register_corrector(name: str):
    return factory().register(name)
