from .factory import register_packer, register_unpacker


@register_packer("foo", "bar", 0)
def foo(data: dict) -> dict:
    return data


@register_unpacker("bar", version=0)
def bar(data: dict) -> dict:
    return data
