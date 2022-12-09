"""
Tools to build argument selection for benchmarks.
"""
import copy


class Variable:
    def __init__(self, *options):
        self.__options = options

    def __iter__(self):
        return iter(self.__options)


def all_possibilities(values: list, base: dict = None):

    base = base if base is not None else {}
    copy_base = copy.copy(base)

    if not values:
        yield copy_base
        return

    argname, value_it = values[0]

    if not isinstance(value_it, Variable):
        copy_base[argname] = value_it
        for out in all_possibilities(values[1:], copy_base):
            yield out
        return

    for inval in value_it:
        copy_base[argname] = inval
        for out in all_possibilities(values[1:], copy_base):
            yield out


EXAMPLE = [
    ("my_arg_1", Variable(1, 2, 4, 5)),
    ("my_arg_2", Variable("Bonjour", "Hello")),
    ("my_arg_3", 42),
]


def call_example(my_arg_1, my_arg_2, my_arg_3):
    print(my_arg_1, my_arg_2, my_arg_3)


def run_example():
    for args in all_possibilities(EXAMPLE):
        call_example(**args)


if __name__ == "__main__":
    run_example()
