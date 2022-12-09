import ast
from typing import Union, List, Any, Dict


KeyvalType = Union[List[Union[str, Any]], None]


def autocast(s: str) -> Union[str, float, int]:
    # If literal_eval fails, take it as a string.
    try:
        return ast.literal_eval(s)
    except ValueError:
        return s


def arg_keyval_list_to_dict(arg: KeyvalType) -> Dict[str, Any]:
    if arg is None:
        return dict()
    assert len(arg) % 2 == 0
    output = dict()
    for i in range(0, len(arg), 2):
        key = str(arg[i])
        value = autocast(arg[i + 1])
        output[key] = value

    return output
