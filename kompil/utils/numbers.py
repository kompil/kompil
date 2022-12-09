import math


def get_order(value: float) -> int:
    """Get the number order (power of 10)"""
    if value == 0:
        return 0

    elif value >= 10:
        return 1 + get_order(value / 10.0)

    elif value < 1:
        return -1 + get_order(value * 10.0)

    return 0


def get_order_bin(value: float) -> int:
    """Get the number order (power of 10)"""
    if value == 0:
        return 0

    elif value >= 2:
        return 1 + get_order_bin(value / 2)

    elif value < 1:
        return -1 + get_order_bin(value * 2)

    return 0


def round_to_sign_fig(value: float, max_sf: int) -> float:
    """get the value with a maximum of N figures after the dot"""
    rounded = round(value, max_sf)
    inted = int(rounded)
    return rounded if inted != rounded else inted


def to_scale(value: float, max_sf: int = 2) -> str:
    UPPER_SCALE_LETTERS = ["K", "M", "G"]
    LOWER_SCALE_LETTERS = ["m", "u", "n"]

    order = get_order(value)
    order_3 = math.floor(order / 3)
    power_3 = math.pow(10, order_3 * 3)

    value_on_order = value / power_3

    scale_letter = ""
    if order_3 > 0:
        scale_letter = UPPER_SCALE_LETTERS[order_3 - 1]
    elif order_3 < 0:
        scale_letter = LOWER_SCALE_LETTERS[-1 - order_3]

    return f"{round_to_sign_fig(value_on_order, max_sf)}{scale_letter}"


def to_scale_bin(value: float, max_sf: int = 2) -> str:
    UPPER_SCALE_LETTERS = ["K", "M", "G"]
    LOWER_SCALE_LETTERS = ["m", "u", "n"]

    order = get_order_bin(value)
    order_10 = math.floor(order / 10)
    power_10 = math.pow(1024, order_10)

    value_on_order = value / power_10

    scale_letter = ""
    if order_10 > 0:
        scale_letter = UPPER_SCALE_LETTERS[order_10 - 1]
    elif order_10 < 0:
        scale_letter = LOWER_SCALE_LETTERS[-1 - order_10]

    return f"{round_to_sign_fig(value_on_order, max_sf)}{scale_letter}"


class PrimeNumbers:

    NUMBERS = [2, 3, 5, 7, 11, 13, 17, 19, 31, 37, 41, 43, 47, 53, 59, 61, 71, 73, 79, 83, 89, 97]

    def __init__(self):
        self.reset()

    def reset(self):
        self.__id = 0

    def pop(self) -> int:
        curr_id = self.__id
        self.__id += 1
        if len(self.NUMBERS) < curr_id:
            raise RuntimeError(f"Max prime number reached: {curr_id}({self.NUMBERS[-1]})")
        return self.NUMBERS[curr_id]
