import math


def inverse_pow_clamped(val, start, stop, curve_pow, clamp_max, clamp_min):
    """
    Function that draw a contiguous curve which follows:
    - val <= start => clamp_max
    - start < val < end => curve shape defined by curve_pow
    - val >= end => clamp_min
    """

    if val <= start:
        return clamp_max

    if val >= stop:
        return clamp_min

    pow_part = math.pow((val - start) / (stop - start), curve_pow)

    output = clamp_max - pow_part * (clamp_max - clamp_min)
    output = min(clamp_max, output)
    output = max(clamp_min, output)

    return output
