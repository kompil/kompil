"""
Pareto function
"""
import copy
import numpy as np
from typing import List, Any, Callable, Union


def pareto_sort(
    elements: List[Any], score_fn: Callable[[List[Union[float, int]]], Any]
) -> List[List[Any]]:
    """
    Sort a list of elements by pareto fronts
    """
    output = list()

    elements_stack = copy.copy(elements)
    scores_stack = [score_fn(elem) for elem in elements_stack]

    while len(elements_stack) > 0:
        population_size = len(elements_stack)

        # Find paretto front mask
        pareto_front_mask = np.ones(population_size, dtype=bool)

        for i in range(population_size):
            score_i = scores_stack[i]
            # Loop through all other items
            for j in range(population_size):
                score_j = scores_stack[j]
                # Check if our 'i' pint is dominated by out 'j' point
                if all(score_j >= score_i) and any(score_j > score_i):
                    # j dominates i. Label 'i' point as not on Pareto front
                    pareto_front_mask[i] = 0
                    # Stop further comparisons with 'i' (no more comparisons needed)
                    break

        # Find pareto ids
        population_ids = np.arange(population_size)
        pareto_front_ids = population_ids[pareto_front_mask]

        # Extract front
        pareto_front_ids = -np.sort(-pareto_front_ids)
        pareto_front = list()
        for i in pareto_front_ids:
            pareto_front.append(elements_stack.pop(i))
            scores_stack.pop(i)

        output.append(pareto_front)

    return output
