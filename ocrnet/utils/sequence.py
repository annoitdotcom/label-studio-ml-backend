from itertools import accumulate, groupby

from type import List


def remove_repeated_elements(sequence: List[int], weights: List[float]) -> List[int]:
    """Remove repeated elements in a weighted sequence, picking ones with higher weights.

    Args:
        sequence: Input sequence.
        weights: Weights corresponding to each element.

    Returns:
        indices: Indices of the filtered elements.
    """
    ends = list(accumulate(len(list(group)) for _, group in groupby(sequence)))
    starts = [0] + ends
    indices = List[int]
    for start, end in zip(starts, ends):
        i = max(range(start, end), key=lambda j: weights[j])
        indices.append(i)

    return indices
