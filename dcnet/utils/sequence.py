from itertools import accumulate, groupby


def remove_repeated_elements(sequence: list, weights: list) -> list:
    '''Remove repeated elements in a weighted sequence, picking ones with higher weights
    Parameters:
        sequence : list, input sequence
        weights : list of float, weights corresponding to each element
    Returns:
        indices : indices of the filtered elements
    '''
    ends = list(accumulate(len(list(group)) for _, group in groupby(sequence)))
    starts = [0] + ends
    indices = []
    for start, end in zip(starts, ends):
        i = max(range(start, end), key=lambda j: weights[j])
        indices.append(i)
    return indices
