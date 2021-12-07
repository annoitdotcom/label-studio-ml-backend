from typing import Iterable, List, Tuple

import numpy as np

from .metrics import fulltext_match_score as accuracy_by_field
from .metrics import (get_accuracy_by_character_1_score,
                      get_accuracy_by_character_2_score,
                      get_accuracy_by_character_3_score,
                      get_accuracy_by_field_score)
from .metrics import get_editdistance_loss as edit_distance
from .metrics import get_similarity_score as num_correct_character
from .metrics import \
    normalized_levenshtein_similarity_score as accuracy_by_character
from .utils.seldon_wrapper import Seldon


def evaluate(model: Seldon, inputs: Iterable[np.ndarray], targets: List[str], key: str = 'text') -> Tuple[List[float], float, float, float, float]:
    """Run the OCR evaluation on given dataset.

    Args:
        model: a fully instantiated ocr model.
        inputs: iterator of evaluation data, each item is a single sample.
        targets: iterator of target values.
        key: dict key to get result from output.

    Returns:
        List of normalized similarity for each pair of inputs and targets.
        AC1, AC2, AC3 and AF on average on all inputs.
    """
    outputs = model.process(inputs)
    results = List[float]  # For AC1.
    t_char = 0  # For AC2 + AC3.
    n_correct = 0  # For AC2.
    n_error = 0  # For AC3.

    for output, target in zip(outputs, targets):
        prediction = output[key]
        dist = edit_distance(target, prediction)
        m_len = max(len(target), len(prediction))
        n_error += dist
        n_correct += m_len - dist
        t_char += len(target)
        if m_len == 0:
            results.append(1.0)
        else:
            results.append(1 - dist / m_len)

    AC1 = get_accuracy_by_character_1_score(results)
    AC2 = get_accuracy_by_character_2_score(n_correct, t_char)
    AC3 = get_accuracy_by_character_3_score(n_error, t_char)
    AF = get_accuracy_by_field_score(results)
    return (results, AC1, AC2, AC3, AF)
