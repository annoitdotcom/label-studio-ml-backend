import editdistance
import numpy as np


def fulltext_match_score(y_true: str, y_pred: str) -> float:
    """Binary scoring for text sequences.

    Args:
        y_true: true value string.
        y_pred: predict string value.

    Return:
        Is match or not.
    """
    return float(y_true == y_pred)


def get_editdistance_loss(y_true: str, y_pred: str) -> int:
    """Edit distance between two string. 

    Args:
        y_true: true value string.
        y_pred: predict string value.

    Return:
        Edit distance between two strings.
    """
    return editdistance.distance(y_true, y_pred)


def normalized_levenshtein_similarity_score(y_true: str, y_pred: str) -> float:
    """Levenshtein similarity normalized between 0 to 1.

    Args:
        y_true: true value string.
        y_pred: predict string value.

    Return:
        Levenshien similarity score between two strings.
    """
    max_len = max(len(y_pred), len(y_true))

    # Both empty string
    if max_len == 0:
        return 1.0
    else:
        return 1 - get_editdistance_loss(y_true, y_pred) / max_len


def get_similarity_score(y_true: str, y_pred: str) -> int:
    """Get similiarity score.

    Args:
        y_true: true value string.
        y_pred: predict string value.

    Return:
        Similarity score between two strings.
    """
    max_len = max(len(y_pred), len(y_true))
    return max_len - get_editdistance_loss(y_true, y_pred)


def get_total_character(y_true: str, y_pred: str) -> int:
    """Total number of characters in y_true.
    Need y_pred parameters due to consistent if want to use in for loop

    Args:
        y_true: true value string.
        y_pred: predict string value.

    Return:
        Number of characters.
    """
    return len(y_true)


def get_accuracy_by_character_1_score(results: list) -> np.float64:
    """Measure the accuracy by levenshtien similarity.

    Args
        results: Levenshtein similarity normalized of each time step.

    Return:
        AC1 metric result.
    """
    results_np = np.array(results)
    return results_np.mean()


def get_accuracy_by_character_2_score(n_norrect: int, t_char: int) -> float:
    """Measure the accuracy by character. 

    Args:
        n_correct: Number correct character in all dataset.
        t_char: Total characters of labels.
    Return:
        AC2 metric result.
    """
    return n_norrect / t_char


def get_accuracy_by_character_3_score(n_error: int, t_char: int) -> float:
    """Measure the accuracy by edit distance.

    Args:
        n_error: Total edit distance between each prediction and labels.
        t_char: Total characters of labels.

    Return:
        AC3 metric result.
    """
    return 1 - n_error / t_char


def get_accuracy_by_field_score(results: list) -> np.float64:
    """Neasure the accuracy by field.

    Args:
        results: Levenshtein similarity normalized of each time step.

    Return:
        AF metrics result.
    """
    results_np = np.array(results)
    return results_np[results_np >= 0.9999999].sum() / len(results_np)
