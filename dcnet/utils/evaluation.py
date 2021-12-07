import statistics
from typing import List

from dcnet.utils.metrics import cal_all_metrics


def evaluate(predictions, targets, iou_threshold: float = 0.5) -> List[dict]:
    """Evaluate model on given dataset
    Note
    ----
    If the `inputs` and `targets` contain multiple input-target pairs,
        then final "Precision", "Recall", "F1-score" metric is calculated by average of all samples
             final "True Positive", "False Positive", "False Negative" metric is calculated by sum of all samples
    Example
    -------
    Example of evaluating layout model on a single image
    >>> from dcnet.model import DcLayout
    >>> model = DcLayout(model_path='path/to/model/weight.pt')
    >>> predictions = model.process("./data/images/doc_image.png")
    >>> results_dict = evaluate(predictions=predictions,
    >>>                         targets="./data/labels/doc_image.json")
    Example for multiple images
    >>> predictions = model.process(["doc_image_1.png","doc_image_2.png"])
    >>> results_dict = evaluate(predictions=predictions,
    >>>                         targets=["doc_image_1.json","doc_image_2.json"])
    Parameters
    ----------
    predictions:
        A single prediction (or multiple predictions). Ex: list of outputs from model.process(), or list of paths to prediction json
    targets:
        A single target (or multiple target). Ex: path (or list of paths) to json label
    iou_threshold: float
        IOU threshold to consider 2 bounding boxes are matched, default=0.5
    Returns
    -------
    List of dictionary of metrics for given model on inputs and targets.
        Each dictionary includes:
            name: 
                Name of the metric, including "Precision", "Recall", "F1-score", "True Positive", "False Positive", "False Negative"
            value:
                Precision: float
                    Precision is how well a model can get only the relevant objects
                    Precision = TP/(TP+FP)
                Recall: float
                    Recall is how well a model can get all the relevant objects
                    Recall = TP/(TP+FN)
                F1-score: float
                    Balance between precision and recall
                    F1 = 2TP/(2TP+FP+FN)
                True Positive: int
                    True Positive - A detection with IOU >= threshold
                False Positive: int
                    False Positive - A detection with IOU < threshold
                False Negative: int
                    False Negative - A ground truth that is not detected
        E.g:
            [
                {"name": "Precision",
                 "value": 0.9},

                {"name": "Recall"}
                "value": 0.8},
                ...
            ]
        Evaluation results corresponding to each metric
    """

    # Check input and target type mismatching, if 1 in 2 is a list, the other should also be a list
    if not type(targets) is type(predictions) and (isinstance(targets, list) or isinstance(predictions, list)):
        raise TypeError(
            f"targets and predictions should be lists at the same time or neither,  {type(targets)} != {type(predictions)} ")

    # Standardize single input and outputs to list of 1 input to work with multi inputs case
    if not isinstance(targets, list):
        targets = [targets]
    if not isinstance(predictions, list):
        predictions = [predictions]

    # Check input and target len mismatching
    if len(targets) != len(predictions):
        raise ValueError(
            f"Mismatched length of inputs and targets,  {len(targets)} != {len(predictions)} ")

    metric_names = ["F1-score", "Precision", "Recall",
                    "True Positive", "False Positive", "False Negative"]

    # Results dict to accumulate metrics for each samples
    results_dict = {metric: [] for metric in metric_names}

    for target, prediction in zip(targets, predictions):

        # Includes (f1, precision, recall, tp, fp, fn)
        metric_result_list = cal_all_metrics(gt_boxes=target, pd_boxes=prediction,
                                             threshold=iou_threshold)
        # Accumulate metrics from single pair of input-target
        for idx, metric in enumerate(metric_names):
            results_dict[metric].append(metric_result_list[idx])

    # Calculate average metrics
    for metric in metric_names:
        if metric in ("Precision", "Recall", "F1-score"):

            # "Precision", "Recall", "F1" is accumulated for average of all pairs
            results_dict[metric] = statistics.mean(results_dict[metric])
        elif metric in ("True Positive", "False Positive", "False Negative"):

            # "True Positive", "False Positive", "False Negative" is accumulated for sum
            results_dict[metric] = sum(results_dict[metric])

    result = []

    # Standardize output format: list of dict
    for metric_name, value in results_dict.items():
        result.append({
            "name": metric_name,
            "value": value
        })
    return result
