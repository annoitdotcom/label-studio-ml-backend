import json
from pathlib import Path

from jsonschema import ValidationError, validate

_output_schema = json.load(
    open(
        Path(__file__).parent / "assets/output_schema.json",
        encoding="utf-8",
    ))


def is_valid_ocr_output(x):
    if isinstance(x, dict):
        x = [x]
    try:
        validate(
            x,
            _output_schema,
            types={"array": (list, tuple)},
        )
    except ValidationError:
        return False
    return True
