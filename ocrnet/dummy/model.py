from ocr.utils.input import cast_image_to_array, handle_single_input
from ocr.utils.seldon_wrapper import Seldon


class DummyOCR(Seldon):
    """Dummy OCR model, for testing purpose."""

    def __init__(self, weights_path: str):
        pass

    @handle_single_input(cast_image_to_array)
    def process(self, x):
        result = [{
            'text': 'abc',
            'confidence_by_character': [1.0],
            'confidence_by_field': 1.0,
        }] * len(x)
        return result
