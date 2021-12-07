import inspect
import types
from functools import partial, wraps
from pathlib import Path, WindowsPath

import cv2
import numpy as np
from cv2 import imread
from PIL import Image


def _is_single_input(x):
    '''Test if input is iterable but not a str or numpy array'''
    return type(x) not in (list, tuple, types.GeneratorType)


def handle_single_input(preprocess_hook=lambda x: x):
    def decorator(func):
        @wraps(func)
        def decorated_func(*args, **kwargs):
            input_index = 0
            if inspect.getfullargspec(func).args[0] == 'self':
                input_index = 1
            input_ = args[input_index]
            is_single_input = _is_single_input(input_)
            if is_single_input:
                input_ = [input_]
            args = list(args)
            args[input_index] = list(map(preprocess_hook, input_))
            try:
                result = func(*args, **kwargs)
            except Exception as e:
                raise TypeError(
                    'Perhaps you function does not accept an Iterable as input?'
                ) from e
            # unpack to a single element if input is single
            if is_single_input:
                [result] = result
            return result

        return decorated_func

    return decorator


def _is(type_):
    return lambda x: isinstance(x, type_)


def _is_windows_path(x):
    try:
        return _is(WindowsPath)(Path(x))
    except:
        return False


def imread_windows(path: str) -> np.array:
    image = bytearray(open(path, 'rb').read())
    image = np.asarray(image, 'uint8')
    image = cv2.imdecode(image, cv2.IMREAD_UNCHANGED)
    return image


def imread_buffer(buffer_):
    image = np.frombuffer(buffer_, dtype='uint8')
    image = cv2.imdecode(image, cv2.IMREAD_UNCHANGED)
    return image


def cast_image_to_array(x):
    handlers = {
        _is_windows_path: imread_windows,
        _is(str): imread,
        _is(Path): lambda x: imread(str(x)),
        _is(bytes): imread_buffer,
        _is(np.ndarray): np.array,
        _is(Image.Image): np.array,
    }
    for condition, handler in handlers.items():
        if condition(x):
            return handler(x)
    raise TypeError(f'Unsupported image type {type(x)}')
