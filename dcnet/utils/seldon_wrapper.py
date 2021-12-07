from typing import Dict, Iterable

import numpy as np


class Seldon(object):
    # Ref: https://github.com/SeldonIO/seldon-core/blob/master/python/seldon_core/user_model.py
    def predict(self, X: np.ndarray, features_names: Iterable[str], meta: Dict = None):
        # the additional argument (features_names) is from Seldon
        # handle it here to not break existing compatibility and comform with Seldon.
        return self.process(X, **meta)

    def process(self, images, **kwargs):
        raise NotImplementedError
