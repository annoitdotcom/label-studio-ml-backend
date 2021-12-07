from typing import Dict, Iterable

import numpy as np


class Seldon(object):
    """ Seldon warper. Ref: https://github.com/SeldonIO/seldon-core/blob/master/python/seldon_core/user_model.py
    """

    def predict(self, x: np.ndarray, features_names: Iterable[str], meta: Dict = None):
        """The additional argument (features_names) is from Seldon
        handle it here to not break existing compatibility and comform with Seldon.

        Args:
            x: input images.
            features_names: names fo features.  
            meta: meta data.
        """
        return self.process(x, **meta)

    def process(self, images, **kwargs):
        raise NotImplementedError
