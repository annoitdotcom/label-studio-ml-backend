from dcnet.dataloader.preprocess.preprocess_base import PreprocessBase


class FilterKeys(PreprocessBase):
    required = []
    superfluous = []

    def __init__(self, opt):
        super(FilterKeys, self).__init__()
        self.opt = opt
        self.required_keys = set(self.required)

        self.superfluous_keys = self.opt.dataset.train.superfluous
        if len(self.required_keys) > 0 and len(self.superfluous_keys) > 0:
            raise ValueError(
                "required_keys and superfluous_keys can not be specified at the same time.")

    @classmethod
    def load_opt(cls, opt, is_training):
        return cls(opt)

    def __call__(self, data):
        for key in self.required:
            assert key in data, "%s is required in data" % key

        superfluous = self.superfluous_keys
        if len(superfluous) == 0:
            for key in data.keys():
                if key not in self.required_keys:
                    superfluous.add(key)

        for key in superfluous:
            del data[key]
        return data
