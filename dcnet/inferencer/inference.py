import logging
import sys

import cv2
import munch
import numpy as np
import onnx
import onnxruntime
import torch
import torch.nn as nn
import torch.onnx

from dcnet.inferencer.auxiliary.utils import preprocess_image
from dcnet.inferencer.postprocessor import DcPostprocessor
from dcnet.network_builder import NetworkBuilder

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


class Inferencer:
    """ Differential contour text-detection network
    Args:
        opt (dict): all configs
    """

    def __init__(self, weights_path=None, onnx_path=None, resize_image_shape=736,
                 box_thresh=0.3, thresh=0.3, return_polygon=False, use_onnx=False,
                 device=torch.device("cpu")):
        self.args = dict()
        self.args["resume"] = weights_path
        self.default_resize_image_shape = resize_image_shape
        self.args["image_short_side"] = self.default_resize_image_shape
        self.args["box_thresh"] = box_thresh
        self.args["thresh"] = thresh
        self.args["resize"] = False
        self.args["polygon"] = return_polygon
        self.device = device
        self._init_model()
        self.post_processor = DcPostprocessor(self.args)
        if self.device.type == "cpu" and use_onnx:
            self.use_onnx = True
            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)
            self.ort_session = onnxruntime.InferenceSession(onnx_path)
        else:
            self.use_onnx = False

    def _init_config(self, configs):
        """ Initialze all configurations 
        Args:
            opt (dict): dict of all configs
        """
        munch_configs = munch.munchify(configs)
        if munch_configs.optimize_settings.distributed:
            torch.cuda.set_device(munch_configs.optimize_settings.local_rank)
            torch.distributed.init_process_group(
                backend="nccl", init_method="env://")
        torch.backends.cudnn.benchmark = munch_configs.optimize_settings.benchmark
        return munch_configs

    def _init_model(self):
        checkpoint = torch.load(self.args["resume"], map_location=self.device)
        dict_config = checkpoint["config"]
        state_dict = checkpoint["state_dict"]
        self.opt = self._init_config(dict_config)
        self.model = self.parallelize(
            NetworkBuilder(opt=self.opt, device=self.device),
            self.opt.optimize_settings.distributed,
            self.opt.optimize_settings.local_rank
        )
        self.model.load_state_dict(state_dict, strict=False)
        if self.device.type == "cpu":
            self.model = self.model.module.to(self.device)
        print("[INFO] Loaded from {}".format(self.args["resume"]))

    def parallelize(self, instance, distributed, local_rank):
        if distributed:
            return torch.nn.parallel.DistributedDataParallel(
                instance,
                device_ids=[local_rank],
                output_device=[local_rank],
                find_unused_parameters=True
            )
        else:
            return torch.nn.DataParallel(instance)

    def to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    def run_onnx(self, tensor_input: torch.Tensor) -> np.ndarray:
        # Compute ONNX Runtime output prediction
        ort_inputs = {self.ort_session.get_inputs(
        )[0].name: self.to_numpy(tensor_input)}
        ort_outs = self.ort_session.run(None, ort_inputs)[0]
        return ort_outs

    def convert_to_onnx(self, save_path: str) -> str:
        rand_input = rand_input = torch.rand(
            1, 3, 800, 800, requires_grad=False)
        torch.onnx.export(self.model, rand_input, save_path, export_params=True,
                          opset_version=10, do_constant_folding=True,
                          input_names=["input"], output_names=["output"],
                          dynamic_axes={"input": {0: "batch_size", 1: "channel", 2: "height", 3: "width"},
                                        "output": {0: "batch_size", 1: "channel", 2: "height", 3: "width"}})

        return save_path

    def predict(self, image, get_mask=False, pred_mask=None,
                inference_resize=None, return_polygon=False):
        self.model.eval()
        if isinstance(image, str):
            img = cv2.imread(image, cv2.IMREAD_COLOR).astype("float32")
        else:
            img = image.astype("float32")

        if inference_resize != None:
            self.args["image_short_side"] = inference_resize
        else:
            self.args["image_short_side"] = self.default_resize_image_shape

        if return_polygon:
            self.args["return_polygon"] = return_polygon

        output = []
        batch = dict()
        img, original_shape = preprocess_image(img, self.args)
        batch["filename"] = [image]
        batch["shape"] = [original_shape]
        with torch.no_grad():
            batch["image"] = img.to(self.device)
            if pred_mask == None:
                if self.use_onnx:
                    ort_output = self.run_onnx(batch["image"])
                    pred = torch.tensor(ort_output).to(self.device)
                else:
                    pred = self.model.forward(batch, is_training=False)
            else:
                pred = pred_mask

            output = self.post_processor.process(
                batch, pred, is_output_polygon=self.args["polygon"])

        if get_mask:
            return output, pred

        return output
