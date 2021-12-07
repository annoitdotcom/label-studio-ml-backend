import functools
import logging
import os
import sys
import time
from datetime import datetime

import cv2
import numpy as np
import yaml
from tensorboardX import SummaryWriter

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


class Logger(object):
    """

    """

    def __init__(self, opt):
        self.logged = -1
        self.speed = None
        self.eta_time = None
        self.opt = opt
        self.timestamp = time.time()

        if self.opt.logging.verbose:
            self._info(
                "Initializing output directory for {0}".format(
                    self.opt.logging.log_dir_name
                )
            )

        self.log_path = os.path.join(
            self.opt.output_dir,
            self.opt.logging.log_dir_name
        )
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)

        summary_path = os.path.join(
            self.log_path,
            self.opt.logging.summary_dir_name
        )
        self.tf_board_logger = SummaryWriter(summary_path)
        self.message_logger = self._init_message_logger()
        self._make_storage()

        self.metrics_writer = open(os.path.join(
            self.log_path,
            self.opt.logging.metrics_file_name
        ), "at")

    def _make_storage(self):
        application = os.path.basename(os.getcwd())
        storage_dir = os.path.join(
            self.log_path,
            application
        )

        if not os.path.exists(storage_dir):
            os.makedirs(storage_dir)

    def _info(self, content: str):
        logging.debug(content)

    def _save_dir(self, dir_name):
        return os.path.join(self.log_path, dir_name)

    def _init_message_logger(self):
        message_logger = logging.getLogger('messages')
        message_logger.setLevel(
            logging.DEBUG if self.opt.logging.verbose else logging.INFO)

        formatter = logging.Formatter(
            '[%(levelname)s] [%(asctime)s] %(message)s')

        std_handler = logging.StreamHandler()
        std_handler.setLevel(message_logger.level)
        std_handler.setFormatter(formatter)

        file_handler = logging.FileHandler(
            os.path.join(self.log_path, self.opt.logging.log_file_name)
        )

        file_handler.setLevel(message_logger.level)
        file_handler.setFormatter(formatter)

        message_logger.addHandler(std_handler)
        message_logger.addHandler(file_handler)
        return message_logger

    def _report_time(self, name: str):
        if self.opt.logging.verbose:
            self._info(
                "{0} time: {1}".format(name, time.time() - self.timestamp)
            )
            self.timestamp = time.time()

    def _report_eta(self, steps, total, epoch):
        self.logged = self.logged % total + 1
        steps = steps % total

        if self.eta_time is None:
            self.eta_time = time.time()
            speed = -1
        else:
            eta_time = time.time()
            speed = eta_time - self.eta_time

            if self.speed is not None:
                speed = ((self.logged - 1) * self.speed + speed) / self.logged

            self.speed = speed
            self.eta_time = eta_time

        seconds = (total - steps) * speed
        hours = seconds // 3600
        minutes = (seconds - (hours * 3600)) // 60
        seconds = seconds % 60
        if steps % 20 == 0:
            self._info(
                "{0}/{1} batches processed in epoch {2}, ETA: {3}:{4}:{5}".format(
                    steps, total, epoch, hours, minutes, seconds
                )
            )

    def _handle_configs(self, parameters=None):
        config_file_path = os.path.join(
            self.log_path,
            self.opt.logging.config_file_name
        )

        if parameters is None:
            with open(config_file_path, 'rt') as reader:
                return yaml.load(reader.read())
        else:
            with open(config_file_path, 'wt') as writer:
                yaml.dump(parameters, writer)

    def _report_metrics(self, epoch, steps, metrics_dict):
        # for name, metric in metrics_dict.items():
        #     results[name] = {"count": metric.count, "value": float(metric.avg)}
        #     self.add_scalar("metrics/" + name, metric.avg, steps)

        result_dict = {
            str(datetime.now()): {
                "epoch": epoch,
                "steps": steps,
                "metrics": metrics_dict
            }
        }

        string_result = yaml.dump(result_dict)
        self._info(string_result)
        self.metrics_writer.write(string_result)
        self.metrics_writer.flush()

    def _named_number(self, name, num=None, default=0):
        if num is None:
            return int(self.has_signal(name)) or default
        else:
            with open(os.path.join(self.log_path, name), 'w') as writer:
                writer.write(str(num))
            return num

    epoch = functools.partialmethod(_named_number, 'epoch')
    iter = functools.partialmethod(_named_number, 'iter')

    def _log_message(self, level, content):
        self.message_logger.__getattribute__(level)(content)

    def _save_images(self, name, images):
        for i, image in enumerate(images):
            if i == 0:
                result = image
            else:
                result = np.concatenate([result, image], 0)
        cv2.imwrite(os.path.join(
            self._make_visualize_dir(), name+'.jpg'), result)

    def _make_visualize_dir(self):
        vis_dir = os.path.join(
            self.log_path,
            self.opt.logging.visualize_dir_name
        )

        if not os.path.exists(vis_dir):
            os.mkdir(vis_dir)
        return vis_dir

    def _save_image_dict(self, images, max_size=1024):
        for file_name, image in images.items():
            height, width = image.shape[:2]

            if height > width:
                actual_height = min(height, max_size)
                actual_width = int(round(actual_height * width / height))
            else:
                actual_width = min(width, max_size)
                actual_height = int(round(actual_width * height / width))

            image = cv2.resize(image, (actual_width, actual_height))
            cv2.imwrite(os.path.join(
                self._make_visualize_dir(), file_name+'.jpg'), image)

    def __getattr__(self, name):
        message_levels = set(['debug', 'info', 'warning', 'error', 'critical'])
        if name == '__setstate__':
            raise AttributeError('haha')
        if name in message_levels:
            return functools.partial(self._log_message, name)
        elif hasattr(self.__dict__.get('tf_board_logger'), name):
            return self.tf_board_logger.__getattribute__(name)
        else:
            super()
