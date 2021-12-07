import logging
import os
import sys

import anyconfig
import munch
import numpy as np
import torch
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)
from tqdm import tqdm

from dcnet.dataloader.augmenter import str2augm
from dcnet.dataloader.data_loader import DataLoader
from dcnet.dataloader.dataset import str2dataset
from dcnet.dataloader.preprocess import str2preprocess
from dcnet.trainer.losses import str2loss
from dcnet.trainer.optimizers.learning_rate_schedulers import str2lr
from dcnet.trainer.optimizers.optimize_schedulers.optimizer_scheduler import \
    OptimizerScheduler
from dcnet.trainer.utilities.checkpoint import Checkpoint
from dcnet.trainer.utilities.logger import Logger
from dcnet.trainer.utilities.model_saver import ModelSaver

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


class Trainer:
    def __init__(self, model, config_path, device="cpu"):
        self.dict_config = None
        self.opt = self._init_config(config_path)

        self.total = 0
        self.current_lr = 0

        self.device = device
        self.opt.output_dir = self._init_output_dir()

        self.lr_scheduler = self._init_lr_scheduler()
        self.model = self._init_model(model)

        self.logger = Logger(self.opt)
        self.model_saver = ModelSaver(self.opt)
        self.min_loss = 100000

    def _init_config(self, config_path):
        """ Initialze all configurations 
        Args:
            opt (dict): dict of all configs
        """
        self.dict_config = anyconfig.load(config_path)
        munch_configs = munch.munchify(self.dict_config)

        if munch_configs.optimize_settings.distributed:
            torch.cuda.set_device(munch_configs.optimize_settings.local_rank)
            torch.distributed.init_process_group(
                backend='nccl',
                init_method='env://'
            )
        torch.backends.cudnn.benchmark = munch_configs.optimize_settings.benchmark
        return munch_configs

    def _init_output_dir(self):
        output_dir = os.path.join(
            self.opt.output_dir,
            self.opt.experiment_name
        )
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        return output_dir

    def _init_model(self, model):
        model = self.parallelize(
            model,
            self.opt.optimize_settings.distributed,
            self.opt.optimize_settings.local_rank
        )
        return model

    def _init_criterion(self):
        try:
            criterion = str2loss[self.opt.optimize_settings.loss.type].load_opt(
                self.opt)
            logging.debug("Loss type: {0}".format(
                criterion.__class__.__name__))

            criterion = self.parallelize(
                criterion,
                self.opt.optimize_settings.distributed,
                self.opt.optimize_settings.local_rank
            )
            return criterion
        except Exception as error:
            logging.error("Error at %s", "division", exc_info=error)

    def _init_lr_scheduler(self):
        try:
            lr_scheduler = str2lr[
                self.opt.optimize_settings.learning_rate_scheduler.type].load_opt(self.opt)
            logging.debug("Learning rate scheduler type: {0}".format(
                lr_scheduler.__class__.__name__)
            )
            return lr_scheduler
        except Exception as error:
            logging.error("Error at %s", "division", exc_info=error)

    def to_np(self, x):
        return x.cpu().data.numpy()

    def update_learning_rate(self, optimizer, epoch, step):
        lr = self.lr_scheduler.get_learning_rate(epoch, step)

        for group in optimizer.param_groups:
            group['lr'] = lr
        self.current_lr = lr

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

    def train(self):
        self.logger._report_time("Start")
        self.logger._handle_configs(self.opt)
        self.best_f1 = 0
        # Encoder network
        processes = []
        for augmenter_name in self.opt.dataset.train.augmenter_names:
            processes.append(str2augm[augmenter_name].load_opt(
                self.opt, is_training=True))

        for preprocess_name in self.opt.dataset.train.preprocess_names:
            processes.append(str2preprocess[preprocess_name].load_opt(
                self.opt, is_training=True))

        # temp_dataset = '/mnt/ai_filestore/home/jason/sota_exp/DCNet/assets/dataset/temp'
        train_dataset = str2dataset[self.opt.dataset.train.dataset].load_opt(
            self.opt, processes, is_training=True)

        train_data_loader = DataLoader(
            **{'opt': self.opt, 'dataset': train_dataset, 'is_training': True}).load_dataloader()

        if self.opt.dataset.validation.activate:
            processes = []

            for augmenter_name in self.opt.dataset.validation.augmenter_names:
                processes.append(str2augm[augmenter_name].load_opt(
                    self.opt, is_training=False))

            for preprocess_name in self.opt.dataset.validation.preprocess_names:
                processes.append(str2preprocess[preprocess_name].load_opt(
                    self.opt, is_training=False))

            validation_dataset = str2dataset[self.opt.dataset.validation.dataset].load_opt(
                self.opt, processes, is_training=False)
            validation_loaders = DataLoader(
                **{'opt': self.opt, 'dataset': validation_dataset, 'is_training': False}).load_dataloader()

        epoch = step = 0
        self.steps = 0
        if self.opt.optimize_settings.restore_path != None:
            checkpoint_instance = Checkpoint(self.opt)
            checkpoint_instance.restore_model(
                self.model, self.device, self.logger)
            epoch, iter_delta = checkpoint_instance.restore_counter()
            self.steps = epoch * self.total + iter_delta

        # Init start epoch and iter
        optimizer = OptimizerScheduler(self.opt).create_optimizer(
            self.model.parameters())

        self.logger._report_time("Init")

        self.model.train()
        while True:
            self.logger.info('Training epoch ' + str(epoch))
            self.logger.epoch(epoch)
            self.total = len(train_data_loader)

            for batch in train_data_loader:
                self.update_learning_rate(optimizer, epoch, self.steps)

                if self.opt.dataset.validation.activate and\
                        self.steps % self.opt.dataset.validation.interval == 0 and\
                        self.steps > self.opt.dataset.validation.exempt:
                    metrics = self.validate(
                        validation_loaders, self.model, epoch, self.steps)
                    print("Metricss", metrics)
                    if metrics['f1'] > self.best_f1:
                        self.best_f1 = metrics['f1']
                        self.model_saver.save_checkpoint(self.model, self.dict_config, "epoch_{}_f1_{:.4f}".format(
                            epoch, float(self.best_f1)))

                if self.logger.verbose:
                    torch.cuda.synchronize()

                self.train_step(self.model, optimizer, batch,
                                epoch=epoch, step=self.steps)
                if self.logger.verbose:
                    torch.cuda.synchronize()
                # self.logger._report_time('Forwarding ')

                self.steps += 1
                self.logger._report_eta(self.steps, self.total, epoch)

            epoch += 1
            if epoch > self.opt.optimize_settings.epochs:
                self.model_saver.save_checkpoint(
                    self.model, self.dict_config, 'final')

                if self.opt.dataset.validation.activate:
                    self.validate(validation_dataset,
                                  self.model, epoch, self.steps)
                self.logger.info('Training done')
                break
            iter_delta = 0

    def train_step(self, model, optimizer, batch, epoch, step, **kwards):
        optimizer.zero_grad()

        # print(batch.keys())s
        results = model.forward(_input=batch, is_training=True)
        if len(results) == 2:
            l, pred = results
            metrics = {}
        elif len(results) == 3:
            l, pred, metrics = results

        if isinstance(l, dict):
            line = []
            loss = torch.tensor(0.).cuda()
            for key, l_val in l.items():
                loss += l_val.mean()
                line.append('loss_{0}:{1:.4f}'.format(key, l_val.mean()))
        else:
            loss = l.mean()
        loss.backward()
        optimizer.step()

        if step % self.opt.logging.log_interval == 0:
            if isinstance(l, dict):
                line = '\t'.join(line)
                log_info = '\t'.join(['step:{:6d}', 'epoch:{:3d}', '{}', 'lr:{:.4f}']).format(
                    step, epoch, line, self.current_lr)
                if self.min_loss > line:
                    self.min_loss = line
                    self.model_saver.save_model_with_loss(
                        model, self.dict_config, epoch, self.steps, self.min_loss)
                    self.logger._report_time('Saving ')
                self.logger.info(log_info)
            else:
                if self.min_loss > loss.item():
                    self.min_loss = loss.item()
                    self.model_saver.save_model_with_loss(
                        model, self.dict_config, epoch, self.steps, self.min_loss)
                    self.logger._report_time('Saving ')
                self.logger.info('step: %6d, epoch: %3d, loss: %.6f, lr: %f' % (
                    step, epoch, loss.item(), self.current_lr))

            self.logger.add_scalar('loss', loss, step)
            self.logger.add_scalar('learning_rate', self.current_lr, step)
            for name, metric in metrics.items():
                self.logger.add_scalar(name, metric.mean(), step)
                self.logger.info('%s: %6f' % (name, metric.mean()))

            self.logger._report_time('Logging')

    def validate(self, validation_loader, model, epoch, step):
        model.eval()
        all_matircs = self.validate_step(validation_loader, model, 0.3)
        # print(">>>>", metrics)
        model.train()
        return all_matircs

    def calculate_iou_dice(self, y_pred, y_true):
        intersection = np.sum(np.abs(y_pred * y_true))
        mask_sum = np.sum(np.abs(y_true)) + np.sum(np.abs(y_pred))
        union = mask_sum - intersection

        smooth = .001
        iou = (intersection + smooth) / (union + smooth)
        dice = 2 * (intersection + smooth)/(mask_sum + smooth)

        iou = np.mean(iou)
        dice = np.mean(dice)

        return (iou, dice)

    def validate_step(self, data_loader, model, thresh=0.3, visualize=False):
        metrics = dict()
        ious = []
        dices = []
        f1s = []
        losses = []
        for i, batch in tqdm(enumerate(data_loader), total=len(data_loader)):
            # for batch in data_loader:
            preds = model.forward(batch, is_training=False)

            for idx, pred in enumerate(preds):
                y_pred = ((pred.cpu().detach().numpy()
                           [0] > thresh)).astype(int)
                y_true = batch['gt'][idx][0].numpy().astype(int)

                import pdb
                pdb.set_trace()
                iou, dice = self.calculate_iou_dice(y_pred, y_true)

                intersection = np.logical_and(y_true, y_pred)
                union = np.logical_or(y_true, y_pred)
                iou = np.sum(intersection) / np.sum(union)
                f1 = f1_score(y_true, y_pred, average="macro")

                ious.append(iou)
                dices.append(dice)
                f1s.append(f1)

        metrics = {'iou': sum(
            ious) / len(ious), 'dice': sum(dices) / len(dices), 'f1': sum(f1s) / len(f1s)}
        print("Real metrics: {}".format(metrics))
        return metrics

    def save(self, weights_path=None):
        self.model_saver.save_checkpoint_with_specific_path(
            self.model, self.dict_config, weights_path)

    def load(self, weights_path=None):
        checkpoint_instance = Checkpoint(self.opt, weights_path)
        checkpoint_instance.restore_model(self.model, self.device, self.logger)
