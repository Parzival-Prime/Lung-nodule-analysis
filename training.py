import datetime
import argparse
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from util.logconf import logging
from .dsets import LunaDataset
from .model import LunaModel

log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)


METRICS_LABEL_NDX = 0
METRICS_PRED_NDX = 1
METRICS_LOSS_NDX = 2
METRICS_SIZE = 3

class LunaTrainingApp:
    def __init__(self, sys_argv=None):
        if sys_argv is None:
            sys_argv = sys.argv[1:]

        parser = argparse.ArgumentParser()
        parser.add_argument('--num-workers', help='Number of worker processes for background data loading', default=8, type=int)
        parser.add_argument('--batch-size', help='Batch size to use for training', default=32,type=int)
        parser.add_argument('--epochs', help='Number of epochs to train for', default=1, type=int)
        parser.add_argument('--tb-prefix', default='p2ch11', help="Data prefix to use for Tensorboard run. Defaults to chapter.")

        parser.add_argument('comment',
            help="Comment suffix for Tensorboard run.",
            nargs='?',
            default='dwlpt',
        )

        self.cli_args = parser.parse_args(sys_argv)
        self.time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H.%M.%S')

        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda' if self.use_cuda else 'cpu')

        self.model = self.initModel()
        self.optimizer = self.initOptimizer()


    def initModel(self):
        model = LunaModel()
        if self.use_cuda:
            log.info("Using CUDA; {} devices.".format(torch.cuda.device_count()))
            if torch.cuda.device_count() > 1:
                model = nn.DataParallel(model)
            model = model.to(self.device)
        return model

    def initOptimizer(self):
        return optim.SGD(self.model.parameters(), lr=0.001, momentum=0.99)


    def initTrainD1(self):
        train_ds = LunaDataset(val_stride=10, isValSet_Bool=False)
        batch_size = self.cli_args.batch_size
        if self.use_cuda:
            batch_size *= torch.cuda.device_count()

        train_dl = DataLoader(train_ds, batch_size=batch_size, num_workers=self.cli_args.num_workers, pin_memory=self.use_cuda)
        return train_dl

    def initValD1(self):
        val_ds = LunaDataset(val_stride=10, isValSet_Bool=True)
        batch_size = self.cli_args.batch_size
        if self.use_cuda:
            batch_size *= torch.cuda.device_count()

        val_dl = DataLoader(val_ds, batch_size=batch_size, num_workers=self.cli_args.num_workers, pin_memory=self.use_cuda)
        return val_dl

    def doTraining(self, epoch_ndx, train_dl):
        self.model.train()
        trnMetrics_g = torch.zeros(METRICS_SIZE, len(train_dl.dataset), device=self.device)

        batch_iter = enumerateWithEstimate(train_dl, "E{} Training".format(epoch_ndx), start_ndx=train_dl.num_workers)
        for batch_ndx, batch_tup in batch_iter:
            self.optimizer.zero_grad()
            loss_var = self.computeBatchLoss(batch_ndx, batch_tup, train_dl.batch_size, trnMetrics_g)
            loss_var.backward()
            self.optimizer.step()

        self.totalTrainingSamples_count += len(train_dl.dataset)

        return trnMetrics_g.to('cpu')

    def doValidation(self, epoch_ndx, val_dl):
        with torch.no_grad():
            self.model.eval()
            valMetrics_g = torch.zeros(METRICS_SIZE, len(val_dl.dataset), device=self.device)

            batch_iter = enumerateWithEstimate(val_dl, "E{} Validation".format(epoch_ndx), start_ndx=val_dl.num_workers)
            for batch_ndx, batch_tup in batch_iter:
                self.computeBatchLoss(batch_ndx, batch_tup, val_dl.batch_size, valMetrics_g)

        return valMetrics_g.to('cpu')

    def computeBatchLoss(self, batch_ndx, batch_tup, batch_size, metrics_g):
        input_t, label_t, _series_list, _center_list = batch_tup

        input_g = input_t.to(self.device, non_blocking=True)
        label_g = label_t.to(self.device, non_blocking=True)

        logits_g, probability_g = self.model(input_g)
        loss_func = nn.CrossEntropyLoss(reduction='none')
        loss_g = loss_func(logits_g, label_g[:, 1])

        start_ndx = batch_ndx * batch_size
        end_ndx = start_ndx + label_t.size(0)

        metrics_g[METRICS_LABEL_NDX, start_ndx:end_ndx] = label_g[:, 1].detach()
        metrics_g[METRICS_PRED_NDX, start_ndx:end_ndx] = probability_g[:, 1].detach()
        metrics_g[METRICS_LOSS_NDX, start_ndx:end_ndx] = loss_g[:, 1].detach()

        return loss_g.mean()

    def logMetrics(self, epoch_ndx, mode_str, metrics_t, classificationThreshold=0.5):
        negLabel_mask = metrics_t[METRICS_LABEL_NDX] <= classificationThreshold
        negPred_mask = metrics_t[METRICS_PRED_NDX] <= classificationThreshold

        posLabel_mask = ~negLabel_mask
        posPred_mask = ~negPred_mask

        neg_count = int(negLabel_mask.sum())
        pos_count = int(posLabel_mask.sum())

        neg_correct = int((negLabel_mask & negPred_mask).sum())
        pos_correct = int((posLabel_mask & posPred_mask).sum())

        metrics_dict = {}
        metrics_dict['loss/all'] = metrics_t[METRICS_LOSS_NDX].mean()
        metrics_dict['loss/neg'] = metrics_t[METRICS_LOSS_NDX, negLabel_mask].mean()
        metrics_dict['loss/pos'] = metrics_t[METRICS_LABEL_NDX, posLabel_mask].mean()

        metrics_dict['correct/all'] = (pos_correct + neg_correct) / np.float32(metrics_t.shape[1]) * 100
        metrics_dict['correct/neg'] = neg_correct / np.float32(neg_count) * 100
        metrics_dict['correct/pos'] = pos_correct / np.float32(pos_correct) * 100

        log.info("E{} {:8} {loss/all:.4f} loss, {correct/all:-5.1f}% correct".format(epoch_ndx, mode_str, **metrics_dict))
        log.info("E{} {:8} {loss/neg:.4f} loss, {correct/neg:-5.1f}% correct ({neg_correct:} of {neg_count:})".format(epoch_ndx, mode_str + 'neg', neg_correct=neg_correct, neg_count=neg_count, **metrics_dict))
        log.info("E{} {:8} {loss/pos:.4f} loss, {correct/pos:-5.1f}% correct ({pos_correct:} of {pos_count:})".format(epoch_ndx, mode_str + 'pos', pos_correct=pos_correct, pos_count=pos_count, **metrics_dict))


    def main(self):
        log.info("Starting {}, {}".format(type(self).__name__, self.cli_args))
        train_dl = self.initTrainD1()
        val_dl = self.initValD1()

        for epoch_ndx in range(1, self.cli_args.epochs + 1):
            trnMetrics_t = self.doTraining(epoch_ndx, train_dl)
            self.logMetrics(epoch_ndx, 'trn', trnMetrics_t)
            val_metrics_t = self.doValidation(epoch_ndx, val_dl)
            self.logMetrics(epoch_ndx, 'val', val_metrics_t)

if __name__ == '__main__':
    LunaTrainingApp().main()
