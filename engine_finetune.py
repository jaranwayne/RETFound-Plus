# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import math
import sys
import numpy as np
from typing import Iterable, Optional

import torch

from timm.data import Mixup
from timm.utils import accuracy

import util.misc as misc
import util.lr_sched as lr_sched

from util.utils import loglogistic_activation, logistic_nll_loss, concordance_index, integrated_brier_score, logistic_survival_fn_gen


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    log_writer=None, args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    
    for data_iter_step, (samples, tte, events) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        tte = tte.to(device, non_blocking=True)
        events = events.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            outputs = model(samples)
            mu, log_sigma = loglogistic_activation(outputs)
            loss = logistic_nll_loss(mu, log_sigma, tte, events)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, args, mode='test', eval_times=None):

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'
    
    # switch to evaluation mode
    model.eval()

    t_list, e_list, m_list, s_list =[], [], [], []
    for images, tte, events  in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        tte = tte.to(device, non_blocking=True)
        events = events.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            mu, log_sigma = loglogistic_activation(output)
            sigma = torch.exp(log_sigma)

            t_list.append(tte)
            e_list.append(events)
            m_list.append(mu)
            s_list.append(sigma)

        batch_size = images.shape[0]
        metric_logger.update(loss=0)

    m_np = torch.cat(m_list,dim=0).cpu().numpy()
    s_np = torch.cat(s_list,dim=0).cpu().numpy()
    e_np = torch.cat(e_list,dim=0).cpu().numpy()
    t_np = torch.cat(t_list,dim=0).cpu().numpy()

    if eval_times is None:
        eval_times = np.quantile(t_np, [0.75])

    c_index_scores, ibs_scores = [], []
    for i, eval_time in enumerate(eval_times):
        c_index_scores.append(concordance_index(eval_time, t_np, m_np ,e_np))
        ibs_scores.append(integrated_brier_score(logistic_survival_fn_gen(m_np, s_np), t_np, e_np, t_max=eval_time))
        print('* c-index {c_index:.3f} ibs {ibs:.3f} at year {eval_time:.2f}'
            .format(c_index=c_index_scores[i], ibs=ibs_scores[i], eval_time=eval_times[i]))
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    return  {"c-index_scores":c_index_scores[-1], "ibs_scores":ibs_scores[-1], 'eval_times': eval_times[-1], 'loss':metric_logger.meters['loss'].global_avg}