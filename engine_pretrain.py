# --------------------------------------------------------
# References:
# MAE: https://github.com/facebookresearch/mae
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import math
import sys
from typing import Iterable

import torch
import numpy as np

import util.misc as misc
import util.lr_sched as lr_sched


def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    # if log_writer is not None:
    #     print('log_dir: {}'.format(log_writer.log_dir))
    total_loss = []

    for data_iter_step, samples in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        (masked_padded_embeddings, ids_restore, ids_shuffle, masked_length,
         normalized_stacked_embeddings, length, mask, attn_mask) = \
            (samples['masked_padded_embeddings'],
             samples['ids_restore'],
             samples['ids_shuffle'],
             samples['masked_length'],
             samples['normalized_stacked_embeddings'],
             samples['length'],
             samples['mask'],
             samples['attn_mask'])

        masked_padded_embeddings = masked_padded_embeddings.to(device, non_blocking=True)
        normalized_stacked_embeddings = normalized_stacked_embeddings.to(device, non_blocking=True)
        attn_mask = attn_mask.to(device, non_blocking=True)
        length = length.to(device, non_blocking=True)
        ids_shuffle = ids_shuffle.to(device, non_blocking=True)
        ids_restore = ids_restore.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            loss, _, _ = model(masked_padded_embeddings, ids_restore, ids_shuffle, masked_length, normalized_stacked_embeddings,
                length, mask, attn_mask, mask_ratio=args.mask_ratio)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        total_loss.append(loss_value_reduce)

    if log_writer is not None:
        log_writer.log('train_loss', np.mean(total_loss), epoch)
        log_writer.log('lr', optimizer.param_groups[0]["lr"], epoch)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    if misc.is_main_process():
        print("Averaged stats:", metric_logger)
    return {'train_loss': np.mean(total_loss)}



@torch.no_grad()
def validate(model: torch.nn.Module,
                       data_loader: Iterable,
                       device: torch.device,
                       epoch: int,
                       log_writer=None,
                       args=None):
    model.eval()  # Set the model to evaluation mode
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Validation:'
    print_freq = 20

    total_loss = []
    for data_iter_step, samples in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        (masked_padded_embeddings, ids_restore, ids_shuffle, masked_length,
         normalized_stacked_embeddings, length, mask, attn_mask) = \
            (samples['masked_padded_embeddings'],
             samples['ids_restore'],
             samples['ids_shuffle'],
             samples['masked_length'],
             samples['normalized_stacked_embeddings'],
             samples['length'],
             samples['mask'],
             samples['attn_mask'])

        masked_padded_embeddings = masked_padded_embeddings.to(device, non_blocking=True)
        normalized_stacked_embeddings = normalized_stacked_embeddings.to(device, non_blocking=True)
        attn_mask = attn_mask.to(device, non_blocking=True)
        length = length.to(device, non_blocking=True)
        ids_shuffle = ids_shuffle.to(device, non_blocking=True)
        ids_restore = ids_restore.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            loss, _, _ = model(masked_padded_embeddings, ids_restore, ids_shuffle, masked_length, normalized_stacked_embeddings,
                length, mask, attn_mask, mask_ratio=args.mask_ratio)
            
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping validation".format(loss_value))
            sys.exit(1)

        torch.cuda.synchronize()
        metric_logger.update(loss=loss_value)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        total_loss.append(loss_value_reduce)

    if log_writer is not None:
        log_writer.log('val_loss', np.mean(total_loss), epoch)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    if misc.is_main_process():
        print("Averaged stats:", metric_logger)
    return {'val_loss': np.mean(total_loss)}