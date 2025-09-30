# --------------------------------------------------------
# References:
# MAE: https://github.com/facebookresearch/mae
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import argparse
import datetime
import json
import numpy as np
import os
import sys
import time
from pathlib import Path

SRC = os.path.join(os.path.realpath(__file__).split('lv_mae_code')[0], 'lv_mae_code')
sys.path.append(SRC)

import torch
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
import timm

# assert timm.__version__ == "0.3.2"  # version check
import timm.optim.optim_factory as optim_factory

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.logger import get_logger
from loggers import WandbLogger, TqdmLogger, CompositeLogger, BaseLogger

import models_mae as models_mae

from engine_pretrain import train_one_epoch, validate
from util.data_util import load_data

os.environ['MASTER_ADDR'] = '127.0.0.1'
os.environ['MASTER_PORT'] = '12365'

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--batch_size', default=16, type=int, help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=150, type=int)
    parser.add_argument('--validate_every', default=10, type=int)
    parser.add_argument('--accum_iter', default=1, type=int, help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='mae_vit_base_custom', type=str, metavar='MODEL', help='Name of model to train')
    parser.add_argument('--input_size', default=1024, type=int, help='images input size')
    parser.add_argument('--mask_ratio', default=0.7, type=float, help='Masking ratio (percentage of removed patches).')
    parser.add_argument('--norm_pix_loss', action='store_true', help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)
    parser.add_argument('--depth', default=32, type=int, help='images input size')
    parser.add_argument('--num_heads', default=8, type=int, help='images input size')
    parser.add_argument('--decoder_embed_dim', default=512, type=int, help='images input size')
    parser.add_argument('--decoder_depth', default=8, type=int, help='images input size')
    parser.add_argument('--decoder_num_heads', default=16, type=int, help='images input size')
    parser.add_argument('--cos_loss', action='store_true', help='cosine similarity loss')
    parser.add_argument('--masking_strategy', action='store_true', help='Apply importance masking strategy')

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05, help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=None, metavar='LR', help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR', help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR', help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N', help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--data_path', default='/mnt/efs0/naimani/data', type=str, help='dataset path')
    parser.add_argument('--dataset', default='combined', type=str, help='dataset name',
                        choices=['combined'])
    parser.add_argument('--output_dir', default='./output_dir', help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir', help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda:0', help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--num_patches', default=256, type=int)
    parser.add_argument('--duration_max', default=1280, type=int)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true', help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--distributed', action='store_true', help='whether running distributed processes')
    parser.add_argument('--world_size', default=8, type=int, help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--rank', default=0, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--debug', action='store_true', help='debugging')
    parser.add_argument('--wandb', action='store_true', help='use wandb logger')

    # evaluation params
    parser.add_argument('--lp_epochs', default=10, type=int)
    parser.add_argument('--lp_lr', type=float, default=1e-4, help='learning rate for linear probing eval')
    parser.add_argument('--lp_batch_size', type=int, default=16, help='learning rate for linear probing eval')

    return parser

def main(rank, world_size, args):

    torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)

    # Set device based on rank
    torch.cuda.set_device(rank)

    args.rank = rank

    log_writer = CompositeLogger(loggers=[TqdmLogger(no_plot=True),
                                      WandbLogger(stdout=False, stderr=False, rank=args.rank)]
                             ,rank=args.rank) if args.wandb else TqdmLogger(rank=args.rank)

    if misc.is_main_process():
        print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
        print("{}".format(args).replace(', ', ',\n'))
        log_writer.log_hparams(dict(vars(args)))

    # device = torch.device(args.device)
    device = args.rank

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    args.duration_max = args.num_patches * 5
    data_loader_train, data_loader_val = load_data(args)

    os.makedirs(args.log_dir, exist_ok=True)

    # define the model
    model = models_mae.__dict__[args.model](args=args, norm_pix_loss=args.norm_pix_loss)

    model.to(device)

    model_without_ddp = model
    if misc.is_main_process():
        print("Model = %s" % str(model_without_ddp))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    if misc.is_main_process():
        print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
        print("actual lr: %.2e" % args.lr)

        print("accumulate grad iterations: %d" % args.accum_iter)
        print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank], find_unused_parameters=True)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])
        model_without_ddp = model.module

    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    if misc.is_main_process():
        print(optimizer)
    loss_scaler = NativeScaler()

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    # tsne_samples = {}
    if misc.is_main_process():
        print(f'Number of parameters: {sum(p.numel() for p in model.parameters())}')

    if misc.is_main_process():
        print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    args.device = device

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args
        )

        if args.output_dir and (epoch % 10 == 0 or epoch + 1 == args.epochs):
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch, }

        if args.output_dir and misc.is_main_process():
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

        if epoch % args.validate_every == 0 or epoch + 1 == args.epochs:  # Add your validation condition here
            val_stats = validate(
                model, data_loader_val,
                device, epoch,
                log_writer=log_writer,
                args=args
            )

            # Log validation stats
            val_log_stats = {**{f'val_{k}': v for k, v in val_stats.items()}, 'epoch': epoch}

            if args.output_dir and misc.is_main_process():
                with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                    f.write(json.dumps(val_log_stats) + "\n")


    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    torch.distributed.destroy_process_group()


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    dir_name = (f'prod_{args.dataset}_b{args.batch_size}_e{args.epochs}_model{args.model}_'
                f'in{args.input_size}_mask{args.mask_ratio}_wd{args.weight_decay}_'
                f'warmup{args.warmup_epochs}_d{args.depth}_heads{args.num_heads}_'
                f'dec_embed_dim{args.decoder_embed_dim}_dec_d{args.decoder_depth}_'
                f'dec_heads{args.decoder_num_heads}_'
                f'distributed_{args.distributed}_cosloss_{args.cos_loss}_'
                f'ms_{args.masking_strategy}')

    # Creating the path
    args.output_dir = os.path.join('./logs', dir_name)
    args.log_dir = os.path.join('./logs', dir_name)
    mp.spawn(main, args=(args.world_size, args), nprocs=args.world_size, join=True)