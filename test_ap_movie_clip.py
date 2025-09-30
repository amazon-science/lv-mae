import argparse
import numpy as np
import os
import sys
import json

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

import util.misc as misc

import models_mae
from data.MovieClipsData import MovieClipsData

SRC = os.path.join(os.path.realpath(__file__).split('lv_mae_code')[0], 'lv_mae_code')
sys.path.append(SRC)

from tqdm import tqdm


class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'    # Reset to default
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def get_args_parser():
    parser = argparse.ArgumentParser('LV-MAE downstream task - attentive probing', add_help=False)
    parser.add_argument('--task', default='scene', type=str, help='downstream task name',
                        choices=['relationship', 'way_speaking', 'scene', 'director', 'genre', 'writer', 'year'])
    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--batch_size', default=16, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--lr', type=float, default=5e-4, help='learning rate for linear probing eval')

    # Model parameters
    parser.add_argument('--model', default='mae_vit_base_custom', type=str, metavar='MODEL', help='Name of model to train')
    parser.add_argument('--input_size', default=1024, type=int, help='images input size')
    parser.add_argument('--mask_ratio', default=0.6, type=float, help='Masking ratio (percentage of removed patches).')
    parser.add_argument('--norm_pix_loss', action='store_true', help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)
    parser.add_argument('--depth', default=32, type=int, help='images input size')
    parser.add_argument('--num_heads', default=16, type=int, help='images input size')
    parser.add_argument('--decoder_embed_dim', default=512, type=int, help='images input size')
    parser.add_argument('--decoder_depth', default=8, type=int, help='images input size')
    parser.add_argument('--decoder_num_heads', default=16, type=int, help='images input size')
    parser.add_argument('--cos_loss', action='store_true', help='cosine similarity loss')

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05, help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=None, metavar='LR', help='learning rate (absolute lr)')

    # Dataset parameters
    parser.add_argument('--data_path', default='/mnt/efs0/naimani/data', type=str, help='dataset path')
    parser.add_argument('--dataset', default='combined', type=str, help='dataset name', choices=['combined', 'prime_video', 'movie_clip', 'activity'])
    parser.add_argument('--output_dir', default='./output_dir', help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir', help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda:0', help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--long_term_task', default='scene', help='classification task to be solved')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--num_patches', default=256, type=int)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true', help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    return parser


def load_pretrained_model(args):
    # Initialize Our LVMAE model
    model_our = models_mae.__dict__[args.model](args=args, norm_pix_loss=args.norm_pix_loss)
    model_our.load_state_dict(torch.load(args.resume, map_location=torch.device(args.device))['model'])
    model_our.to(args.device)
    model_our.eval()
    print('Model loaded')
    return model_our


def load_data(args):
    import pandas as pd
    base_dir = os.path.dirname(os.path.realpath(__file__))
    segments_metadata_train = os.path.join(base_dir, f'data_preprocess/movie_clip/lvu_1.0_langbind/{args.long_term_task}/train.csv')
    segments_metadata_val = os.path.join(base_dir, f'data_preprocess/movie_clip/lvu_1.0_langbind/{args.long_term_task}/val.csv')
    segments_metadata_test = os.path.join(base_dir, f'data_preprocess/movie_clip/lvu_1.0_langbind/{args.long_term_task}/test.csv')
    dataset_train = MovieClipsData(args=args, csv_file=segments_metadata_train)
    dataset_val = MovieClipsData(args=args, csv_file=segments_metadata_val)
    dataset_test = MovieClipsData(args=args, csv_file=segments_metadata_test)

    df = pd.read_csv(segments_metadata_train)

    # Extract the number of unique values in the 'label' column
    num_classes = df['label'].nunique()

    # Initialize DataLoaders
    data_loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True,
                                                    num_workers=args.num_workers, pin_memory=args.pin_mem,
                                                    drop_last=True)
    data_loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False,
                                                  num_workers=args.num_workers, pin_memory=args.pin_mem,
                                                  drop_last=False)
    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False,
                                                   num_workers=args.num_workers, pin_memory=args.pin_mem,
                                                   drop_last=False)
    # print(dataset_train)
    print(f"Training dataset size: {len(dataset_train)}")
    print(f"Validation dataset size: {len(dataset_val)}")
    print(f"Test dataset size: {len(dataset_test)}")

    return dataset_train, data_loader_train, data_loader_val, data_loader_test, num_classes


class AttentiveProbing(nn.Module):
    def __init__(self, model, input_dim, num_classes, num_heads=8):
        super(AttentiveProbing, self).__init__()
        from models_mae import Block
        self.enc = model
        self.blocks = nn.ModuleList([
            Block(input_dim, num_heads, 4., qkv_bias=True, norm_layer=nn.LayerNorm)
            for i in range(1)])
        self.norm = nn.LayerNorm(input_dim)
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x, attn_mask=None):

        with torch.no_grad():
            x = self.enc(x, attn_mask)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x, attn_mask)
        x = self.norm(x)

        CLS_TOKEN = x[:, 0]

        return self.linear(CLS_TOKEN)


# Training loop
def train_one_epoch_lp(probe, dataloader, optimizer, criterion, device):

    probe.train()  # We only train the linear layer

    total_loss = 0
    correct = 0
    total = 0

    for batch in dataloader:
        inputs = batch['normalized_stacked_embeddings'].to(device)
        attn_mask = batch['attn_mask'].to(device)
        labels = batch['label'].to(device)

        # Forward pass through the linear probe
        logits = probe(inputs, attn_mask)

        # Compute loss
        loss = criterion(logits, labels)
        total_loss += loss.item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Compute accuracy
        _, preds = torch.max(logits, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    print(f"Train Loss: {avg_loss:.4f}, Train Accuracy: {accuracy:.4f}")
    return avg_loss, accuracy


def evaluate(probe, dataloader, criterion, device):
    probe.eval()  # Linear probe should also be in eval mode

    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            inputs = batch['normalized_stacked_embeddings'].to(device)
            attn_mask = batch['attn_mask'].to(device)
            labels = batch['label'].to(device)

            logits = probe(inputs, attn_mask)
            loss = criterion(logits, labels)
            total_loss += loss.item()

            # Compute accuracy
            _, preds = torch.max(logits, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    print(f"Eval Loss: {avg_loss:.4f}, Eval Accuracy: {accuracy:.4f}")
    return avg_loss, accuracy


def fine_tuning(args, model, data_loader_train, data_loader_val, data_loader_test, num_classes, device):

    input_dim = args.input_size
    ft_attentive_probe = AttentiveProbing(model=model.forward_encoder, input_dim=input_dim, num_classes=num_classes).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(ft_attentive_probe.parameters(), lr=args.lr, amsgrad=True, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Train and evaluate
    num_epochs = args.epochs
    accuracy = 0.
    test_accuracy = 0.
    test_avg_loss = -1.
    for epoch in tqdm(range(num_epochs)):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        train_one_epoch_lp(ft_attentive_probe, data_loader_train, optimizer, criterion, device)
        print("Validation performance:")
        val_avg_loss_, val_accuracy_ = evaluate(ft_attentive_probe, data_loader_val, criterion, device)
        print(f"{Colors.OKGREEN}Our Test performance:")
        avg_loss_, accuracy_ = evaluate(ft_attentive_probe, data_loader_test, criterion, device)
        if val_accuracy_ >= accuracy:
            accuracy = val_accuracy_
            test_accuracy = accuracy_
            test_avg_loss = avg_loss_
        print(f'{Colors.ENDC}')
        scheduler.step()

    return test_avg_loss, test_accuracy


args = get_args_parser().parse_args()
args.resume = ('/mnt/efs0/naimanil/research/lv_mae_code/logs/prod_combined_b16_e151_modelmae_vit_base_custom_in1024_'
               'mask0.5_wd0.05_warmup40_d32_heads16_dec_embed_dim512_dec_d8_dec_heads16_distributed_True_cosloss_'
               'False_ms_True/checkpoint-150.pth')

model = load_pretrained_model(args)
res_dict = {}

print("Evaluating model on {} task...".format(args.task))

# fix the seed for reproducibility
seed = args.seed + misc.get_rank()
torch.manual_seed(seed)
np.random.seed(seed)
cudnn.benchmark = True

_, data_loader_train, data_loader_val, data_loader_test, num_classes = load_data(args)
print(f"{Colors.OKCYAN}Compute our score...{Colors.ENDC}")
avg_loss, ft_accuracy = fine_tuning(args, model, data_loader_train, data_loader_val, data_loader_test, num_classes, args.device)
res_dict[args.task] = {'ft_accuracy': ft_accuracy}

for task, metrics in res_dict.items():
    print(f"{task}/ft_accuracy: ", metrics['ft_accuracy'])
