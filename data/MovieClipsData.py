import os
import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset


class MovieClipsData(Dataset):
    def __init__(self, args, csv_file, embedding_dim=1024):
        """
        Args:
            csv_file (string): Path to the csv file.
            duration (int): Duration in seconds to determine the subset of paths.
        """
        self.args = args
        self.long_term_task = args.long_term_task
        self.df = pd.read_csv(csv_file)
        self.SEGMENT_LENGTH = 5
        self.max_num_patch = args.num_patches
        self.pad_token = torch.nn.Parameter(torch.zeros(embedding_dim), requires_grad=False)

        if self.long_term_task not in ['view_count', 'like_ratio']:
            # Get the unique labels and sort them
            unique_labels = sorted(self.df['label'].unique())
            # Create a mapping from original label to sequential label
            self.label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Get the list of paths and the corresponding length
        row = self.df.iloc[idx]
        path_list = eval(row['segment_paths'])  # Converting string representation of list to actual list
        number_of_segments = row['number_of_segments']
        duration = number_of_segments * self.SEGMENT_LENGTH
        label = row['label']
        if self.long_term_task != 'view_count':
            label_name = row['label_name']
        else:
            label_name = self.long_term_task

        if self.long_term_task == 'view_count':
            label = float(np.log(float(label)))

            # make zero-mean
            label -= 11.76425435683139
        elif self.long_term_task == 'like_ratio':
            like, dislike = float(label), float(label_name)
            label = like / (like + dislike) * 10.0

            # make zero-mean
            label -= 9.138220535629456
        else:
            label = self.label_to_idx[label]

        # Load and stack the embeddings
        embeddings = []
        for path in path_list:
            data = torch.load(path)
            embeddings.append(data['embedding'])

        stacked_embeddings = torch.stack(embeddings, dim=0)
        normalized_stacked_embeddings = torch.nn.functional.normalize(stacked_embeddings, p=2, dim=-1)

        n_pad = self.max_num_patch - number_of_segments

        if n_pad > 0:
            ##### pad everything else:
            normalized_stacked_embeddings = torch.cat((normalized_stacked_embeddings,
                                                  torch.stack([self.pad_token] * n_pad, dim=0)),
                                                 dim=0)

            attn_mask = torch.cat((torch.zeros(number_of_segments, dtype=torch.bool),
                                      torch.ones(n_pad+1, dtype=torch.bool)), dim=0)

        else:
            attn_mask = torch.zeros(number_of_segments, dtype=torch.bool)

        return {'normalized_stacked_embeddings': normalized_stacked_embeddings,
                'attn_mask': attn_mask,
                'label': label,
                'label_name': label_name,
                'duration': duration,
                'length': number_of_segments}