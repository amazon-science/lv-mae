import os
import torch
import random
import pandas as pd
from torch.utils.data import Dataset

# 1280
class CombinedData(Dataset):
    def __init__(self, csv_file, duration_min=10, duration_max=1280, embedding_dim=1024, num_patches=120, mask_ratio=.4, importance_masking_strategy=False):
        """
        Args:
            csv_file (string): Path to the csv file.
            duration (int): Duration in seconds to determine the subset of paths.
        """
        self.df = pd.read_csv(csv_file)

        self.duration_min = duration_min
        self.duration_max = duration_max

        self.SEGMENT_LENGTH = 5
        # self.SEGMENT_LENGTH = 10
        self.max_len = num_patches
        self.pad_token = torch.nn.Parameter(torch.zeros(embedding_dim), requires_grad=False)

        self.mask_ratio = mask_ratio

        self.importance_masking_strategy = importance_masking_strategy


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        max_attempts = 10  # Prevent infinite loop
        for _ in range(max_attempts):
            try:
                # Get the list of paths and the corresponding length
                row = self.df.iloc[idx]
                path_list = eval(row['segment_paths'])  # Converting string representation of list to actual list
                number_of_segments = row['number_of_segments']

                while number_of_segments * (1 - self.mask_ratio) < 1:
                    # Get the list of paths and the corresponding length
                    idx = random.randint(0, len(self.df) - 1)
                    row = self.df.iloc[idx]
                    path_list = eval(row['segment_paths'])  # Converting string representation of list to actual list
                    number_of_segments = row['number_of_segments']

                # Sample video duration between duration_min and duration_max. If it exceeds take the max len.
                self.duration = torch.randint(self.duration_min, self.duration_max + 1, (1,)).item()

                # Calculate the number of paths to extract based on the duration
                subset_length = self.duration // self.SEGMENT_LENGTH
                # this is also the unpadded length of the sample

                if number_of_segments < subset_length:
                    subset_length = number_of_segments

                # Sample the starting index j
                max_j = number_of_segments - subset_length
                j = torch.randint(0, max_j + 1, (1,)).item()

                # Extract the subset of paths
                subset_paths = path_list[j:j + subset_length]

                # Load and stack the embeddings
                embeddings = []
                for path in subset_paths:
                    data = torch.load(path)
                    embeddings.append(data['embedding'])

                if number_of_segments < 2:
                    embeddings = embeddings * 8
                    subset_length = subset_length * 8
                elif number_of_segments < 5:
                    embeddings = embeddings * 3
                    subset_length = subset_length * 3

                stacked_embeddings = torch.stack(embeddings, dim=0)
                normalized_stacked_embeddings = torch.nn.functional.normalize(stacked_embeddings, p=2, dim=-1)

                if self.importance_masking_strategy:
                    masked_embeddings, mask, ids_restore, ids_shuffle = self.importance_masking(
                        normalized_stacked_embeddings)
                else:
                    # Random
                    masked_embeddings, mask, ids_restore, ids_shuffle = self.random_masking(
                        normalized_stacked_embeddings)

                masked_length = masked_embeddings.shape[0]

                n_pad_masked = self.max_len - masked_length
                n_pad_original = self.max_len - subset_length

                # emebddings_padded = embeddings + [self.pad_token] * (self.max_len - masked_embeddings.shape[0])
                # Pad the embedding to max len
                if n_pad_masked > 0:
                    masked_padded_embeddings = torch.cat((masked_embeddings,
                                                          torch.stack([self.pad_token] * n_pad_masked, dim=0)), dim=0)

                    attn_mask = torch.cat((torch.zeros(masked_length + 1, dtype=torch.bool),
                                           torch.ones(n_pad_masked, dtype=torch.bool)), dim=0)

                else:
                    masked_padded_embeddings = masked_embeddings
                    attn_mask = torch.zeros(masked_length + 1, dtype=torch.bool)

                if n_pad_original > 0:
                    ##### pad everything else:
                    normalized_stacked_embeddings = torch.cat((normalized_stacked_embeddings,
                                                               torch.stack([self.pad_token] * n_pad_original, dim=0)),
                                                              dim=0)
                    mask = torch.cat((mask, torch.zeros(n_pad_original, dtype=mask.dtype)), dim=0)
                    ids_shuffle = torch.cat((ids_shuffle, torch.zeros(n_pad_original, dtype=ids_shuffle.dtype)), dim=0)
                    ids_restore = torch.cat((ids_restore, torch.zeros(n_pad_original, dtype=ids_restore.dtype)), dim=0)

                return {'masked_padded_embeddings': masked_padded_embeddings,
                        'attn_mask': attn_mask,
                        'normalized_stacked_embeddings': normalized_stacked_embeddings,
                        'mask': mask,
                        'ids_restore': ids_restore,
                        'ids_shuffle': ids_shuffle,
                        'masked_length': masked_length,
                        'length': subset_length}

            except Exception as e:
                print(f"Error loading file {path}: {str(e)}")
                # Sample a new index
                idx = random.randint(0, len(self.df) - 1)

        # If we've tried max_attempts times and still failed, raise an exception
        raise RuntimeError(f"Failed to load data after {max_attempts} attempts")
    
    def random_masking(self, x):

        L, D = x.shape  # batch, length, dim

        len_keep = int(L * (1 - self.mask_ratio))

        noise = torch.rand(L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=0)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=0)

        # keep the first subset
        ids_keep = ids_shuffle[:len_keep]
        x_masked = torch.gather(x, dim=0, index=ids_keep.unsqueeze(-1).repeat(1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([L], device=x.device)
        mask[:len_keep] = 0

        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=0, index=ids_restore)

        return x_masked, mask, ids_restore, ids_shuffle

    def importance_masking(self, x):

        L, D = x.shape  # length, dim

        len_keep = int(L * (1 - self.mask_ratio))

        x_past = x[:-1]
        x_future = x[1:]

        cos_similarities = torch.nn.functional.cosine_similarity(x_past, x_future, dim=1)
        cos_similarities = torch.cat([torch.zeros(1), cos_similarities], dim=0)
        # small cos_similarity means change, so we want to remove those and keep large. (always keep first)

        # sort noise for each sample
        ids_shuffle = torch.argsort(cos_similarities, dim=0)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=0)

        # keep the first subset
        ids_keep = ids_shuffle[:len_keep]
        x_masked = torch.gather(x, dim=0, index=ids_keep.unsqueeze(-1).repeat(1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([L], device=x.device)
        mask[:len_keep] = 0

        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=0, index=ids_restore)

        return x_masked, mask, ids_restore, ids_shuffle