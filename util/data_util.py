import torch
from torch.utils.data.distributed import DistributedSampler
from util import misc
from data.CombinedData import CombinedData


def load_data(args):

    if args.dataset == 'combined':
        train_segments_metadata = 'path_to_preprocessed_all_langbind_embeddings/segments_metadata.csv'
        val_segments_metadata = 'path_to_preprocesses_activity_net_dataset'

        # Create the full dataset
        dataset_tr = CombinedData(csv_file=train_segments_metadata, mask_ratio=args.mask_ratio, num_patches=args.num_patches, duration_max=args.duration_max, importance_masking_strategy=args.masking_strategy)
        dataset_val = CombinedData(csv_file=val_segments_metadata, mask_ratio=args.mask_ratio, num_patches=args.num_patches, duration_max=args.duration_max, importance_masking_strategy=args.masking_strategy)

        if args.distributed:
            data_loader_train = torch.utils.data.DataLoader(dataset_tr, batch_size=args.batch_size,
                                                            num_workers=args.num_workers, pin_memory=args.pin_mem,
                                                            drop_last=True, sampler=DistributedSampler(dataset_tr))
            data_loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=args.batch_size,
                                                          num_workers=args.num_workers, pin_memory=args.pin_mem,
                                                          drop_last=False, sampler=DistributedSampler(dataset_val))

        else:
            data_loader_train = torch.utils.data.DataLoader(dataset_tr, batch_size=args.batch_size, shuffle=True,
                                                            num_workers=args.num_workers, pin_memory=args.pin_mem,
                                                            drop_last=True)
            data_loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False,
                                                          num_workers=args.num_workers, pin_memory=args.pin_mem,
                                                          drop_last=False)

        if misc.is_main_process():
            print(f"Training dataset size: {len(dataset_tr)}")
            print(f"Validation dataset size: {len(dataset_val)}")

    else:
        raise "Not implemented"

    return data_loader_train, data_loader_val