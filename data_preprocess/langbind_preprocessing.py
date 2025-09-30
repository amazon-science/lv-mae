import sys
import os

# Get the directory of the current script
current_file_directory = os.path.dirname(os.path.realpath(__file__))
parent_directory = os.path.dirname(current_file_directory)
SRC = os.path.join(parent_directory, 'data_preprocess')

# Add the constructed path to sys.path
sys.path.append(SRC)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from languagebind import LanguageBind, to_device, transform_dict

from tqdm import tqdm
import torch
import argparse
from more_itertools import chunked

from data_preprocess.helper_functions import load_json

DEFAULT_VIDEO_PATH_FILE = os.path.join(SRC, 'test_assets/prepro_path_tuples.json') # use your saved path_tuples
DEFAULT_END_JOB_FILE = '~/tmp/tmp_end_job_file.txt'


def main():
    # Load and initialize LanguageBind model
    parser = argparse.ArgumentParser(description="Run preprocessing jobs on a dataset.")
    parser.add_argument('--video_path_file', type=str, default=DEFAULT_VIDEO_PATH_FILE)  # default values are only for testing.
    parser.add_argument('--end_job_file', type=str, default='~/tmp/tmp_end_job_file.txt')  # default values are only for testing.
    args = parser.parse_args()

    # Model initialization
    batch_size = 16
    video_model_name = 'LanguageBind_Video_Huge_V1.5_FT'
    device = 'cuda'
    clip_type = {'video': video_model_name}  # use only video type

    model = LanguageBind(clip_type=clip_type, cache_dir='./cache_dir')
    model = model.to(device)
    model.eval()
    modality_transform = {c: transform_dict[c](model.modality_config[c]) for c in clip_type.keys()}

    # Iterate over batches
    video_paths = load_json(args.video_path_file)['file_paths']
    # for batch_idx, paths in tqdm(enumerate(video_paths)):
    chunked_videos = list(chunked(video_paths, batch_size))
    for i, batch in enumerate(tqdm(chunked_videos, total=len(chunked_videos))):

        # in_path, out_path = batch[0], batch[1]
        in_paths = [x[0] for x in batch]
        out_paths = [x[1] for x in batch]

        inputs = {'video': to_device(modality_transform['video'](in_paths), device)}

        ###########################################################################################################

        with torch.no_grad():
            embeddings = model(inputs)
        embeddings = embeddings['video'].to('cpu')

        ###########################################################################################################

        for k, out_path in enumerate(out_paths):
            os.makedirs(out_path, exist_ok=True)
            save_path = os.path.join(out_path, 'langbind_output.pt')
            start_time_str, end_time_str = os.path.basename(out_path).split('-')

            # Convert the split strings to integers
            start_time = int(start_time_str)
            end_time = int(end_time_str)
            res_dict = {
                'embedding': embeddings[k],
                'start': start_time,
                'end': end_time,
            }
            torch.save(res_dict, save_path)

    os.makedirs(os.path.split(args.end_job_file)[0], exist_ok=True)
    with open(args.end_job_file, 'w') as file:
        file.write("finished")


if __name__ == '__main__':
    main()