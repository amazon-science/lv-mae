import os
from os import path
import pandas as pd
from tqdm import tqdm
import argparse

# ------------------------------------------------------------------------------------------------------------------
# Parameters
# ------------------------------------------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='Movie-Clip preprocessing dataframe')
parser.add_argument('--long_term_task', type=str, default='scene', help='Path to the data')
parser.add_argument('--load_embeddings_path', type=str, default=None, help='Path to pre-computed embeddings')
parser.add_argument('--split', type=str, default='train', help='train, val or test')

args = parser.parse_args()


def run_dataframe_builder(args):
    print("Parsed arguments:")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    SRC = os.path.join(os.path.realpath(__file__).split('lv_mae_code')[0], 'lv_mae_code')
    DATA_ROOT = os.path.join(SRC, f'data_preprocess/movie_clip/lvu_1.0/{args.long_term_task}/{args.split}.csv')
    VIDEO_LANGBIND_LIBRARY_PATH = 'path_to_preprocessed_lb_embeddings'
    SEGMENT_VERSION = '5_second_embeddings'
    found = 0
    original = 0
    segment_details = []
    # Create a DataFrame with details of video segments:
    extensions = ['.mp4', '.webm', '.mkv']
    print('Creating a dataframe with the metadata of the generated video segments...')
    with open(DATA_ROOT, 'r') as f:
        f.readline()
        for line in tqdm(f):
            original += 1
            video_id = line.split()[-2].strip()
            label = line.split()[0].strip()
            if args.long_term_task != 'view_count':
                label_name = line.split()[1].strip()
            else:
                label_name = args.long_term_task
            entry_path = f'{VIDEO_LANGBIND_LIBRARY_PATH}/{video_id}'
            if not os.path.exists(entry_path):
                # print("Features not found for video : ", video_id)
                continue
            # for ext in extensions:
            #     if any(file.endswith(ext) for file in os.listdir(entry_path)):
            #         print(f'Video {video_id} already downloaded')
            #         found += 1
            found += 1

            segment_paths = []
            for segment_dir in sorted(os.listdir(entry_path)):
                segment_entry_path = path.join(entry_path, segment_dir)
                for file in os.listdir(segment_entry_path):
                    if file.endswith("langbind_output.pt"):
                        segment_paths.append(path.join(segment_entry_path, file))
            sorted_segment_paths = sorted(segment_paths)
            segment_details.append({
                'segment_idx': len(segment_details),
                'video_filename': video_id,
                'segment_paths': sorted_segment_paths,
                'number_of_segments': len(sorted_segment_paths),
                'label': label,
                'label_name': label_name
            })
    print('Number of found files: {}'.format(found))
    print('Number of original files: {}'.format(original))
    print('Number of missing files: {}'.format(original - found))
    print('% of missing files: {}%'.format(((original - found) / original)*100))
    DEST_ROOT = os.path.join(SRC, f'data_preprocess/movie_clip/lvu_1.0_langbind_5_segments/{args.long_term_task}')
    segments_df = pd.DataFrame(segment_details)
    os.makedirs(DEST_ROOT, exist_ok=True)
    segments_df_save_path = path.join(DEST_ROOT, f'{args.split}.csv')
    segments_df.to_csv(segments_df_save_path, index=False)
    print(f'The metadata dataframe was saved at: {segments_df_save_path}')
    print(f'the dataframe contains metadata for {len(segment_details)} video segments')


tasks = ['director', 'genre', 'like_ratio', 'relationship', 'scene', 'view_count', 'way_speaking', 'writer', 'year']
splits = ['train', 'val', 'test']
for task in tqdm(tasks):
    for split in splits:
        args.long_term_task = task
        args.split = split
        run_dataframe_builder(args)