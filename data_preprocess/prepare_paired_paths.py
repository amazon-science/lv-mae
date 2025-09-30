import os
import pickle
import pandas as pd

from tqdm import tqdm
from joblib import Parallel, delayed


from helper_functions import fetch_video

def validate_and_filter_videos(in_dataset_path, video_files):
    bad_files = []
    for video in video_files:
        try:
            video_frames = fetch_video(os.path.join(in_dataset_path, video), preprocess=None)
            if len(video_frames) == 0:
                print(f'Zero number of frames for video file {video}')
                bad_files.append(video)
        except Exception as e:
            print(f'Could not read video file {video}')
            bad_files.append(video)

    # print(f'Total number of unreadable video files: {len(bad_files)}')
    video_files = list(filter(lambda x: x not in bad_files, video_files))
    return video_files


def make_in_out_video_paths(video_paths, out_dataset_path):
    path_tuples = []

    for video_path in video_paths:
        # Extracting the relevant parts of the path
        video_id = os.path.basename(os.path.dirname(
            os.path.dirname(os.path.dirname(video_path))))  # Correctly extracts the 'amzn1.dv.vcid...' part
        segment_info = os.path.basename(os.path.dirname(video_path))  # Extracts the '00141-00175' part

        # Construct the output path
        out_path = os.path.join(out_dataset_path, f"{video_id}/{segment_info}")

        # Append the tuple (input path, output path) to the list
        path_tuples.append((os.path.join('/', video_path), out_path))

    return path_tuples


n_jobs = -1
base_out_path = "path_to_your_langbind_future_processed_output/mc_videos_langbind"  # Assuming output is stored in the same base directory
# phase = "train"
out_dataset_path = f"{base_out_path}/prepro"

metadata_path = "path_to_your_metedata/segments_metadata.csv"
df = pd.read_csv(metadata_path)
video_paths = [
    os.path.join('/mnt/efs0/naimanil/data/mc_videos/', df['full_video_directory'][i], df['segment_version'][i],
                 df['segment_timestamp'][i], 'video.mp4') for i in tqdm(range(len(df)))]
# video_paths = validate_and_filter_videos('/', video_paths[:100])
# Parallelize the video validation and filtering with progress bar
video_paths = Parallel(n_jobs=n_jobs)(
    delayed(validate_and_filter_videos)('/', [video_path])
    for video_path in tqdm(video_paths, desc="Validating and filtering videos")
)

# Flatten the list of lists returned by Parallel
video_paths = [item for sublist in video_paths for item in sublist]
print(f'Total number of video files after filtering: {len(video_paths)}')
# video_in_out_paths = make_in_out_video_paths_pv(video_paths, out_dataset_path)

# Parallelize the creation of input-output video paths with progress bar
video_in_out_paths = Parallel(n_jobs=n_jobs)(
    delayed(make_in_out_video_paths)([video_path], out_dataset_path)
    for video_path in tqdm(video_paths, desc="Creating input-output video paths")
)

# Flatten the list of lists returned by Parallel
video_in_out_paths = [item for sublist in video_in_out_paths for item in sublist]