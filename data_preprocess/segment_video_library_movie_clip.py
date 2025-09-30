import os
from os import path
import shutil
import subprocess
from joblib import Parallel, delayed
import multiprocessing
import pandas as pd
from tqdm import tqdm
from helper_functions import fetch_video
import argparse

ffmpeg_bin_path = f'/home/linuxbrew/.linuxbrew/bin'
os.environ['PATH'] = f"{os.environ['PATH']}:{ffmpeg_bin_path}"

env_name = 'lv_mae' # your env name
VIDEO_LIBRARY_PATH = "Path_to_movie_clip_video_data" # enter your local path here
SEGMENT_LENGTH_IN_SECONDS = 5  # choose segment length
SEGMENT_VERSION = f"{SEGMENT_LENGTH_IN_SECONDS}_second_segments"
# List of video file extensions to check
video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']


def validate_and_filter_videos(video_files):
    bad_files = []
    for video in video_files:
        try:
            video_frames = fetch_video(video, preprocess=None)
            if len(video_frames) == 0:
                print(f'Zero number of frames for video file {video}')
                bad_files.append(video)
        except Exception as e:
            print(f'Could not read video file {video}')
            bad_files.append(video)

    # print(f'Total number of unreadable video files: {len(bad_files)}')
    video_files = list(filter(lambda x: x not in bad_files, video_files))
    return video_files


def format_time(time: float):
    """
    Format time to have 5 digits with leading zeros if necessary.
    """
    return str(time).zfill(5)


def get_video_duration(video_path):
    """
    Get the duration of the video in seconds.
    """
    result = subprocess.run([f'/opt/conda/envs/{env_name}/bin/ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of',
                             'default=noprint_wrappers=1:nokey=1', video_path], stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
    duration = float(result.stdout)
    return duration


def split_video(video_path):
    video_name = path.basename(video_path)
    video_name_no_ext, ext = path.splitext(video_name)
    save_dir = path.join(path.dirname(video_path), SEGMENT_VERSION)
    os.makedirs(save_dir, exist_ok=True)

    ext = '.mp4'
    duration = get_video_duration(video_path)
    segment_template = os.path.join(save_dir, f"%05d.mp4")
    ffmpeg_command = [
        f"/opt/conda/envs/{env_name}/bin/ffmpeg",
        "-y",  # Overwrite output files without asking for confirmation
        "-i", video_path,
        "-c:v", "libx264",  # Use the H.264 codec for the video
        "-crf", "18",  # Constant Rate Factor (quality), lower values mean higher quality (18-28 is a good range)
        "-preset", "slow",  # Preset for better quality (slower encoding)
        "-c:a", "copy",  # Copy audio stream without re-encoding
        # force a keyframe every SEGMENT_LENGTH_IN_SECONDS seconds (to ensure that the segments are of equal length)
        f"-force_key_frames", f'expr:gte(t,n_forced*{SEGMENT_LENGTH_IN_SECONDS})',
        "-f", "segment",
        "-segment_time", str(SEGMENT_LENGTH_IN_SECONDS),
        "-reset_timestamps", "1",  # reset timestamps at the beginning of each segment (each one start at timestamp 0)
        segment_template
    ]
    subprocess.run(ffmpeg_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Create a directory for each segment according to its timestamp:
    for entry in os.listdir(save_dir):
        entry_path = path.join(save_dir, entry)
        if path.isfile(entry_path) and entry.endswith(ext):
            segment_number = int(entry.split('.')[0])
            start_time = format_time(segment_number * SEGMENT_LENGTH_IN_SECONDS)
            end_time = format_time(min((segment_number + 1) * SEGMENT_LENGTH_IN_SECONDS, int(duration)))
            segment_timestamp = f"{start_time}-{end_time}"
            segment_dir_path = path.join(save_dir, segment_timestamp)
            os.makedirs(segment_dir_path, exist_ok=True)
            new_file_path = path.join(segment_dir_path, f'video{ext}')
            shutil.move(entry_path, new_file_path)


def process_video_directory():

    parser = argparse.ArgumentParser(description="Run preprocessing jobs on a dataset.")
    parser.add_argument('--data_paths_cache', action='store_true', help="Use dataset paths cache usage")
    args = parser.parse_args()

    video_paths = []

    for entry in tqdm(os.listdir(VIDEO_LIBRARY_PATH)):
        entry_path = os.path.join(VIDEO_LIBRARY_PATH, entry)

        # Check if the file has a valid video extension
        for file in os.listdir(entry_path):
            if any(file.endswith(ext) for ext in video_extensions):
                video_paths.append(os.path.join(entry_path, file))


    if not args.data_paths_cache:
        # Parallel validation
        video_paths = Parallel(n_jobs=48)(
            delayed(validate_and_filter_videos)([video_path])
            for video_path in tqdm(video_paths, desc="Validating and filtering videos")
        )

        # Flatten the list of lists returned by Parallel
        video_paths = [item for sublist in video_paths for item in sublist]

        # Filter out any empty paths (if a list was returned empty)
        video_paths = [path for path in video_paths if path]

        # Save valid video paths to a file
        output_file = "valid_video_paths_movie_clip.txt"
        with open(output_file, "w") as f:
            for path in video_paths:
                f.write(path + "\n")

        print(f"Saved {len(video_paths)} valid video paths to {output_file}.")
    else:
        with open("valid_video_paths_movie_clip.txt", "r") as f:
            video_paths = [line.strip() for line in f.readlines()]

    num_cores_to_use = min(multiprocessing.cpu_count(), len(video_paths))
    print(f'Processing the video library at: {VIDEO_LIBRARY_PATH}')
    print(f'Splitting the video library into {SEGMENT_LENGTH_IN_SECONDS}-second segments using {num_cores_to_use}'
          f' CPU cores...')
    Parallel(n_jobs=num_cores_to_use, backend='multiprocessing')(delayed(split_video)(p) for p in tqdm(video_paths))
    # for p in tqdm(video_paths):
    #     split_video(p)
    # Create a DataFrame with details of video segments:
    print('Creating a dataframe with the metadata of the generated video segments...')
    segment_details = []
    for entry in sorted(os.listdir(VIDEO_LIBRARY_PATH)):
        entry_path = os.path.join(VIDEO_LIBRARY_PATH, entry)
        if os.path.isdir(entry_path):
            for file in os.listdir(entry_path):
                if any(file.endswith(ext) for ext in video_extensions):
                    original_video_name = file
                    video_segments_path = os.path.join(entry_path, SEGMENT_VERSION)
                    if os.path.exists(video_segments_path):
                        for segment_timestamp in os.listdir(video_segments_path):
                            segment_details.append({
                                'segment_idx': len(segment_details),
                                'segment_timestamp': segment_timestamp,
                                'segment_version': SEGMENT_VERSION,
                                'full_video_directory': entry,
                                'full_video_filename': original_video_name,
                            })

    segments_df = pd.DataFrame(segment_details)
    segment_version_dir_path = os.path.join(VIDEO_LIBRARY_PATH, f'{SEGMENT_VERSION}_metadata')
    os.makedirs(segment_version_dir_path, exist_ok=True)
    segments_df_save_path = os.path.join(segment_version_dir_path, f'segments_metadata.csv')
    segments_df.to_csv(segments_df_save_path, index=False)
    print(f'The metadata dataframe was saved at: {segments_df_save_path}')
    print(f'the dataframe contains metadata for {len(segment_details)} video segments')


if __name__ == "__main__":
    process_video_directory()