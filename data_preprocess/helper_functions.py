import json

import cv2
from PIL import Image


def load_json(file_path):
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None
    except json.JSONDecodeError:
        print(f"Error: Unable to decode JSON from file '{file_path}'.")
        return None

def read_text_file(path):
    with open(path, 'r') as file:
        text = file.read()
    return text


def fetch_video(video_file, frame_rate=2, preprocess=True):
    cap = cv2.VideoCapture(video_file)
    frame_count = 0
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        # Only extract frames at the desired frame rate
        if frame_count % int(cap.get(5) / frame_rate) == 0:

            if preprocess is not None:
                # per-frame preprocessing
                frame = Image.fromarray(frame)
                frame = preprocess(frame)
            frames.append(frame)

    cap.release()
    cv2.destroyAllWindows()
    return frames