from __future__ import annotations
import os
import json
import glob
import tqdm
import argparse
import pickle as pkl
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation

# Project root is two levels up from scripts/navqa_scripts/
PROJECT_ROOT = str(Path(__file__).resolve().parents[2])

DATA_CSV = os.path.join(PROJECT_ROOT, "data_vlm", "navqa", "data.csv")
DATA_PATH = os.path.join(PROJECT_ROOT, "data_vlm")

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def run_video_in_segs(args):
    SEQUENCE_ID=args.seq_id
    # load folders
    pkl_files = glob.glob(os.path.join(args.data_path, str(SEQUENCE_ID), '*.pkl'))
    pkl_files.sort(key=lambda x: float(x.split('/')[-1][:-4]))

    times = [float(x.split('/')[-1][:-4]) for x in pkl_files]

    segments = []
    current_segment = []
    time_start = times[0]
    for t, file in zip(times, pkl_files):
        if t - time_start > args.seconds_per_caption:
            # Then start over. Add the previous group. This item is the first of the new group
            segments.append(current_segment)
            current_segment = [file]
            time_start = t
        else:
            # Add current file to group
            current_segment.append(file)

    convert_path = args.out_path.format(seq_id=SEQUENCE_ID)
    if os.path.exists(convert_path) and not args.overwrite:
        print(f"Output directory {convert_path} already exists. Use --overwrite to overwrite.")
        exit()
    os.makedirs(convert_path, exist_ok=True)

    outputs = []
    
    for i, file_names in tqdm.tqdm(enumerate(segments), total=len(segments)):
        images_filenames = []
        position = []
        rotation = []
        timestamp = []
        
        for file in file_names:
            # file is the absolute path to the pkl file
            # file = "/home/yhu/codespace/nav_mem/remembr/coda_data/0/1673884185.689107.pkl"
            images_filenames.append(file)
            with open(file, 'rb') as f:
                data = pkl.load(f)
                position.append(data['position'])
                rotation.append(data['rotation'])
                timestamp.append(data['timestamp'])
        
        position = np.array(position)
        rotation = np.array(rotation)
        rotation = Rotation.from_quat(rotation).as_euler('xyz', degrees=True)
        timestamp = np.array(timestamp)

        images_filenames = images_filenames[::30//args.num_video_frames]

        print(f"Processing segment {i+1}/{len(segments)}...")
        filename_start = os.path.basename(file_names[0])
        start_time = timestamp[0]
        filename_end = os.path.basename(file_names[-1])
        end_time = timestamp[-1]

        entity = {
            'id': file_names[0],
            'position': position.mean(axis=0),
            'theta': 3.14,
            'time': timestamp.mean(),
            'start_time': start_time,
            'end_time': end_time,
            'image_file_path': images_filenames[0],
            'image_filenames': images_filenames,
            'caption': "unknown",
            'file_start': filename_start,
            'file_end': filename_end,
            'vlm_embedding': None
        }

        outputs.append(entity)

    output_path = os.path.join(convert_path, f'convert_{args.seconds_per_caption}_secs.json')
    with open(output_path, 'w') as f:
        json.dump(outputs, f, cls=NumpyEncoder, indent=2)
    
    print(f"\nProcessing completed!")
    print(f"Processed {len(segments)} segments")
    print(f"Output saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build frames JSON and QA JSON for vlm on navqa dataset")
    parser.add_argument("--seq_id", type=int, default=0)
    parser.add_argument("--data_path", type=str, default=os.path.join(PROJECT_ROOT, "coda_data"))
    parser.add_argument("--out_path", type=str, default=os.path.join(PROJECT_ROOT, "data_vlm", "convert", "{seq_id}"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--seconds_per_caption", type=int, default=3)
    parser.add_argument("--num_video_frames", type=int, default=6)
    args = parser.parse_args()
    run_video_in_segs(args)