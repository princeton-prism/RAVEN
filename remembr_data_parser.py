import sys
import termios
import tty
import os
import argparse
import json
import select
from os.path import join
import glob
import re

import numpy as np
from scipy.spatial.transform import Rotation as R 

# import rospy
import cv2

import pickle as pkl
from tqdm import tqdm 
from typing import List

# For imports
import sys

parser = argparse.ArgumentParser(description="video captioner (to build memory)")
parser.add_argument("-i", "--input_path", type=str, default="./data/RealEstate10K/videos/0000.mp4", 
                    help="Path to the input MP4 video file")
parser.add_argument("-o", "--output_dir", type=str, default="./data/RealEstate10K/pickles/0000", 
                    help="Directory to save the pickle files")
parser.add_argument("-s", "--sequence", type=str, default="0", 
                    help="Sequence number (Default 0)")
parser.add_argument("-f", "--start_frame", type=str, default="0",
                    help="Frame to start at (Default 0)")
parser.add_argument("-c", "--color_type", type=str, default="classId", 
                    help="Color map to use for coloring boxes Options: [isOccluded, classId] (Default classId)")
parser.add_argument("-l", "--log", type=str, default="",
                    help="Logs point cloud and bbox annotations to file for external usage")
parser.add_argument("-n", "--namespace", type=str, default="cda",
                    help="Select a namespace to use for published topics")
parser.add_argument("--pic_parser", action='store_true', help="If true, parse the image instead of the video")
parser.add_argument("--debug", action='store_true', help="If true, prints out debug info")


def extract_images_to_pickle(input_path: str, output_dir: str):
    """
    convert images in `input_path` to pickle files, saved in `output_dir`
    
    Args:
        input_path (str): input directory path (e.g. ./data/downloads/)
        output_dir (str): output directory path
    """
    print(f"input_path: {input_path}")
    print(f"output_dir: {output_dir}")
    # import ipdb; ipdb.set_trace()
    # create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    total_processed = 0
    image_files = glob.glob(os.path.join(input_path, "*.jpg"))
    if not image_files:
        print(f"no image files found")
        raise ValueError(f"no image files found in {input_path}")
    image_files.sort()
    print(f"found {len(image_files)} image files")
    for image_file in tqdm(image_files, desc="processing image files"):
        try:
            # extract number from filename as timestamp
            filename = os.path.basename(image_file)
            
            # use regex to extract number
            numbers = re.findall(r'\d+', filename)
            if not numbers:
                print(f"warning: cannot extract number from filename {filename}, skip")
                continue
            
            # use first number as timestamp
            timestamp_str = numbers[0]
            
            # read image
            image = cv2.imread(image_file)
            if image is None:
                print(f"error: cannot read image {image_file}")
                continue
            
            # create output dictionary
            out_dict = {
                'cam0': image,
                'position': [0.0, 0.0, 0.0],  # default position
                'timestamp': timestamp_str
            }
            
            # create output path - use original filename (remove extension) + .pkl
            base_filename = os.path.splitext(filename)[0]
            output_filename = f"{base_filename}.pkl"
            output_path = os.path.join(output_dir, output_filename)
            
            # save as pickle file
            with open(output_path, 'wb') as handle:
                pkl.dump(out_dict, handle, protocol=pkl.HIGHEST_PROTOCOL)
            
            total_processed += 1
            
        except Exception as e:
            print(f"error processing image {image_file}: {e}")
            continue
    
    print(f"\ncompleted! processed {total_processed} image files")
    print(f"pickle files saved to: {output_dir}")


def extract_frames_to_pickle(video_path: str, output_dir: str, position: List[str]):
    """
    Extract frames from MP4 video and save each frame as a pickle file.
    
    Args:
        video_path (str): Path to the input MP4 video file
        output_dir (str): Directory to save the pickle files
        position (List[str]): List of position strings, length equals frame number
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Error: Could not open video in {video_path}")
    
    # Get video properties
    # import ipdb; ipdb.set_trace()
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration_seconds = total_frames / fps
    video_duration_microseconds = int(video_duration_seconds * 1_000_000)
    
    # Calculate the number of digits needed for timestamp formatting
    timestamp_digits = len(str(video_duration_microseconds))
    
    print(f"Video info:")
    print(f"  - FPS: {fps}")
    print(f"  - Total frames: {total_frames}")
    print(f"  - Duration: {video_duration_seconds:.2f} seconds")
    print(f"  - Duration: {video_duration_microseconds} microseconds")
    print(f"  - Timestamp digits: {timestamp_digits}")
    print(f"  - Position list length: {len(position)}")
    
    # Verify position list length matches frame count
    if len(position) != total_frames:
        print(f"Warning: Position list length ({len(position)}) doesn't match frame count ({total_frames})")
        # Adjust to minimum of both
        frame_count = min(len(position), total_frames)
    else:
        frame_count = total_frames
    
    frame_idx = 0
    processed_frames = 0
    
    print(f"\nProcessing {frame_count} frames...")
    
    while frame_idx < frame_count:
        # Read frame
        ret, frame = cap.read()
        
        if not ret:
            print(f"Could not read frame {frame_idx}")
            break
        
        # Calculate timestamp in microseconds for this frame
        timestamp_microseconds = int((frame_idx / fps) * 1_000_000)
        
        # Format timestamp with leading zeros
        timestamp_str = str(timestamp_microseconds).zfill(timestamp_digits)
        
        # Create output dictionary
        out_dict = {}
        out_dict['cam0'] = frame
        out_dict['position'] = position[frame_idx]
        out_dict['timestamp'] = timestamp_str
        
        # Create output path
        output_path = os.path.join(output_dir, f'{timestamp_str}.pkl')
        
        # Save to pickle file
        try:
            with open(output_path, 'wb') as handle:
                pkl.dump(out_dict, handle, protocol=pkl.HIGHEST_PROTOCOL)
            
            processed_frames += 1
            
            # Progress update every 100 frames
            if processed_frames % 100 == 0:
                print(f"Processed {processed_frames}/{frame_count} frames")
                
        except Exception as e:
            print(f"Error saving frame {frame_idx}: {e}")
        
        frame_idx += 1
    
    # Clean up
    cap.release()
    
    print(f"\nCompleted! Processed {processed_frames} frames.")
    print(f"Pickle files saved to: {output_dir}")

def recursive_extract_images_to_pickle(input_path: str, output_dir: str):
    """
    Recursively convert images in `input_path` to pickle files, saved in `output_dir`
    """
    # import ipdb; ipdb.set_trace()
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
    input_path = os.path.abspath(input_path)
    output_dir = os.path.abspath(output_dir)
    found_any_images = False
    processed_dirs = 0
    for root, dirs, files in os.walk(input_path):
        # Only consider .jpg files (case-insensitive), to match extract_images_to_pickle behavior
        if os.path.basename(root).lower() != "frames":
            continue
        has_jpg = any(f.lower().endswith(".jpg") for f in files)
        if not has_jpg:
            continue

        found_any_images = True

        # Compute the relative path from input_path to current root
        rel_root = os.path.relpath(root, input_path)
        # Map to the corresponding output subdirectory
        out_subdir = os.path.join(output_dir, rel_root)

        print(f"\n[recursive] Processing directory:\n  in : {root}\n  out: {out_subdir}")
        try:
            extract_images_to_pickle(root, out_subdir)
            processed_dirs += 1
        except Exception as e:
            print(f"[recursive] Error processing '{root}': {e}")
    if not found_any_images:
        raise ValueError(f"No .jpg images found anywhere under '{input_path}'")
    print(f"\n[recursive] Completed! Processed {processed_dirs} directories")
    print(f"[recursive] Pickle files saved to: {output_dir}")


def main(args):
    if args.pic_parser:
        # image parser mode
        print("using image parser mode")
        recursive_extract_images_to_pickle(args.input_path, args.output_dir)
    else:
        # video parser mode (original functionality)
        print("using video parser mode")
        
        # Example position list - in real usage, this should match your frame count
        # For demonstration, creating a dummy list
        position = ["pos_0", "pos_1", "pos_2", "pos_N"]  # Replace with your actual position data
        
        # Note: You need to ensure the position list has the same length as video frames
        # You can get frame count first:
        cap = cv2.VideoCapture(args.input_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        # Create position list with correct length (example)
        position = [f"position_{i}" for i in range(total_frames)]
        
        # Extract frames
        extract_frames_to_pickle(args.input_path, args.output_dir, position)


if __name__ == "__main__":
    args = parser.parse_args()
    # Uncomment the line below to run the example
    main(args)