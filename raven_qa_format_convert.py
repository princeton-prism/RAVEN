#!/usr/bin/env python3
"""
Build per-video JSON files:
  1) Frames listing in your target schema.
  2) QA listing derived from questions.json in the requested format.

Example frame item:
{
    "id": "./data/image_reverse_search_dataset/indoor/_BHE9vqx-qk/frames/frame_001.jpg",
    "position": ["unknown"],
    "time": 1,
    "image_file_path": "./data/image_reverse_search_dataset/indoor/_BHE9vqx-qk/frames/frame_001.jpg",
    "caption": "unknown",
    "vlm_embedding": null
}

Example QA output (per video):
{
  "data": [
    { "id": "1", "question": "Where is this object: <desc> ?", "answer": 221003116 },
    ...
  ]
}

Notes on QA:
- We read "questions (ques, answer_in_filename, category, query_img)" from questions.json.
  (Also supports "questions (ques, answer_in_position)" for backward compatibility)
- The answer_in_filename/answer_in_position field can be in multiple formats:
  * Format v1: A filename (string) - mapped to timestamp via "timestamp_to_filename"
  * Format v2: A position coordinate [x, y, z] (list) - matched to timestamp via position in "timestamp_to_position"
  * Format v3: Combined format [[x, y, z], "filename"] - tries filename matching first, then position matching
- For each question, we map its answer (filename or position) to a timestamp.
- If multiple matches are found, we pick the earliest timestamp (microseconds). If none map, answer is null.
- The schema requested a single numeric 'answer', so we return one integer (microseconds) or null.
- Position matching uses (x, y) coordinates with configurable tolerance for approximate matching.
- In Format v3, filename matching is preferred over position matching for better accuracy.
"""

from __future__ import annotations
import os
from pathlib import Path
import re
import json
import argparse
from typing import Dict, List, Optional, Tuple, Any

def parse_hms_to_microseconds(hms: str) -> Optional[int]:
    """Parse HH:MM:SS, MM:SS, or SS to microseconds. Returns None if invalid."""
    parts = [float(p) for p in hms.split(":")]
    if len(parts) == 1:
        h, m, s = 0, 0, parts[0]
    elif len(parts) == 2:
        h, m, s = 0, parts[0], parts[1]
    elif len(parts) == 3:
        h, m, s = parts
    else:
        raise ValueError("Too many parts")
    if min(h, m, s) < 0:
        raise ValueError("Negative time component")
    return int((h*3600 + m*60 + s) * 1e6)

def natural_key(s: str):
    """Human-friendly sort key: splits numbers to sort numerically (frame_2 < frame_10)."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

def relpath_under(root: str, path: str) -> str:
    """Return POSIX-style relative path from root to path."""
    rel = os.path.relpath(path, root)
    return rel.replace(os.sep, "/")

def build_position_to_time_map(qjson_path: str) -> Tuple[Dict[str, Optional[int]], Dict[tuple, Optional[int]], Dict[str, List[float]]]:
    """
    From questions.json, build three mappings:
    1. filename -> time_in_microseconds (for backward compatibility)
    2. position_tuple -> time_in_microseconds (for position-based matching)
    3. filename -> position [x, y, z] (for frames.json position field)
    """
    filename_to_time = {}
    position_to_time = {}
    filename_to_position = {}
    
    if not os.path.exists(qjson_path):
        return filename_to_time, position_to_time, filename_to_position
    
    try:
        with open(qjson_path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        
        # Support both old and new field names
        ts2val = obj.get("timestamp_to_position", obj.get("timestamp_to_filename", {}))
        fn2pos = obj.get("filename_to_position", {})
        for ts, val in ts2val.items():
            secs = parse_hms_to_microseconds(ts)
            if secs is None:
                continue
            
            # Check if value is a position (list of numbers) or filename (string)
            if isinstance(val, list) and len(val) >= 2:
                # It's a position coordinate [x, y, z] or [x, y]
                # Use tuple of first 2 elements (x, y) as key for matching
                pos_key = (float(val[0]), float(val[1]))
                position_to_time[pos_key] = secs
            elif isinstance(val, str):
                # It's a filename (backward compatibility)
                bname = os.path.basename(val)
                filename_to_time[bname] = secs
                filename_to_time[val.replace("\\", "/")] = secs
        
        # Extract filename -> position mapping from questions
        # Support both old and new field names
        q_list = obj.get("questions (ques, answer_in_position)", None)
        if q_list is None:
            q_list = obj.get("questions (ques, answer_in_filename)", None)
        if q_list is None:
            q_list = obj.get("questions", [])
        
        for entry in q_list:
            if len(entry) < 2:
                continue
            ans = entry[1]
            
            # Helper function to process a single combined format answer
            def process_single_answer(ans_item):
                """Process a single [[x, y, z], "filename"] format answer."""
                if isinstance(ans_item, list) and len(ans_item) == 2:
                    first_elem = ans_item[0]
                    second_elem = ans_item[1]
                    
                    if (isinstance(first_elem, list) and len(first_elem) >= 3 and 
                        isinstance(first_elem[0], (int, float)) and 
                        isinstance(second_elem, str)):
                        # Combined format: [[x, y, z], "filename"]
                        position = first_elem
                        filename = second_elem
                        bname = os.path.basename(filename)
                        filename_to_position[bname] = position
                        filename_to_position[filename] = position
            
            # Check if this is a list of multiple combined format answers
            # Format: [[[x1, y1, z1], "file1"], [[x2, y2, z2], "file2"], ...]
            if isinstance(ans, list) and len(ans) > 0:
                # Check if first element looks like a combined format answer
                first_item = ans[0]
                if (isinstance(first_item, list) and len(first_item) == 2 and
                    isinstance(first_item[0], list) and len(first_item[0]) >= 3 and
                    isinstance(first_item[0][0], (int, float)) and
                    isinstance(first_item[1], str)):
                    # This is a list of multiple combined format answers
                    for ans_item in ans:
                        process_single_answer(ans_item)
                # Handle single combined format: [[x, y, z], "filename"]
                elif len(ans) == 2:
                    process_single_answer(ans)
                # Handle position-only format: [x, y, z]
                elif len(ans) >= 3 and isinstance(ans[0], (int, float)):
                    # This is a position, but we don't have filename here
                    # Skip for now, as we need filename to create the mapping
                    pass
            # Handle filename-only format: "filename"
            elif isinstance(ans, str):
                # Filename only, no position info
                pass
        
        for fname, pos in fn2pos.items():
            # if fname in filename_to_position:
            #     print(filename_to_position[fname], pos)
            filename_to_position[fname] = pos

        return filename_to_time, position_to_time, filename_to_position
    except Exception as e:
        print(f"[WARN] Failed reading {qjson_path}: {e}", flush=True)
        return filename_to_time, position_to_time, filename_to_position

def collect_video_dirs(data_root: str, splits: List[str], auto_discover: bool = False) -> List[str]:
    """
    Collect video directories. Supports multiple structures:
    1. With splits: data_root/split/video/frames/
    2. Without splits: data_root/video/frames/ (with questions.json)
    3. Direct structure: data_root/frames/ (with questions.json in data_root)
    
    Args:
        data_root: Root directory to search
        splits: List of split names to search (e.g., ["darpa_sim_v3", "darpa_sim_v4"])
        auto_discover: If True, automatically discover all subdirectories as splits
    """
    found = []
    
    # Auto-discover splits if requested
    if auto_discover:
        if os.path.isdir(data_root):
            discovered_splits = []
            for item in os.listdir(data_root):
                item_path = os.path.join(data_root, item)
                if os.path.isdir(item_path):
                    # Check if this directory contains video subdirectories (has frames or questions.json)
                    has_videos = False
                    for subitem in os.listdir(item_path):
                        subitem_path = os.path.join(item_path, subitem)
                        if os.path.isdir(subitem_path):
                            frames_dir = os.path.join(subitem_path, "frames")
                            qjson_path = os.path.join(subitem_path, "questions.json")
                            if os.path.isdir(frames_dir) or os.path.exists(qjson_path):
                                has_videos = True
                                break
                    if has_videos:
                        discovered_splits.append(item)
            if discovered_splits:
                splits = discovered_splits
                print(f"[INFO] Auto-discovered splits: {splits}")
    
    # First, try structure with splits
    for split in splits:
        split_dir = os.path.join(data_root, split)
        if not os.path.isdir(split_dir):
            continue
        for vid in os.listdir(split_dir):
            vdir = os.path.join(split_dir, vid)
            if os.path.isdir(vdir):
                frames_dir = os.path.join(vdir, "frames")
                if os.path.isdir(frames_dir):
                    found.append(vdir)
    
    # If no videos found with splits, try direct structure (no split level)
    if not found:
        if os.path.isdir(data_root):
            for vid in os.listdir(data_root):
                vdir = os.path.join(data_root, vid)
                if os.path.isdir(vdir):
                    # Check if it has questions.json (indicates it's a video directory)
                    qjson_path = os.path.join(vdir, "questions.json")
                    if os.path.exists(qjson_path):
                        found.append(vdir)
            
            # Also check if data_root itself has questions.json and frames
            qjson_path = os.path.join(data_root, "questions.json")
            frames_dir = os.path.join(data_root, "frames")
            if os.path.exists(qjson_path) and os.path.isdir(frames_dir):
                found.append(data_root)
    
    return sorted(found, key=natural_key)

def pick_answer_time_us(answer_value: Any, fname_to_time: Dict[str, Optional[int]], 
                        pos_to_time: Dict[tuple, Optional[int]], 
                        position_tolerance: float = 0.01) -> Tuple[List[int], List[str], List[List[float]]]:
    """
    Given an answer (filename, position, or list), return matching time(s) in microseconds.
    Supports multiple formats:
    1. Filename (str): "frame_001.jpg"
    2. Position (list): [x, y, z] or [x, y]
    3. Combined format (list): [[x, y, z], "filename"] - tries filename first, then position
    4. List of combined formats: [[[x1, y1, z1], "file1"], [[x2, y2, z2], "file2"]] - multiple answers
    5. List of filenames: ["frame_001.jpg", "frame_002.jpg"]
    
    Args:
        answer_value: Can be a filename (str), position [x, y, z] (list), 
                     combined format [[x, y, z], "filename"], or list of answers
        fname_to_time: Mapping from filename to time
        pos_to_time: Mapping from (x, y) tuple to time
        position_tolerance: Tolerance for position matching (default: 0.01)
    
    Returns:
        Tuple of (list of times in microseconds, list of filenames, list of positions)
    """
    if answer_value is None:
        return [], [], []
    
    # Helper function to process a single combined format answer: [[x, y, z], "filename"]
    def process_single_combined_answer(ans_item):
        """Process a single [[x, y, z], "filename"] format answer."""
        if not isinstance(ans_item, list) or len(ans_item) != 2:
            return [], [], []
        
        first_elem = ans_item[0]
        second_elem = ans_item[1]
        
        # Check if first element is a position (list of numbers) and second is a filename (string)
        if (isinstance(first_elem, list) and len(first_elem) >= 2 and 
            isinstance(first_elem[0], (int, float)) and 
            isinstance(second_elem, str)):
            # Combined format: [[x, y, z], "filename"]
            filename = second_elem
            position = first_elem
            
            secs_list = []
            filename_list = []
            position_list = []
            
            # Try filename matching first
            bname = os.path.basename(filename)
            if bname in fname_to_time:
                secs = fname_to_time[bname]
                secs_list.append(secs)
                filename_list.append(filename)
                position_list.append(list(position))  # Store position as list
            elif filename.replace("\\", "/") in fname_to_time:
                secs = fname_to_time[filename.replace("\\", "/")]
                secs_list.append(secs)
                filename_list.append(filename)
                position_list.append(list(position))  # Store position as list
            
            # If filename didn't match, try position matching
            if not secs_list:
                try:
                    pos_key = (float(position[0]), float(position[1]))
                    
                    # Try exact match first
                    if pos_key in pos_to_time:
                        secs = pos_to_time[pos_key]
                        secs_list.append(secs)
                        filename_list.append("")  # No filename match
                        position_list.append(list(position))  # Store position as list
                    else:
                        # Try approximate match with tolerance
                        for (px, py), time_val in pos_to_time.items():
                            if abs(px - pos_key[0]) < position_tolerance and abs(py - pos_key[1]) < position_tolerance:
                                secs_list.append(time_val)
                                filename_list.append("")  # No filename match
                                position_list.append(list(position))  # Store position as list
                                break
                except (ValueError, TypeError, IndexError):
                    pass
            
            return secs_list, filename_list, position_list
        return [], [], []
    
    # Check if this is a list of multiple combined format answers
    # Format: [[[x1, y1, z1], "file1"], [[x2, y2, z2], "file2"], ...]
    if isinstance(answer_value, list) and len(answer_value) > 0:
        # Check if first element looks like a combined format answer
        first_item = answer_value[0]
        if (isinstance(first_item, list) and len(first_item) == 2 and
            isinstance(first_item[0], list) and len(first_item[0]) >= 2 and
            isinstance(first_item[0][0], (int, float)) and
            isinstance(first_item[1], str)):
            # This is a list of multiple combined format answers
            all_secs = []
            all_filenames = []
            all_positions = []
            for ans_item in answer_value:
                secs_list, filename_list, position_list = process_single_combined_answer(ans_item)
                all_secs.extend(secs_list)
                all_filenames.extend(filename_list)
                all_positions.extend(position_list)
            return all_secs, all_filenames, all_positions
        
        # Check if it's a single combined format: [[x, y, z], "filename"]
        if len(answer_value) == 2:
            first_elem = answer_value[0]
            second_elem = answer_value[1]
            if (isinstance(first_elem, list) and len(first_elem) >= 2 and 
                isinstance(first_elem[0], (int, float)) and 
                isinstance(second_elem, str)):
                # Single combined format
                return process_single_combined_answer(answer_value)
    
    # Handle other formats (backward compatibility)
    candidates = []
    if isinstance(answer_value, str):
        candidates = [answer_value]
    elif isinstance(answer_value, list):
        # Check if it's a position (list of numbers) or list of filenames
        if len(answer_value) > 0 and isinstance(answer_value[0], (int, float)):
            # It's a position [x, y, z] or [x, y]
            candidates = [answer_value]
        else:
            # It's a list of filenames or other format
            candidates = answer_value
    else:
        return [], [], []

    secs_list = []
    filename_list = []
    position_list = []
    
    for candidate in candidates:
        secs = None
        filename = None
        position = None
        
        if isinstance(candidate, list) and len(candidate) >= 2:
            # It's a position coordinate
            try:
                pos_key = (float(candidate[0]), float(candidate[1]))
                position = list(candidate)  # Store full position [x, y, z]
                
                # Try exact match first
                if pos_key in pos_to_time:
                    secs = pos_to_time[pos_key]
                else:
                    # Try approximate match with tolerance
                    for (px, py), time_val in pos_to_time.items():
                        if abs(px - pos_key[0]) < position_tolerance and abs(py - pos_key[1]) < position_tolerance:
                            secs = time_val
                            break
            except (ValueError, TypeError, IndexError):
                pass
        elif isinstance(candidate, str):
            # It's a filename
            filename = candidate
            bname = os.path.basename(candidate)
            if bname in fname_to_time:
                secs = fname_to_time[bname]
            elif candidate.replace("\\", "/") in fname_to_time:
                secs = fname_to_time[candidate.replace("\\", "/")]
        
        if secs is not None:
            secs_list.append(secs)
            filename_list.append(filename if filename else "")
            position_list.append(position if position else [])
    
    return secs_list, filename_list, position_list

def load_questions(qjson_path: str) -> List[Tuple[str, Any, Optional[str], Optional[str]]]:
    """
    Return list of tuples: (query_text, answer_value, category, query_img_filename)
    answer_value can be in multiple formats:
    - A filename (str): "frame_001.jpg"
    - A position coordinate [x, y, z] (list): [9.2, 0.6, 0.44]
    - Combined format [[x, y, z], "filename"]: [[9.2, 0.6, 0.44], "frame_001.jpg"]
    If questions.json is missing or malformed, return empty list.
    """
    if not os.path.exists(qjson_path):
        return []
    try:
        with open(qjson_path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        
        # Support both old and new field names
        q_list = obj.get("questions (ques, answer_in_position)", None)
        if q_list is None:
            q_list = obj.get("questions (ques, answer_in_filename)", None)
        if q_list is None:
            q_list = obj.get("questions", [])
        
        out = []
        for entry in q_list:
            # Safely unpack: allow 2..4 elements
            query = entry[0] if len(entry) > 0 else ""
            ans = entry[1] if len(entry) > 1 else None
            cat = entry[2] if len(entry) > 2 else None
            qimg = entry[3] if len(entry) > 3 else None
            out.append((query, ans, cat, qimg))
        return out
    except Exception as e:
        print(f"[WARN] Failed parsing questions from {qjson_path}: {e}", flush=True)
        return []

def main():
    ap = argparse.ArgumentParser(description="Build per-video frames JSON and QA JSON for reverse search benchmark.")
    ap.add_argument("--data_root", required=True, help="Dataset root (e.g., ./data/darpa_sim_v2).")
    ap.add_argument("--out_dir", required=True, help="Where to write per-video JSON files.")
    ap.add_argument("--splits", nargs="+", default=["darpa_sim_v3"], help="Dataset splits to include (default: darpa_sim_v3). Use --no-auto-discover to disable auto-discovery.")
    ap.add_argument("--no-auto-discover", action="store_true", help="Disable automatic discovery of subdirectories as splits. By default, auto-discovery is enabled.")
    ap.add_argument("--frames_subdir", default="frames", help="Subdirectory containing frames (default: frames).")
    ap.add_argument("--frames_json_name", default="{split}_{video_id}_frames.json", help="Output filename for frames JSON.")
    ap.add_argument("--qa_json_name", default="{split}_{video_id}_qa.json", help="Output filename for QA JSON derived from questions.json.")
    ap.add_argument("--fallback_index_time", action="store_true", help="If no time found for a frame, use 1-based index as 'time'.")
    ap.add_argument("--position_tolerance", type=float, default=0.01, help="Tolerance for position matching (default: 0.01)")
    ap.add_argument("--caption_path", type=str, default=None, help="The path of caption output e.g. data/output/pic_caption")
    args = ap.parse_args()

    data_root = os.path.abspath(args.data_root)
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # Default to auto-discover (unless --no-auto-discover is specified)
    auto_discover = not getattr(args, 'no_auto_discover', False)
    video_dirs = collect_video_dirs(data_root, args.splits, auto_discover=auto_discover)
    if not video_dirs:
        print("[INFO] No video directories with frames found. Check --data-root and --splits.")
        return

    total_items = 0
    outputs = []

    for vdir in video_dirs:
        # Determine split: use parent directory's basename as split name
        # This works for both cases:
        # 1. With split level: data_root/split/video/ -> split = "split"
        # 2. Without split level: data_root/video/ -> split = basename(data_root)
        parent_dir = os.path.dirname(vdir)
        if parent_dir == data_root:
            # No split level, use data_root's basename as split name
            # e.g., data_root = /path/to/data/darpa_sim -> split = "darpa_sim"
            split = os.path.basename(data_root)
        else:
            # Has split level, use parent directory's basename
            split = os.path.basename(parent_dir)  # indoor/outdoor/etc
        
        vid = os.path.basename(vdir) if vdir != data_root else os.path.basename(data_root)
        frames_dir = os.path.join(vdir, args.frames_subdir)
        qjson_path = os.path.join(vdir, "questions.json")

        # Build position-to-time map (needed for both frames and QA)
        fname_to_time, pos_to_time, fname_to_position = build_position_to_time_map(qjson_path)
        
        # Check if frames directory exists
        if not os.path.isdir(frames_dir):
            print(f"[WARN] {vid}: frames directory '{frames_dir}' not found, skipping frames JSON generation")
            frame_files = []
        else:
            # --- Frames JSON ---
            frame_files = [f for f in os.listdir(frames_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
            frame_files.sort(key=natural_key)

        items = []
        max_t, min_t = 0, float('inf')
        

        # load captions if caption_path is provided
        captions = None
        if args.caption_path is not None:
            caption_json_file = [f for f in (Path(args.caption_path) / os.path.basename(parent_dir) / os.path.basename(vdir)).iterdir()][0]
            with open(caption_json_file, "r", encoding="utf-8") as f:
                captions = json.load(f)


        if frame_files:
            for idx, fname in enumerate(frame_files, 1):
                abs_path = os.path.join(frames_dir, fname)
                rel = relpath_under(data_root, abs_path)  # e.g., indoor/VID/frames/frame_001.jpg
                export_path = (args.data_root.rstrip("/")
                               + ("/" if args.data_root else "")
                               + rel)

                t = None
                if fname in fname_to_time:
                    t = fname_to_time[fname]
                else:
                    rel_posix = rel.replace("\\", "/")
                    if rel_posix in fname_to_time:
                        t = fname_to_time[rel_posix]

                if t is None and args.fallback_index_time:
                    t = idx  # 1-based index as a rough fallback

                max_t = max(max_t, t) if t is not None else max_t
                min_t = min(min_t, t) if t is not None else min_t

                # Get position for this frame
                position = None
                if fname in fname_to_position:
                    position = fname_to_position[fname]
                else:
                    # Try with basename if full path doesn't match
                    bname = os.path.basename(fname)
                    if bname in fname_to_position:
                        position = fname_to_position[bname]
                
                # Use position if found, otherwise use "unknown"
                position_field = position if position is not None else ["unknown"]

                item = {
                    "position": position_field,
                    "time": t,
                    "image_file_path": export_path,
                    "vlm_embedding": None,
                    "caption": "unknown" if captions is None else captions[idx-1]["caption"],
                    "text_embedding": None if captions is None else captions[idx-1]["text_embedding"],
                    "segment_index": None if captions is None else captions[idx-1]["segment_index"],
                    "num_frames": None if captions is None else captions[idx-1]["num_frames"],
                    "sampled_frames": None if captions is None else captions[idx-1]["sampled_frames"]
                }
                
                items.append(item)

            frames_out_name = args.frames_json_name.format(split=split, video_id=vid)
            frames_out_path = os.path.join(out_dir, frames_out_name)
            with open(frames_out_path, "w", encoding="utf-8") as f:
                json.dump(items, f, ensure_ascii=False, indent=2)
        else:
            # No frames found, skip frames JSON generation
            frames_out_path = None
            print(f"[INFO] {vid}: No frames found, skipping frames JSON")

        # --- QA JSON ---
        qa_entries_src = load_questions(qjson_path)
        qa_data = []
        for i, (query_text, ans_value, _cat, _qimg) in enumerate(qa_entries_src, 1):
            ans_us_list, answer_files, answer_positions = pick_answer_time_us(ans_value, fname_to_time, pos_to_time, args.position_tolerance)
            
            # Extract all positions from answer_value (even if no timestamp match)
            # This ensures we capture all positions for questions with multiple answers
            all_positions = []
            if isinstance(ans_value, list):
                # Check if it's a list of multiple combined format answers
                if (len(ans_value) > 0 and isinstance(ans_value[0], list) and 
                    len(ans_value[0]) == 2 and isinstance(ans_value[0][0], list) and 
                    len(ans_value[0][0]) >= 3 and isinstance(ans_value[0][0][0], (int, float))):
                    # Multiple answers: [[[x1, y1, z1], "file1"], [[x2, y2, z2], "file2"]]
                    for ans_item in ans_value:
                        if isinstance(ans_item, list) and len(ans_item) >= 1:
                            if isinstance(ans_item[0], list) and len(ans_item[0]) >= 3:
                                all_positions.append(list(ans_item[0]))
                elif len(ans_value) == 2 and isinstance(ans_value[0], list) and len(ans_value[0]) >= 3:
                    # Single combined format: [[x, y, z], "filename"]
                    all_positions.append(list(ans_value[0]))
                elif len(ans_value) >= 3 and isinstance(ans_value[0], (int, float)):
                    # Position format: [x, y, z]
                    all_positions.append(list(ans_value))
            
            # Use positions from pick_answer_time_us if available, otherwise use extracted positions
            if answer_positions:
                # Filter out empty positions
                answer_positions_filtered = [pos for pos in answer_positions if pos]
            else:
                answer_positions_filtered = all_positions
            
            # Filter out empty filenames
            answer_files_filtered = [f for f in answer_files if f]
            
            # Use original question directly - model will provide position based on prompt instructions
            qa_item = {
                "id": str(i),
                "question": query_text,  # Use original question directly
                "answer_ts": ans_us_list if ans_us_list else [],  # list of timestamps in microseconds
                "answer_pos": answer_positions_filtered if answer_positions_filtered else [],  # list of positions [x, y, z]
                "answer_file": answer_files_filtered if answer_files_filtered else [],  # list of filenames
                "category": _cat if _cat is not None else "unknown",
                "query_img": _qimg if _qimg is not None else "unknown",
            }
            qa_data.append(qa_item)

        # Get time range from mappings (more reliable than iterating frames)
        # This ensures we get the actual min/max from timestamp_to_filename/timestamp_to_position
        all_times = []
        if fname_to_time:
            all_times.extend([t for t in fname_to_time.values() if t is not None])
        if pos_to_time:
            all_times.extend([t for t in pos_to_time.values() if t is not None])
        
        if all_times:
            # Use the actual min/max from the timestamp mappings
            actual_min_t = min(all_times)
            actual_max_t = max(all_times)
            # Only update if we found valid times from mappings
            if min_t == float('inf') or actual_min_t < min_t:
                min_t = actual_min_t
            if actual_max_t > max_t:
                max_t = actual_max_t
        
        # Fallback: if still no valid times, use 0
        if min_t == float('inf'):
            min_t = 0
        
        qa_obj = {"start_time": min_t, 
                  "end_time": max_t, 
                  "frame_num": len(items), 
                  "data": qa_data}
        qa_out_name = args.qa_json_name.format(split=split, video_id=vid)
        qa_out_path = os.path.join(out_dir, qa_out_name)
        with open(qa_out_path, "w", encoding="utf-8") as f:
            json.dump(qa_obj, f, ensure_ascii=False, indent=2)

        outputs.append((vdir, frames_out_path, qa_out_path, len(items), len(qa_data)))
        total_items += len(items)
        if frames_out_path:
            print(f"[OK] {vid} -> frames:{frames_out_path} ({len(items)}), qa:{qa_out_path} ({len(qa_data)})")
        else:
            print(f"[OK] {vid} -> qa:{qa_out_path} ({len(qa_data)}) [no frames]")

    print(f"\n[SUMMARY] Videos processed: {len(outputs)} | Total frames: {total_items}")
    for vdir, frames_path, qa_path, n_frames, n_q in outputs:
        if frames_path:
            print(f"  - {os.path.basename(vdir)}: {n_frames} frames -> {frames_path}; {n_q} Qs -> {qa_path}")
        else:
            print(f"  - {os.path.basename(vdir)}: {n_q} Qs -> {qa_path} [no frames]")

if __name__ == "__main__":
    main()
