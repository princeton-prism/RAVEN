
#!/usr/bin/env python3
"""
Evaluate prediction accuracy for time-based and position-based Q/A JSON files.

- Prediction files live under a "pred_dir". Each file has a "responses" list where each item
  contains fields including:
      - "id": question id (string or int)
      - "time": predicted HH:MM:SS string (e.g. "19:01:34")
      - "position": predicted position [x, y, z] (e.g. [9.918, 0.399, 0.44])

- Ground-truth files live under a "gt_dir". Each GT filename should contain the substring
  extracted from the prediction filename: the token between the first "__" and the next "_frames".
  Example:
      pred filename: remembr+gpt-4o+oc+ViT-B-32+...__indoor_2JYMWPElMHk_frames__._0.json
      token: "indoor_2JYMWPElMHk"
      GT file to use: any file in gt_dir whose name contains "indoor_2JYMWPElMHk"
                      (typically "indoor_2JYMWPElMHk_qa.json")

- GT format contains "data": [{"id": "...", "answer_ts": 94000000, "answer_pos": [x, y, z], ...}, ...]
  where "answer_ts" is a relative timestamp in microseconds and "answer_pos" is position [x, y, z].
  Convert to HH:MM:SS using (as requested) localtime:
      t = answer_ts / 1e6
      hhmmss = time.strftime("%H:%M:%S", time.localtime(t))

- Compare prediction time string vs. GT time string for equality.
- Compare prediction position vs. GT position using Euclidean distance.
  Compute per-file accuracy and overall accuracy for both time and position.

Outputs:
- A CSV summary at "time_eval_summary.csv" (or custom name)
- Prints a per-file table and overall accuracy for both time and position.
"""

import argparse
import json
import re, os
import time
import math
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import csv
from collections import defaultdict
IMG_TAG_RE = re.compile(r"\[IMG\](.*?)\[/IMG\]")


def extract_token_from_pred_filename(fname: str) -> Optional[str]:
    """
    Extract the token between the first '__' and the next '_frames' in the filename.
    Returns None if not found.
    """
    # We only look at the stem to make matching simpler
    m = re.search(r'(.*?)_frames__', fname)
    if m:
        return m.group(1)
    return None

def find_gt_file(gt_dir: Path, token: str) -> Optional[Path]:
    """
    Find the GT file in gt_dir whose filename contains the given token.
    Prefer files ending with '_qa.json', but fall back to any .json containing the token.
    """
    candidates: List[Path] = sorted([p for p in gt_dir.glob("*.json") if token in p.name])
    if not candidates:
        return None
    # prefer *_qa.json if available
    qa = [p for p in candidates if p.name.endswith("_qa.json")]
    return qa[0] if qa else candidates[0]

def load_gt_map(gt_path: Path) -> Tuple[Dict[str, list], Dict[str, list], Dict[str, list]]:
    """
    Load GT JSON and return:
      - gt_map: dict mapping id (string) -> list of seconds (via localtime)
      - gt_map_file: dict mapping id (string) -> list of filenames
      - gt_map_pos: dict mapping id (string) -> list of positions [x, y, z]
    """
    with gt_path.open("r", encoding="utf-8") as f:
        obj = json.load(f)

    gt_map: Dict[str, list] = {}
    gt_map_file: Dict[str, list] = {}
    gt_map_pos: Dict[str, list] = {}
    data = obj.get("data", [])
    frame_num = int(obj["frame_num"]) if "frame_num" in obj else 1e10
    total_random_acc = []
    for item in data:
        qid = str(item.get("id"))
        ans_us = item.get("answer_ts", None)
        if ans_us is None:
            # Fallback to "answer" for backward compatibility
            ans_us = item.get("answer", None)
        if ans_us is not None:
            if isinstance(ans_us, list):
                total_random_acc.append(len(ans_us) / frame_num)
                for ans in ans_us:
                    seconds = float(ans) / 1e6
                    # Per user spec: use localtime
                    tstruct = time.localtime(seconds)
                    # turn to seconds for easier comparison
                    sec = tstruct.tm_hour * 3600 + tstruct.tm_min * 60 + tstruct.tm_sec
                    gt_map.setdefault(qid, []).append(sec)
            elif isinstance(ans_us, (int, float)):
                seconds = float(ans_us) / 1e6
                tstruct = time.localtime(seconds)
                sec = tstruct.tm_hour * 3600 + tstruct.tm_min * 60 + tstruct.tm_sec
                gt_map.setdefault(qid, []).append(sec)
        
        # Load position
        ans_pos = item.get("answer_pos", None)
        if ans_pos is not None:
            if isinstance(ans_pos, list) and len(ans_pos) > 0:
                # Check if it's a list of positions (nested list)
                if isinstance(ans_pos[0], list):
                    # List of positions: [[x1, y1, z1], [x2, y2, z2], ...]
                    for pos in ans_pos:
                        if isinstance(pos, list) and len(pos) >= 3:
                            gt_map_pos.setdefault(qid, []).append(pos[:3])
                elif len(ans_pos) >= 3:
                    # Single position: [x, y, z]
                    gt_map_pos.setdefault(qid, []).append(ans_pos[:3])
        
        ans_filename = item.get("answer_file", None)
        if ans_filename is not None:
            if isinstance(ans_filename, list):
                for fn in ans_filename:
                    gt_map_file.setdefault(qid, []).append(fn)
            else:
                gt_map_file.setdefault(qid, []).append(ans_filename)
    return gt_map, gt_map_file, gt_map_pos, total_random_acc



def load_pred_map(pred_path: Path) -> Tuple[Dict[str, Optional[str]], Dict[str, Optional[List[float]]], Dict[str, Optional[str]]]:
    """
    Load prediction JSON and return:
      - pred_map: mapping id (string) -> HH:MM:SS string (or None if missing)
      - pred_map_pos: mapping id (string) -> position [x, y, z] (or None if missing)
      - pred_map_file: mapping id (string) -> filename string (e.g., "frame_001.jpg") if time is a filename, else None
    """
    with pred_path.open("r", encoding="utf-8") as f:
        obj = json.load(f)

    # Pattern to match filename format like "frame_001.jpg"
    FILENAME_PATTERN = re.compile(r'^frame_\d+\.jpg$', re.IGNORECASE)

    # 1) existing time parsing
    pred_map: Dict[str, Optional[str]] = {}
    pred_map_pos: Dict[str, Optional[List[float]]] = {}
    pred_map_file: Dict[str, Optional[str]] = {}  # New: for filename-based predictions
    for resp in obj.get("responses", []):
        qid = str(resp.get("id"))
        t = resp.get("time")
        pred_map_file[qid] = None  # Initialize as None
        
        if isinstance(t, str):
            # Check if it's a filename format (e.g., "frame_001.jpg")
            filename_only = os.path.basename(t.strip())
            if FILENAME_PATTERN.match(filename_only):
                pred_map_file[qid] = filename_only
                pred_map[qid] = None  # Don't treat as time string
            elif re.fullmatch(r"\d{2}:\d{2}:\d{2}", t):
                pred_map[qid] = t
            else:
                # attempt to extract HH:MM:SS from text field or from t itself
                # look for pattern in the "time" string
                m = re.search(r"(\d{2}:\d{2}:\d{2})", t)
                pred_map[qid] = m.group(1) if m else None
        else:
            # try the "response"->"time" field if present
            inner = resp.get("response", {})
            tin = inner.get("time")
            if isinstance(tin, str):
                filename_only = os.path.basename(tin.strip())
                if FILENAME_PATTERN.match(filename_only):
                    pred_map_file[qid] = filename_only
                    pred_map[qid] = None
                elif re.fullmatch(r"\d{2}:\d{2}:\d{2}", tin):
                    pred_map[qid] = tin
                else:
                    pred_map[qid] = None
            else:
                pred_map[qid] = None
        
        # Parse position
        pos = resp.get("position")
        if pos is None:
            # Try "response"->"position" field
            inner = resp.get("response", {})
            pos = inner.get("position")
        
        if pos is not None:
            if isinstance(pos, list) and len(pos) >= 3:
                try:
                    pred_map_pos[qid] = [float(pos[0]), float(pos[1]), float(pos[2])]
                except (ValueError, TypeError):
                    pred_map_pos[qid] = None
            elif isinstance(pos, str):
                # Try to parse string like "[9.918, 0.399, 0.44]"
                try:
                    pos_parsed = json.loads(pos)
                    if isinstance(pos_parsed, list) and len(pos_parsed) >= 3:
                        pred_map_pos[qid] = [float(pos_parsed[0]), float(pos_parsed[1]), float(pos_parsed[2])]
                    else:
                        pred_map_pos[qid] = None
                except (json.JSONDecodeError, ValueError, TypeError):
                    pred_map_pos[qid] = None
            else:
                pred_map_pos[qid] = None
        else:
            pred_map_pos[qid] = None

    return pred_map, pred_map_pos, pred_map_file


def load_gt_category_map(gt_path: Path) -> Dict[str, Optional[str]]:
    """
    Load GT JSON and return a dict mapping id (string) -> category (if available).
    Tries common keys: 'category', 'type', 'question_type', 'Question\nCategory'.
    Fallback to 'unknown' if not found.
    """
    with gt_path.open("r", encoding="utf-8") as f:
        obj = json.load(f)

    cat_map: Dict[str, Optional[str]] = {}
    data = obj.get("data", [])
    for item in data:
        qid = str(item.get("id"))
        cat = (
            item.get("category")
            or item.get("type")
            or item.get("question_type")
            or item.get("Question\nCategory")
        )
        if cat is None:
            cat = "unknown"
        cat_map[qid] = str(cat)
    return cat_map

def euclidean_distance(pos1: List[float], pos2: List[float]) -> float:
    """Calculate 2D Euclidean distance between two positions (ignoring z coordinate)."""
    if len(pos1) < 2 or len(pos2) < 2:
        return float('inf')
    return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

def compare_times(pred_map: Dict[str, Optional[str]], gt_map: Dict[str, list], gt_map_file=None, pred_map_file=None) -> Tuple[int, int, List[Tuple[str, Optional[str], Optional[str], bool]]]:
    """
    Compare predicted vs. ground-truth times for matching question ids.
    If pred_map_file is provided and contains a filename for a qid, compare with gt_map_file instead of time.
    Returns: (num_correct, num_total, rows) where rows are (id, pred, gt, is_correct)
    Only questions present in GT are counted toward total.
    """
    num_correct = 0
    num_total = 0
    rows = []

    for qid, gt in gt_map.items():
        # Check if prediction is a filename format
        pred_filename = pred_map_file.get(qid) if pred_map_file else None
        
        if pred_filename and gt_map_file and qid in gt_map_file:
            # Compare filename instead of time
            gt_filenames = gt_map_file[qid]
            ok = pred_filename in gt_filenames
            pred = pred_filename  # For display purposes
        else:
            # Original time-based comparison
            pred = pred_map.get(qid)
            if isinstance(pred, str):
                ph, pm, ps = map(float, pred.split(":"))
                pred_sec = ph * 3600 + pm * 60 + ps
            elif isinstance(pred, (float, int)):
                pred_sec = float(pred)
            else:
                print(f"Unexpected pred format for id={qid}: {pred}")
                pred_sec = -1.0  # invalid
                        
            if isinstance(gt, (float, int)):
                ok = (abs(pred_sec - float(gt)) < 2.)  
            else:
                ok = False
                for g in gt:
                    if abs(pred_sec - g) < 2.:
                        ok = True
                        break
        if gt is not None:
            num_total += 1
            if ok:
                num_correct += 1

        rows.append((qid, pred, gt, ok, gt_map_file.get(qid, []) if gt_map_file else []))
    return num_correct, num_total, rows

def compare_positions(pred_map_pos: Dict[str, Optional[List[float]]], gt_map_pos: Dict[str, list], pos_threshold: float = 0.1) -> Tuple[int, int, List[Tuple[str, Optional[List[float]], Optional[List[float]], bool, float]], float, List[float]]:
    """
    Compare predicted vs. ground-truth positions for matching question ids.
    Returns: (num_correct, num_total, rows, total_error, errors) where rows are (id, pred_pos, gt_pos, is_correct, error_distance)
    Only questions present in GT are counted toward total.
    
    Supports:
    - pred_pos can be a single position [x, y, z] or a list of positions [[x1, y1, z1], [x2, y2, z2], ...]
    - If pred_pos is a list of positions, selects the one with minimum L2 distance to any GT position
    - Multiple GT positions: pred is correct if it matches any GT position (within threshold)
    """
    num_correct = 0
    num_total = 0
    rows = []
    total_error = 0.0
    errors = []
    
    for qid, gt_positions in gt_map_pos.items():
        if not gt_positions or len(gt_positions) == 0:
            # No GT position data, skip
            continue
            
        pred_pos_raw = pred_map_pos.get(qid)
        
        # Normalize pred_pos: handle case where it might be a list of lists
        pred_positions = []
        if pred_pos_raw is None:
            pred_positions = []
        elif isinstance(pred_pos_raw, list) and len(pred_pos_raw) > 0:
            # Check if it's a list of lists (e.g., [[x, y, z], [x2, y2, z2]])
            if isinstance(pred_pos_raw[0], list):
                # Multiple predicted positions
                for pos in pred_pos_raw:
                    if isinstance(pos, list) and len(pos) >= 2:
                        try:
                            # Ensure it's a list of numbers
                            normalized = [float(pos[0]), float(pos[1])]
                            pred_positions.append(normalized)
                        except (ValueError, TypeError, IndexError):
                            continue
            # Check if it's a single position list [x, y, z]
            elif isinstance(pred_pos_raw[0], (int, float)) and len(pred_pos_raw) >= 2:
                try:
                    normalized = [float(pred_pos_raw[0]), float(pred_pos_raw[1])]
                    pred_positions.append(normalized)
                except (ValueError, TypeError, IndexError):
                    pass
        
        if not pred_positions:
            # No valid prediction
            print(f"Missing/invalid position prediction for id={qid}: {pred_pos_raw}")
            num_total += 1
            rows.append((qid, pred_pos_raw, gt_positions[0] if gt_positions else None, False, float('inf')))
            continue
        
        # For each predicted position, find minimum distance to any GT position
        # Then select the pred position that gives the overall minimum distance
        global_min_dist = float('inf')
        best_pred_pos = None
        best_gt_pos = None
        
        for pred_pos in pred_positions:
            # Find minimum distance from this pred_pos to any GT position
            min_dist_for_this_pred = float('inf')
            best_gt_for_this_pred = None
            
            for gt_pos in gt_positions:
                if isinstance(gt_pos, list) and len(gt_pos) >= 2:
                    try:
                        # Normalize GT position to [x, y]
                        gt_pos_normalized = [float(gt_pos[0]), float(gt_pos[1])]
                        dist = euclidean_distance(pred_pos, gt_pos_normalized)
                        if dist < min_dist_for_this_pred:
                            min_dist_for_this_pred = dist
                            best_gt_for_this_pred = gt_pos
                    except (ValueError, TypeError, IndexError):
                        continue
            
            # Update global minimum if this pred_pos is better
            if min_dist_for_this_pred < global_min_dist:
                global_min_dist = min_dist_for_this_pred
                best_pred_pos = pred_pos
                best_gt_pos = best_gt_for_this_pred
        
        # Check if correct: pred matches any GT position (within threshold)
        ok = global_min_dist < pos_threshold
        if ok:
            num_correct += 1
        else:
            print(f"Position error for id={qid}: pred={best_pred_pos}, gt={best_gt_pos}, error={global_min_dist:.3f}m")
            errors.append(global_min_dist)
        
        total_error += global_min_dist
        num_total += 1
        
        rows.append((qid, best_pred_pos, best_gt_pos, ok, global_min_dist))
    
    return num_correct, num_total, rows, total_error, errors

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", required=True, help="Directory with prediction JSON files")
    ap.add_argument("--gt_dir", required=True, help="Directory with ground-truth JSON files")
    ap.add_argument("--out_csv", default="output/time_eval_summary.csv", help="Path to write CSV summary")
    ap.add_argument("--category_out_csv", help="Path to write per-category CSV summary")
    ap.add_argument("--pos_threshold", type=float, default=0.1, help="Position error threshold in meters (default 0.1)")
    ap.add_argument("--splits", nargs="+", default=[], help="Dataset splits to include (default: indoor outdoor).")
    args = ap.parse_args()

    pred_dir = Path(args.pred_dir)
    gt_dir = Path(args.gt_dir)
    out_csv = Path(args.out_csv)
    all_files = sorted([p for p in pred_dir.glob("*.json")])
    if not all_files:
        print(f"No prediction JSONs found in: {pred_dir}")
        return

    overall_correct = 0
    overall_total = 0
    overall_pos_correct = 0
    overall_pos_total = 0
    overall_pos_error = 0.0
    overall_pos_errors = []
    overall_cat_correct: Dict[str, int] = defaultdict(int)
    overall_cat_total: Dict[str, int] = defaultdict(int)
    overall_cat_pos_correct: Dict[str, int] = defaultdict(int)
    overall_cat_pos_total: Dict[str, int] = defaultdict(int)
    per_file_summaries = []  # list of dicts for CSV

    # Header row for CSV
    fieldnames = ["pred_file", "gt_file", "file_correct", "file_total", "file_accuracy", 
                  "file_pos_correct", "file_pos_total", "file_pos_accuracy", "file_pos_avg_error"]

    if not os.path.exists(out_csv.parent):
        os.makedirs(out_csv.parent, exist_ok=True)

    with out_csv.open("w", newline="", encoding="utf-8") as fcsv:
        writer = csv.DictWriter(fcsv, fieldnames=fieldnames)
        writer.writeheader()
        overall_random_acc = []
        for pred_path in all_files:
            if not args.splits or any(split in pred_path.name for split in args.splits):
                pass
            else:
                continue
            token = extract_token_from_pred_filename(pred_path.name)
            if not token:
                print(f"[WARN] Could not extract token from '{pred_path.name}', skipping.")
                continue

            gt_path = find_gt_file(gt_dir, token)
            if not gt_path:
                print(f"[WARN] No GT file found in '{gt_dir}' containing token '{token}' for '{pred_path.name}', skipping.")
                continue

            try:
                gt_map, gt_map_file, gt_map_pos, total_random_acc = load_gt_map(gt_path)
            except Exception as e:
                print(f"[ERROR] Failed to read GT '{gt_path}': {e}")
                continue

            overall_random_acc.extend(total_random_acc)
            
            pred_map, pred_map_pos, pred_map_file = load_pred_map(pred_path)

            # Attempt to load categories for per-category accuracy
            try:
                gt_cat_map = load_gt_category_map(gt_path)
            except Exception:
                gt_cat_map = {}

            # ############################ Notes ################################
            # print(gt_map_file)
            # {'1': ['frame_010.jpg'], '2': ['frame_015.jpg'], '3': ['frame_014.jpg'], '4': ['frame_011.jpg'], '5': ['frame_005.jpg'], '6': ['frame_009.jpg'], '7': ['frame_008.jpg']}
            # ############################ Notes ################################

            file_correct, file_total, rows = compare_times(pred_map, gt_map, gt_map_file, pred_map_file)
            file_acc = (file_correct / file_total) if file_total else 0.0

            # Compare positions
            file_pos_correct, file_pos_total, pos_rows, file_pos_error, file_pos_errors = compare_positions(
                pred_map_pos, gt_map_pos, pos_threshold=args.pos_threshold
            )
            file_pos_acc = (file_pos_correct / file_pos_total) if file_pos_total else 0.0
            file_pos_avg_error = (file_pos_error / file_pos_total) if file_pos_total else float('inf')

            # Print a compact per-file report
            print(f"\n== File: {pred_path.name}")
            print(f"   GT : {gt_path.name}")
            print(f"   Time Correct: {file_correct} / {file_total}  (acc={file_acc:.3f})")
            print(f"   Position Correct: {file_pos_correct} / {file_pos_total}  (acc={file_pos_acc:.3f}, avg_error={file_pos_avg_error:.3f}m)")

            # show a few mismatches for debugging
            mismatches = [(qid, p, g) for (qid, p, g, ok, _) in rows if not ok]
            if mismatches:
                print("   Examples of mismatches:")
                for qid, p, g in mismatches[:]:
                    print(f"     - id={qid}: pred={p} | gt={g}")

            per_file_summaries.append({
                "pred_file": pred_path.name,
                "gt_file": gt_path.name,
                "file_correct": file_correct,
                "file_total": file_total,
                "file_accuracy": f"{file_acc:.6f}",
                "file_pos_correct": file_pos_correct,
                "file_pos_total": file_pos_total,
                "file_pos_accuracy": f"{file_pos_acc:.6f}",
                "file_pos_avg_error": f"{file_pos_avg_error:.6f}",
            })

            writer.writerow(per_file_summaries[-1])

            overall_correct += file_correct
            overall_total += file_total
            overall_pos_correct += file_pos_correct
            overall_pos_total += file_pos_total
            overall_pos_error += file_pos_error
            overall_pos_errors.extend(file_pos_errors)
            
            # Update per-category aggregates (overall)
            for (qid, _, _, ok, _) in rows:
                cat = gt_cat_map.get(qid, "unknown")
                overall_cat_total[cat] += 1
                if ok:
                    overall_cat_correct[cat] += 1
            
            # Update per-category position aggregates
            for (qid, _p, _g, ok, _err) in pos_rows:
                cat = gt_cat_map.get(qid, "unknown")
                overall_cat_pos_total[cat] += 1
                if ok:
                    overall_cat_pos_correct[cat] += 1

    overall_acc = (overall_correct / overall_total) if overall_total else 0.0
    overall_pos_acc = (overall_pos_correct / overall_pos_total) if overall_pos_total else 0.0
    overall_pos_avg_error = (overall_pos_error / overall_pos_total) if overall_pos_total else float('inf')
    overall_pos_median_error = sorted(overall_pos_errors)[len(overall_pos_errors)//2] if overall_pos_errors else float('inf')
    overall_random_acc_value = (sum(overall_random_acc) / len(overall_random_acc)) if overall_random_acc else 0.0
    print(f"\nAverage random baseline accuracy (based on #GT answers / #frames): {overall_random_acc_value:.6f}")
    # Print overall
    print("\n================ OVERALL ================")
    print(f"Time - Total Correct: {overall_correct} / {overall_total}")
    print(f"Time - Overall Accuracy: {overall_acc:.6f}")
    print(f"Position - Total Correct: {overall_pos_correct} / {overall_pos_total} (threshold={args.pos_threshold}m)")
    print(f"Position - Overall Accuracy: {overall_pos_acc:.6f}")
    print(f"Position - Average Error: {overall_pos_avg_error:.3f}m")
    print(f"Position - Median Error: {overall_pos_median_error:.3f}m")

    # Print per-category accuracy
    if overall_cat_total:
        print("\nPer-category time accuracy:")
        for cat in sorted(overall_cat_total.keys()):
            c = overall_cat_correct.get(cat, 0)
            t = overall_cat_total[cat]
            acc = (c / t) if t else 0.0
            print(f"  - {cat}: {c} / {t} (acc={acc:.6f})")
    
    if overall_cat_pos_total:
        print("\nPer-category position accuracy:")
        for cat in sorted(overall_cat_pos_total.keys()):
            c = overall_cat_pos_correct.get(cat, 0)
            t = overall_cat_pos_total[cat]
            acc = (c / t) if t else 0.0
            print(f"  - {cat}: {c} / {t} (acc={acc:.6f})")

    # Write per-category CSV
    cat_out_csv = Path(args.out_csv[:-4] + ".category.csv") if not args.category_out_csv else Path(args.category_out_csv)
    with cat_out_csv.open("w", newline="", encoding="utf-8") as fcsv:
        writer = csv.DictWriter(fcsv, fieldnames=["category", "time_correct", "time_total", "time_accuracy", 
                                                   "pos_correct", "pos_total", "pos_accuracy"])
        writer.writeheader()
        all_cats = set(overall_cat_total.keys()) | set(overall_cat_pos_total.keys())
        for cat in sorted(all_cats):
            c_time = overall_cat_correct.get(cat, 0)
            t_time = overall_cat_total.get(cat, 0)
            acc_time = (c_time / t_time) if t_time else 0.0
            c_pos = overall_cat_pos_correct.get(cat, 0)
            t_pos = overall_cat_pos_total.get(cat, 0)
            acc_pos = (c_pos / t_pos) if t_pos else 0.0
            writer.writerow({
                "category": cat,
                "time_correct": c_time,
                "time_total": t_time,
                "time_accuracy": f"{acc_time:.6f}",
                "pos_correct": c_pos,
                "pos_total": t_pos,
                "pos_accuracy": f"{acc_pos:.6f}",
            })

    print(f"\nCSV summary written to: {out_csv.resolve()}")
    print(f"Category CSV summary written to: {cat_out_csv.resolve()}")

if __name__ == "__main__":
    main()