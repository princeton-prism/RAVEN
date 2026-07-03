import json
from collections import defaultdict, deque
from datetime import datetime, time
import os
from typing import List, Dict, Any, Optional, Tuple
import argparse
from datasets import load_dataset
from pathlib import Path
import cv2
import re
FPS_PER_MIN = 4.168
ANCHOR = time(6, 0, 0)  # 06:00:00


def load_data(args):
    dataset = load_dataset("yali30/findingdory", cache_dir="./data/.cache")["validation"]
    data_length = len(dataset)
    eps = None
    if args.eval_ep_ids != "-1": # which episode IDs to run
        eps = args.eval_ep_ids.split(',')

    dict_qa_data = defaultdict(list)
    if args.eval_num > 0: # how many examples to eval
        data_length = min(data_length, args.eval_num)
    elif args.eval_num == 0:
        raise Exception("eval_num cannot be 0")
    for i in range(data_length):
        instance = dataset[i]
        if eps is not None and instance['ep_id'] not in eps:
            continue
        instance["answer"] = json.loads(instance["answer"])
        assert isinstance(instance["answer"], list) and all(isinstance(elem, list) for elem in instance["answer"]), \
            f"Answer format incorrect for instance {i}, got: {instance['answer']}"
        dict_qa_data[instance['ep_id']].append(instance)
    ids = list(dict_qa_data.keys())
    print(f"Loaded {len(ids)} videos with total {data_length} QA pairs.")
    qa_data = []
    for id in ids:
        qa_data.append(dict_qa_data[id])

    return ids, qa_data


# ------------------------------ EVALUATION CODE PART START ------------------------------
def _hhmmss_to_frame_id(hhmmss: str) -> int:
    """
    Convert 'HH:MM:SS' to frame id:
      minutes_since_06:00:00 * 4.168, rounded to nearest integer.
    Negative minutes (before 06:00:00) are allowed but will likely not match GT.
    """
    dt = datetime.strptime(hhmmss, "%H:%M:%S").time()
    # Compute minutes since ANCHOR
    minutes = ((dt.hour - ANCHOR.hour) * 60) + (dt.minute - ANCHOR.minute) + (dt.second / 60.0)
    # Scale to frames
    return int(round(minutes * FPS_PER_MIN))

def _answers_match(pred_times: Optional[List[str]], gt_ans_lists: List[List[int]]) -> bool:
    """
    Apply the matching policy described:
      - If gt == [[-1]] (no answer), only correct if predicted is null.
      - Else if predicted is null -> wrong.
      - Else convert predictions to frames, compare in order to the first N gt lists.
      - If M < N -> wrong; if M >= N and first N are all in their candidate sets -> correct.
    """
    # Handle no-answer case
    # print(  f"GT answers: {gt_ans_lists}, Predicted times: {pred_times}"  )
    if len(gt_ans_lists) == 1 and len(gt_ans_lists[0]) == 1 and gt_ans_lists[0][0] == -1:
        # print("! No-answer case: ", pred_times is None)
        return pred_times is None

    # Otherwise we expect actual predictions
    if pred_times is None:
        # print("! Predicted times are None while GT expects answers.")
        return False

    # Convert predicted times to frame ids
    pred_frames = []
    for t in pred_times:
        try:
            pred_frames.append(_hhmmss_to_frame_id(t))
        except ValueError:
            # Bad time string -> treat as mismatch
            # print("! Bad time string encountered:", t)
            return False

    N = len(gt_ans_lists)
    # print("==============================")
    # print(len(gt_ans_lists), N, gt_ans_lists)
    # print("==============================")
    M = len(pred_frames)

    # If we have fewer predictions than needed, it's wrong
    if M < N:
        # print(f"! Fewer predictions ({M}) than GT answers ({N}).")
        return False

    # Check first N predicted frames against corresponding candidate lists
    for i in range(N):
        cand = set(gt_ans_lists[i])
        if pred_frames[i] not in cand:
            # if len(gt_ans_lists) > 1:
                # print(f"! Prediction {pred_frames[i]} not in GT candidates {cand} for answer {i}. {pred_frames} vs. {gt_ans_lists}")
            return False

    # If first N are correct, it's correct even if extra predictions exist
    # print("  > All required predictions matched GT candidates.", pred_frames[:N], gt_ans_lists)
    return True

def evaluate_response_file(out_dir, ep_id: str, gt: List[Dict[str, Any]]
) -> Tuple[float, Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, float]]:
    """
    Evaluate a single response JSON file against the provided ground-truth list.

    Returns:
      overall_acc,
      per_task_id_acc, per_high_level_acc, per_low_level_acc, per_num_interactions_acc
    """

    response_json_path = os.path.join(out_dir, (f'{ep_id}_output.json').replace("/", "_"))
    

    with open(response_json_path, "r", encoding="utf-8") as f:
        resp_data = json.load(f)
    


    # Build response mapping question -> predicted times (or None)
    responses = resp_data.get("responses", [])

    debug_logs = resp_data.get("debug_logs", [])
    memory_visited_logs = []
    this_set = None
    for log in debug_logs:
        if log == "[RST]":
            if this_set is not None:
                memory_visited_logs.append(len(this_set))
            this_set = set()
        else:
            text = str(log)
            pattern = r"At time=1969-12-31 (\d{2}:\d{2}:\d{2})"
            pattern2 = r"At time=1970-01-01 (\d{2}:\d{2}:\d{2})"

            matches = re.findall(pattern, text)
            matches2 = re.findall(pattern2, text)
            matches = ["1969-12-31 " + m for m in matches] + ["1970-01-01 " + m for m in matches2]
            this_set.update(matches)        
    if this_set is not None:
        memory_visited_logs.append(len(this_set))

    # Tally
    total = 0
    correct = 0

    random_num = 0.0

    # Per-category tallies
    per_task_id_tot = defaultdict(int); per_task_id_ok = defaultdict(int)
    per_high_tot = defaultdict(int);    per_high_ok = defaultdict(int)
    per_low_tot = defaultdict(int);     per_low_ok = defaultdict(int)
    per_numint_tot = defaultdict(int);  per_numint_ok = defaultdict(int)
    per_high_random_tot = defaultdict(float);    per_high_random_ok = defaultdict(float)

    video_path = Path("data/findingdory") / gt[0]["video"]
    # get frame count using opencv
    cap = cv2.VideoCapture(str(video_path))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
        

    # Evaluate only for GT questions that appear in the response file
    for kk, gt_row, pred_times in zip(range(len(responses)), gt, responses):
        prob = 1.0
        for ans in gt_row["answer"]:
            prob *= len(ans) / frame_count
        random_num += prob
        is_ok = _answers_match(pred_times["time"].split() if pred_times["time"] else None, gt_row["answer"])
        total += 1
        if is_ok:
            correct += 1
            # print(f"[ep{ep_id}-id{kk}] Question: {gt_row['question']} --> Correct")
        # else:
        #     print(f"[ep{ep_id}-id{kk}] Question: {gt_row['question']} --> Wrong. Predicted: {pred_times['time']}")

        # Update category buckets
        _update_category_tallies(
            is_ok, gt_row,
            per_task_id_tot, per_task_id_ok,
            per_high_tot, per_high_ok,
            per_low_tot, per_low_ok,
            per_numint_tot, per_numint_ok,
            per_high_random_tot, per_high_random_ok, prob
        )

    overall_acc = (correct / total) if total > 0 else 0.0

    per_task_id_acc = _ratio_dict(per_task_id_ok, per_task_id_tot)
    per_high_acc    = _ratio_dict(per_high_ok,    per_high_tot)
    per_low_acc     = _ratio_dict(per_low_ok,     per_low_tot)
    per_numint_acc  = _ratio_dict(per_numint_ok,  per_numint_tot)

    print(f"Evaluation Results for {ep_id}:")
    print(f"Overall Accuracy: {overall_acc:.4f}")
    print(f"Per Task ID Accuracy: {per_task_id_acc}")
    print(f"Per High Level Category Accuracy: {per_high_acc}")
    print(f"Per Low Level Category Accuracy: {per_low_acc}")
    print(f"Per Number of Interactions Accuracy: {per_numint_acc}")

    return overall_acc, per_task_id_acc, per_high_acc, per_low_acc, per_numint_acc, \
        correct, total, per_task_id_ok, per_task_id_tot, \
            per_high_ok, per_high_tot, per_low_ok, per_low_tot, \
                per_numint_ok, per_numint_tot, random_num, frame_count, \
                per_high_random_ok, per_high_random_tot, memory_visited_logs


def _ratio_dict(ok: Dict[Any, int], tot: Dict[Any, int]) -> Dict[str, float]:
    return {str(k): (ok[k] / tot[k] if tot[k] > 0 else 0.0) for k in tot.keys()}

def _update_category_tallies(
    is_ok: bool,
    row: Dict[str, Any],
    per_task_id_tot, per_task_id_ok,
    per_high_tot, per_high_ok,
    per_low_tot, per_low_ok,
    per_numint_tot, per_numint_ok, 
    per_high_random_tot, per_high_random_ok, rand_ok
):
    # task_id
    tid = row.get("task_id", "UNKNOWN")
    per_task_id_tot[tid] += 1
    if is_ok: per_task_id_ok[tid] += 1

    # high_level_category
    hi = row.get("high_level_category", "UNKNOWN")
    per_high_tot[hi] += 1
    if is_ok: per_high_ok[hi] += 1

    # low_level_category
    lo = row.get("low_level_category", "UNKNOWN")
    per_low_tot[lo] += 1
    if is_ok: per_low_ok[lo] += 1

    # num_interactions
    ni = row.get("num_interactions", "UNKNOWN")
    per_numint_tot[ni] += 1
    if is_ok: per_numint_ok[ni] += 1

    # high_level_random_category
    hi_ran = row.get("high_level_category", "UNKNOWN")
    per_high_random_tot[hi_ran] += 1
    per_high_random_ok[hi_ran] += rand_ok

# ------------------------------ EVALUATION CODE PART END ------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                        prog='Long Horizon Robot QA',
                        description='Runs various LLMs on the QA dataset',)
    
    parser.add_argument("--out_dir", type=str, default="./output/xxxx", help="output directory where the response files are stored")
    parser.add_argument("--eval_num", type=int, help="how many eval examples to run, -1 means all", default=-1)
    parser.add_argument("--eval_ep_ids", type=str, help="which episode IDs to run, -1 means all, e.g., ep_1,ep_3", default="-1")
    args = parser.parse_args()

    ids, qa_data = load_data(args)
    all_results = {"frame_count": {}}
    for ep_id, qa_dp in zip(ids, qa_data):
        print(ep_id)
        name = os.path.join(args.out_dir, (f'{ep_id}_output.json').replace("/", "_"))
        if not os.path.exists(name):
            break
        print(f"Processing {ep_id}...")
        overall_acc, per_task_id_acc, per_high_acc, per_low_acc, per_numint_acc, \
            correct, total, per_task_id_ok, per_task_id_tot, \
                per_high_ok, per_high_tot, per_low_ok, per_low_tot, per_numint_ok, per_numint_tot, random_num, \
                    frame_count, per_high_random_ok, per_high_random_tot, memory_visited_logs = evaluate_response_file(args.out_dir, ep_id, qa_dp)
        # update all_results
        all_results["frame_count"][ep_id] = frame_count
        all_results[f'memory_visited_logs'] = all_results.get(f'memory_visited_logs', []) + memory_visited_logs
        all_results["overall_correct"] = all_results.get("overall_correct", 0) + correct
        all_results["overall_total"] = all_results.get("overall_total", 0) + total
        all_results["random_num"] = all_results.get("random_num", 0) + random_num
        if "per_task_id_ok" not in all_results:
            all_results["per_task_id_ok"] = {k: 0 for k in per_task_id_ok}
            all_results["per_task_id_tot"] = {k: 0 for k in per_task_id_tot}
        for k in per_task_id_ok:
            all_results["per_task_id_ok"][k] = all_results["per_task_id_ok"].get(k, 0) + per_task_id_ok[k]
            all_results["per_task_id_tot"][k] = all_results["per_task_id_tot"].get(k, 0) + per_task_id_tot[k]
        if "per_high_ok" not in all_results:
            all_results["per_high_ok"] = {k: 0 for k in per_high_ok}
            all_results["per_high_tot"] = {k: 0 for k in per_high_tot}
        for k in per_high_ok:
            all_results["per_high_ok"][k] = all_results["per_high_ok"].get(k, 0) + per_high_ok[k]
            all_results["per_high_tot"][k] = all_results["per_high_tot"].get(k, 0) + per_high_tot[k]
        if "per_low_ok" not in all_results:
            all_results["per_low_ok"] = {k: 0 for k in per_low_ok}
            all_results["per_low_tot"] = {k: 0 for k in per_low_tot}
        for k in per_low_ok:
            all_results["per_low_ok"][k] = all_results["per_low_ok"].get(k, 0) + per_low_ok[k]
            all_results["per_low_tot"][k] = all_results["per_low_tot"].get(k, 0) + per_low_tot[k]
        if "per_numint_ok" not in all_results:
            all_results["per_numint_ok"] = {k: 0 for k in per_numint_ok}
            all_results["per_numint_tot"] = {k: 0 for k in per_numint_tot}
        for k in per_numint_ok:
            all_results["per_numint_ok"][k] = all_results["per_numint_ok"].get(k, 0) + per_numint_ok[k]
            all_results["per_numint_tot"][k] = all_results["per_numint_tot"].get(k, 0) + per_numint_tot[k]

        if "per_high_random_ok" not in all_results:
            all_results["per_high_random_ok"] = {k: 0.0 for k in per_high_random_ok}
            all_results["per_high_random_tot"] = {k: 0.0 for k in per_high_random_tot}
        for k in per_high_random_ok:
            all_results["per_high_random_ok"][k] = all_results["per_high_random_ok"].get(k, 0.0) + per_high_random_ok[k]
            all_results["per_high_random_tot"][k] = all_results["per_high_random_tot"].get(k, 0.0) + per_high_random_tot[k]
    random_acc = all_results["random_num"] / all_results["overall_total"] if all_results["overall_total"] > 0 else 0.0
    overall_acc = all_results["overall_correct"] / all_results["overall_total"] if all_results["overall_total"] > 0 else 0.0
    per_task_id_acc = _ratio_dict(all_results["per_task_id_ok"], all_results["per_task_id_tot"])
    per_high_acc = _ratio_dict(all_results["per_high_ok"], all_results["per_high_tot"])
    per_low_acc = _ratio_dict(all_results["per_low_ok"], all_results["per_low_tot"])
    per_numint_acc = _ratio_dict(all_results["per_numint_ok"], all_results["per_numint_tot"])
    per_high_random_acc = _ratio_dict(all_results["per_high_random_ok"], all_results["per_high_random_tot"])
    print("Final Evaluation Results:")
    print(f"Overall Accuracy: {overall_acc:.4f}")
    print(f"Per Task ID Accuracy: {per_task_id_acc}")
    print(f"Per High Level Category Accuracy: {per_high_acc}")
    print(f"Per Low Level Category Accuracy: {per_low_acc}")
    print(f"Per Number of Interactions Accuracy: {per_numint_acc}")
    print(f"Mean frame count: { float(sum(all_results['frame_count'][k] for k in all_results['frame_count'])) / len(all_results['frame_count']) if len(all_results['frame_count']) > 0 else 0 }")
    print(f"Total query number: { all_results['overall_total']  }")
    print(f"Memory visited logs: {float(sum(all_results['memory_visited_logs'])) / len(all_results['memory_visited_logs']) if len(all_results['memory_visited_logs']) > 0 else 0}")


    print(f"Per High Level Category Random Baseline Accuracy: {per_high_random_acc}")
    print(f"Random Baseline Accuracy: {random_acc:.4f}")
