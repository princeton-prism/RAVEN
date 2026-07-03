"""
Unified accuracy computation script.

Usage examples:
  # Caption mode (replaces eval_acc.py)
  python scripts/navqa_acc.py --mode caption \
      --out_root ./out --qa_root ./data/questions --result_root ./data/test_results

  # VLM mode - single file (replaces postprocess_vlm_acc.py --pred_path)
  python scripts/navqa_acc.py --mode vlm \
      --pred_path ./output/result.json --seq_id 3 --qa_root ./data_vlm/questions

  # VLM mode - batch directory (replaces postprocess_vlm_acc.py --result_dir)
  python scripts/navqa_acc.py --mode vlm \
      --result_dir ./output/results --qa_root ./data_vlm/questions
"""

import argparse
import json
import math
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from raven.utils.util import load_json
import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CATEGORIES = ['SHORT', 'MEDIUM', 'LONG']

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

class TeeOutput:
    """Write to both stdout and a log file."""
    def __init__(self, file_path: Path):
        self.terminal = sys.stdout
        self.log_file = open(file_path, 'w', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

    def close(self):
        if self.log_file:
            self.log_file.close()

# ---------------------------------------------------------------------------
# Ground-truth indexing
# ---------------------------------------------------------------------------

def _strip_seq_prefix(qid: str) -> str:
    """Remove leading ``{digits}_`` prefix from a question ID."""
    if '_' in qid and qid.split('_', 1)[0].isdigit():
        return qid.split('_', 1)[1]
    return qid


def build_gt_map(human_qa: dict) -> Dict[str, Dict[str, dict]]:
    """Build multiple indices for robust ID matching.

    Returns a dict containing:
      - by_full:   exact id -> item
      - by_no_seq: id with leading digits+underscore removed -> item
      - by_uuid:   last segment after '_' (UUID) -> item
    """
    by_full: Dict[str, dict] = {}
    by_no_seq: Dict[str, dict] = {}
    by_uuid: Dict[str, dict] = {}
    for item in human_qa['data']:
        qid = item['id']
        if not qid:
            continue
        by_full[qid] = item

        by_no_seq[_strip_seq_prefix(qid)] = item

        uuid_part = qid.rsplit('_', 1)[-1]
        by_uuid[uuid_part] = item

    return {"by_full": by_full, "by_no_seq": by_no_seq, "by_uuid": by_uuid}


def _resolve_gt_item(qid: str, gt_maps: dict):
    """Resolve a ground-truth item via full / no-seq / uuid fallbacks."""
    gt_item = gt_maps["by_full"].get(qid)
    if gt_item is None:
        gt_item = gt_maps["by_no_seq"].get(_strip_seq_prefix(qid))
    if gt_item is None:
        uuid_part = qid.rsplit('_', 1)[-1]
        gt_item = gt_maps["by_uuid"].get(uuid_part)
    return gt_item


# ---------------------------------------------------------------------------
# Metric primitives
# ---------------------------------------------------------------------------

def compute_position_error(pred: List[float], gt: List[float]) -> Optional[float]:
    """L2 norm between predicted and ground-truth 3-D positions."""
    if pred is None or gt is None:
        return None
    if len(pred) != 3 or len(gt) != 3:
        return None
    diff = np.array(pred, dtype=float) - np.array(gt, dtype=float)
    return float(np.linalg.norm(diff))


def extract_number_from_text(text: str) -> Optional[float]:
    """Extract the first reasonable numeric value from free-form text."""
    if not text:
        return None
    best = None
    for m in re.findall(r'-?\d+\.?\d*', text):
        try:
            val = float(m)
        except ValueError:
            continue
        if 0 <= val < 1000:
            return val
        if best is None and 0 <= val < 10000:
            best = val
    return best


def compute_time_error(
    pred: Optional[float],
    gt: Optional[float],
    max_value: float = 1000.0,
) -> Optional[float]:
    """Absolute difference between *pred* and *gt* in the caller's unit.

    Parameters
    ----------
    pred : float or None
        Predicted temporal value (seconds or minutes -- depends on caller).
    gt : float or None
        Ground-truth temporal value in the same unit as *pred*.
    max_value : float
        Reject *pred* or *gt* whose absolute value exceeds this bound
        (default 1000, suitable for minutes after caption-mode normalisation).

    Returns
    -------
    float or None
        ``abs(pred - gt)``, or None when either input is None / non-finite /
        or unreasonably large per *max_value*.
        # NOTE: max_value can be modified to filter out outliers
    """
    if pred is None or gt is None:
        return None
    try:
        pred_val = float(pred)
        gt_val = float(gt)
        if not math.isfinite(pred_val) or not math.isfinite(gt_val):
            return None
        if abs(pred_val) > max_value or abs(gt_val) > max_value:
            return None
        return abs(pred_val - gt_val)
    except Exception:
        return None


def is_binary_correct(pred: Optional[str], gt_text_list: List[str]) -> Optional[int]:
    """Return 1 if pred matches any GT answer (case-insensitive), else 0."""
    if pred is None or not gt_text_list:
        return None
    pred_norm = str(pred).strip().lower()
    gt_norms = [str(x).strip().lower() for x in gt_text_list]
    return 1 if pred_norm in gt_norms else 0


def aggregate_mean_std(values: List[float]) -> Tuple[Optional[float], Optional[float]]:
    """Mean and std (ddof=0) of finite values; (None, None) if empty."""
    valid = [v for v in values if v is not None and math.isfinite(v)]
    if not valid:
        return None, None
    arr = np.array(valid, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=0))

# ---------------------------------------------------------------------------
# Shared temporal-entry processor (time / duration)
# ---------------------------------------------------------------------------

def _process_temporal_entry(
    label: str,
    entry: dict,
    error_key: str,
    field_name: str,
    gt_ans: dict,
    get_field,
    gt_item: dict,
    cat: str,
    qid: str,
    recompute_errors: bool,
    result_list: List[float],
    seconds_to_minutes: bool = False,
) -> None:
    """Unified logic for 'time' and 'duration' question types.

    Handles precomputed errors, suspicious values, text-extraction fallback,
    and inf/nan filtering consistently for both question types.

    Parameters
    ----------
    label : str
        Display label for log messages (e.g. 'TIME', 'DURATION').
    entry : dict
        Single prediction entry from the responses list.
    error_key : str
        Key inside ``entry['error']`` for the precomputed error value.
    field_name : str
        Key for the predicted / GT temporal field ('time' or 'duration').
    gt_ans : dict
        Ground-truth answers dict from human_qa.json.
    get_field : callable
        Accessor that looks up a field in the prediction entry.
    gt_item : dict
        Full ground-truth item (used for the question text fallback).
    cat : str
        Length category ('SHORT', 'MEDIUM', 'LONG').
    qid : str
        Question ID.
    recompute_errors : bool
        If True, ignore precomputed errors and recompute from raw values.
    result_list : list of float
        Accumulator list; computed errors (in minutes) are appended here.
    seconds_to_minutes : bool
        If True, *pred_raw*, *gt_val*, and *precomputed* are in seconds;
        they are converted to minutes before thresholds and aggregation.
    """
    err = entry['error']
    precomputed = (err.get(error_key) if not recompute_errors else None)
    pred_raw = get_field(field_name)
    gt_val = gt_ans[field_name]
    text = get_field('text') or ''
    question = entry.get('question') or gt_item['question']

    if seconds_to_minutes:
        if pred_raw is not None:
            try:
                pred_raw = float(pred_raw) / 60.0
            except (ValueError, TypeError):
                pass
        if gt_val is not None:
            try:
                gt_val = float(gt_val) / 60.0
            except (ValueError, TypeError):
                pass
        if precomputed is not None:
            try:
                precomputed = float(precomputed) / 60.0
            except (ValueError, TypeError):
                pass

    # --- forced recompute path ---
    if recompute_errors:
        error = compute_time_error(pred_raw, gt_val)
        if error is not None:
            result_list.append(float(error))
            print(f"[{label}] cat={cat}, qid={qid}, pred={pred_raw}, gt={gt_val}, "
                  f"error={float(error):.3f}min (recomputed), question={question[:80]}...")
        return

    # --- check if predicted value looks suspicious ---
    pred_invalid = False
    if pred_raw is not None:
        try:
            pv = float(pred_raw)
            if not math.isfinite(pv) or pv < 0 or abs(pv) > 1000:
                pred_invalid = True
        except (ValueError, TypeError):
            pred_invalid = True

    # --- check if precomputed error looks suspicious ---
    err_invalid = False
    if precomputed is not None:
        try:
            ev = float(precomputed)
            if not math.isfinite(ev) or ev > 50:
                err_invalid = True
        except (ValueError, TypeError):
            err_invalid = True

    if pred_invalid or err_invalid:
        # Suspicious pred/error: try extracting a number from the text answer
        extracted = extract_number_from_text(text)
        if seconds_to_minutes and extracted is not None:
            extracted = extracted / 60.0
        if extracted is not None and gt_val is not None:
            error = compute_time_error(extracted, gt_val)
            if error is not None:
                result_list.append(float(error))
                print(f"[{label}] cat={cat}, qid={qid}, pred={pred_raw}->extracted={extracted}, "
                      f"gt={gt_val}, error={float(error):.3f}min (from_text), question={question[:80]}...")
        else:
            error = compute_time_error(pred_raw, gt_val)
            if error is not None:
                result_list.append(float(error))
                print(f"[{label}] cat={cat}, qid={qid}, pred={pred_raw}, gt={gt_val}, "
                      f"error={float(error):.3f}min (direct), question={question[:80]}...")
            else:
                print(f"[{label}] cat={cat}, qid={qid}, pred={pred_raw}, gt={gt_val}, "
                      f"error=SKIPPED (invalid), question={question[:80]}...")
    elif precomputed is None:
        error = compute_time_error(pred_raw, gt_val)
        if error is not None:
            result_list.append(float(error))
            print(f"[{label}] cat={cat}, qid={qid}, pred={pred_raw}, gt={gt_val}, "
                  f"error={float(error):.3f}min, question={question[:80]}...")
    else:
        try:
            ev = float(precomputed)
        except (ValueError, TypeError):
            ev = float('nan')
        if math.isfinite(ev):
            result_list.append(ev)
            print(f"[{label}] cat={cat}, qid={qid}, pred={pred_raw}, gt={gt_val}, "
                  f"error={ev:.3f}min (precomputed), question={question[:80]}...")
        else:
            print(f"[{label}] cat={cat}, qid={qid}, pred={pred_raw}, gt={gt_val}, "
                  f"error=SKIPPED (invalid precomputed), question={question[:80]}...")

# ---------------------------------------------------------------------------
# Core: compute per-file metric lists
# ---------------------------------------------------------------------------

def compute_per_file_lists(
    pred_path: Path,
    qa_path: Path,
    *,
    mode: str,
    recompute_errors: bool = False,
) -> Dict[str, Dict[str, List[float]]]:
    """Return raw metric lists per category for one prediction file.

    Parameters
    ----------
    pred_path : Path
        Path to prediction JSON.
    qa_path : Path
        Path to ground-truth human_qa.json.
    mode : str
        ``"caption"`` or ``"vlm"``.  Caption-mode predictions store
        temporal values in seconds; VLM-mode stores them in minutes.
        The function normalises everything to minutes before returning.
    recompute_errors : bool
        If True, ignore precomputed errors and recompute from GT.

    Returns
    -------
    dict
        ``{ category: { 'acc_list': [...], 'pos_list': [...], 'tmp_list': [...],
                         'text_acc_list': [...] } }``
    """
    preds = load_json(pred_path)
    human_qa = load_json(qa_path)
    gt_maps = build_gt_map(human_qa)

    all_responses = preds['responses']
    seconds_to_minutes = (mode == "caption")

    # Buckets
    desc_acc:  Dict[str, List[int]]   = {k: [] for k in CATEGORIES}
    pos_err:   Dict[str, List[float]] = {k: [] for k in CATEGORIES}
    temp_err:  Dict[str, List[float]] = {k: [] for k in CATEGORIES}
    text_acc:  Dict[str, List[int]]   = {k: [] for k in CATEGORIES}

    seen_qids: set = set()
    for entry in all_responses:
        if not isinstance(entry, dict) or not entry:
            continue
        qid = entry['id']
        if not qid or qid in seen_qids:
            continue
        seen_qids.add(qid)

        gt_item = _resolve_gt_item(qid, gt_maps)
        if gt_item is None:
            continue

        # Determine category
        cat = str(gt_item['length_category']).upper()
        if cat not in CATEGORIES:
            potential = _strip_seq_prefix(qid).split('_', 1)[0].upper()
            cat = potential if potential in CATEGORIES else None
        if cat not in CATEGORIES:
            continue

        gt_ans = gt_item['answers']
        gt_type = gt_item['type'].strip().lower()

        resp_nested = entry.get('response')

        # Look up field in entry first, then in nested 'response' dict
        def get_field(field_name):
            if field_name in entry:
                return entry[field_name]
            if resp_nested is not None and isinstance(resp_nested, dict) and field_name in resp_nested:
                return resp_nested[field_name]
            return None

        kind = gt_type or str(get_field('type') or '').strip().lower()

        # --- binary ---
        if kind == 'binary':
            err = entry['error']
            corr = (err.get('binary_iscorrect') if not recompute_errors else None)
            pred_bin = get_field('binary')
            gt_texts = gt_ans['text']
            question = entry.get('question') or gt_item['question']
            if corr is None:
                corr = is_binary_correct(pred_bin, gt_texts)
            if corr is not None:
                desc_acc[cat].append(int(corr))
                src = "(recomputed)" if recompute_errors else "(from file)"
                print(f"[BINARY] cat={cat}, qid={qid}, pred={pred_bin}, gt={gt_texts}, "
                      f"correct={corr} {src}, question={question[:80]}...")

        # --- position ---
        elif kind == 'position':
            err = entry['error']
            pe = (err.get('position_error') if not recompute_errors else None)
            pred_pos = get_field('position')
            gt_pos = gt_ans['position']
            question = entry.get('question') or gt_item['question']
            if pe is None:
                pe = compute_position_error(pred_pos, gt_pos)
            if pe is not None:
                pos_err[cat].append(float(pe))
                src = "(recomputed)" if recompute_errors else "(from file)"
                print(f"[POSITION] cat={cat}, qid={qid}, pred={pred_pos}, gt={gt_pos}, "
                      f"error={pe:.3f}m {src}, question={question[:80]}...")

        # --- time ---
        elif kind == 'time':
            _process_temporal_entry(
                'TIME', entry, 'time_error', 'time', gt_ans, get_field,
                gt_item, cat, qid, recompute_errors, temp_err[cat],
                seconds_to_minutes=seconds_to_minutes,
            )

        # --- duration ---
        elif kind == 'duration':
            _process_temporal_entry(
                'DURATION', entry, 'duration_error', 'duration', gt_ans, get_field,
                gt_item, cat, qid, recompute_errors, temp_err[cat],
                seconds_to_minutes=seconds_to_minutes,
            )

        # --- text ---
        elif kind == 'text':
            pred_text = get_field('text')
            gt_text_list = gt_ans['text']
            question = entry.get('question') or gt_item['question']

            err = entry['error']
            corr = err.get('text_iscorrect')
            if corr is not None:
                text_acc[cat].append(int(corr))
                print(f"[TEXT] cat={cat}, qid={qid}, "
                      f"pred={str(pred_text)[:60] if pred_text else 'None'}..., "
                      f"gt={str(gt_text_list[0])[:60] if gt_text_list else 'None'}..., "
                      f"correct={corr} (from file), question={question[:80]}...")
            else:
                print(f"[TEXT] cat={cat}, qid={qid}, skipped "
                      f"(text_iscorrect not found), question={question[:80]}...")

    result = {
        k: {
            'acc_list': desc_acc[k],
            'pos_list': pos_err[k],
            'tmp_list': temp_err[k],
            'text_acc_list': text_acc[k],
        } for k in CATEGORIES
    }

    return result

# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def aggregate_across_files(
    per_file_lists: List[Dict[str, Dict[str, List[float]]]],
) -> Dict[str, Dict[str, List[float]]]:
    """Concatenate per-file metric lists across all files."""
    agg = {k: {'acc_list': [], 'pos_list': [], 'tmp_list': [], 'text_acc_list': []} for k in CATEGORIES}
    for per_cat in per_file_lists:
        for cat in CATEGORIES:
            agg[cat]['acc_list'].extend(per_cat[cat]['acc_list'])
            agg[cat]['pos_list'].extend(per_cat[cat]['pos_list'])
            agg[cat]['tmp_list'].extend(per_cat[cat]['tmp_list'])
            agg[cat]['text_acc_list'].extend(per_cat[cat]['text_acc_list'])
    return agg

# ---------------------------------------------------------------------------
# Output: save summary to JSON + TXT
# ---------------------------------------------------------------------------

def _build_saved_dict(
    agg_lists: Dict[str, Dict[str, List[float]]],
    model_name: Optional[str] = None,
    extra_meta: Optional[dict] = None,
) -> dict:
    """Build the structured dict that gets dumped to JSON."""
    saved: dict = {}
    if model_name is not None:
        saved["model_name"] = model_name
    if extra_meta:
        saved.update(extra_meta)
    saved["by_category"] = {}

    for cat in CATEGORIES:
        acc_vals = agg_lists[cat]['acc_list']
        pos_vals = agg_lists[cat]['pos_list']
        tmp_vals = agg_lists[cat]['tmp_list']
        text_acc_vals = agg_lists[cat]['text_acc_list']

        binary_acc_mean = None if not acc_vals else float(np.mean(acc_vals))
        pos_mean, pos_std = aggregate_mean_std(pos_vals)
        tmp_mean, tmp_std = aggregate_mean_std(tmp_vals)
        text_acc_mean = None if not text_acc_vals else float(np.mean(text_acc_vals))

        combined = acc_vals + text_acc_vals
        desc_mean = None if not combined else float(np.mean(combined))

        cat_dict: dict = {
            "descriptive_accuracy": None if desc_mean is None else {
                "mean": desc_mean,
                "n": len(combined),
                "n_binary": len(acc_vals),
                "n_text": len(text_acc_vals),
            },
            "binary_accuracy": None if binary_acc_mean is None else {
                "mean": binary_acc_mean,
                "n": len(acc_vals),
            },
            "positional_error_m": None if pos_mean is None else {
                "mean": pos_mean,
                "std": pos_std or 0.0,
                "n": len(pos_vals),
            },
            "temporal_error_min": None if tmp_mean is None else {
                "mean": tmp_mean,
                "std": tmp_std or 0.0,
                "n": len(tmp_vals),
            },
            "text_accuracy": None if text_acc_mean is None else {
                "mean": text_acc_mean,
                "n": len(text_acc_vals),
            },
        }
        saved["by_category"][cat] = cat_dict

    return saved


def _write_txt_summary(f, saved: dict, header_note: str = "") -> None:
    """Write a human-readable summary to file handle *f*."""
    if header_note:
        f.write(header_note + "\n")

    f.write("[Descriptive Accuracy]\n")
    for cat in CATEGORIES:
        entry = saved["by_category"][cat]
        da = entry["descriptive_accuracy"]
        if da is None:
            f.write(f"  {cat}: N/A (no binary or text questions)\n")
        else:
            f.write(f"  {cat}: {da['mean']:.3f} "
                    f"(n={da['n']}, binary={da['n_binary']}, "
                    f"text={da['n_text']})\n")
    f.write("\n")

    f.write("[Binary Accuracy]\n")
    for cat in CATEGORIES:
        ba = saved["by_category"][cat]["binary_accuracy"]
        if ba is None:
            f.write(f"  {cat}: N/A (no binary questions)\n")
        else:
            f.write(f"  {cat}: {ba['mean']:.3f} (n={ba['n']})\n")
    f.write("\n")

    f.write("[Positional Error]\n")
    for cat in CATEGORIES:
        pe = saved["by_category"][cat]["positional_error_m"]
        if pe is None:
            f.write(f"  {cat}: N/A (n=0)\n")
        else:
            f.write(f"  {cat}: {pe['mean']:.3f}±{pe['std']:.3f} m (n={pe['n']})\n")
    f.write("\n")

    f.write("[Temporal Error]\n")
    for cat in CATEGORIES:
        te = saved["by_category"][cat]["temporal_error_min"]
        if te is None:
            f.write(f"  {cat}: N/A (n=0)\n")
        else:
            f.write(f"  {cat}: {te['mean']:.3f}±{te['std']:.3f} min (n={te['n']})\n")
    f.write("\n")

    f.write("[Text Accuracy]\n")
    for cat in CATEGORIES:
        ta = saved["by_category"][cat]["text_accuracy"]
        if ta is None:
            f.write(f"  {cat}: N/A (no text questions)\n")
        else:
            f.write(f"  {cat}: {ta['mean']:.3f} (n={ta['n']})\n")
    f.write("\n")


def save_summary(
    agg_lists: Dict[str, Dict[str, List[float]]],
    out_dir: Path,
    *,
    model_name: Optional[str] = None,
    header_note: str = "",
    extra_meta: Optional[dict] = None,
) -> None:
    """Save aggregate JSON + TXT summary to *out_dir*."""
    saved = _build_saved_dict(agg_lists,
                              model_name=model_name, extra_meta=extra_meta)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "acc_cat_summmary.json", "w") as f:
        json.dump(saved, f, indent=2)

    with open(out_dir / "acc_cat_summmary.txt", "w") as f:
        _write_txt_summary(f, saved, header_note=header_note)


def postprocess_single(
    pred_path: Path,
    qa_path: Path,
    out_dir: Optional[Path] = None,
    *,
    mode: str,
    recompute_errors: bool = False,
) -> None:
    """Evaluate a single prediction file and print + save results.

    Parameters
    ----------
    pred_path : Path
        Path to prediction JSON.
    qa_path : Path
        Path to ground-truth human_qa.json.
    out_dir : Path or None
        Output directory for summaries.  Falls back to a VLM default
        when None.
    mode : str
        ``"caption"`` or ``"vlm"`` -- controls temporal-error unit
        normalisation (caption stores seconds, vlm stores minutes).
    recompute_errors : bool
        If True, ignore precomputed errors and recompute from GT.
    """
    per_cat = compute_per_file_lists(
        pred_path, qa_path,
        mode=mode,
        recompute_errors=recompute_errors,
    )

    def fmt(m, s, unit=""):
        if m is None:
            return "N/A"
        return f"{m:.3f}±{(s or 0.0):.3f} {unit}" if unit else f"{m:.3f}±{(s or 0.0):.3f}"

    print(f"Prediction file: {pred_path}")
    print(f"Questions file:  {qa_path}\n")

    for cat in CATEGORIES:
        acc = per_cat[cat]['acc_list']
        pos = per_cat[cat]['pos_list']
        tmp = per_cat[cat]['tmp_list']
        txt = per_cat[cat]['text_acc_list']
        combined = acc + txt

        desc_mean = None if not combined else float(np.mean(combined))
        bin_mean = None if not acc else float(np.mean(acc))
        pos_mean, pos_std = aggregate_mean_std(pos)
        tmp_mean, tmp_std = aggregate_mean_std(tmp)
        txt_mean = None if not txt else float(np.mean(txt))

        print(f"[{cat}]")
        if desc_mean is None:
            print("  Descriptive Accuracy: N/A")
        else:
            print(f"  Descriptive Accuracy: {desc_mean:.3f} "
                  f"(n={len(combined)}, binary={len(acc)}, text={len(txt)})")
        if bin_mean is not None:
            print(f"  Binary Accuracy:      {bin_mean:.3f} (n={len(acc)})")
        if txt_mean is not None:
            print(f"  Text Accuracy:        {txt_mean:.3f} (n={len(txt)})")
        print(f"  Positional Error:     {fmt(pos_mean, pos_std, 'm')} (n={len(pos)})")
        print(f"  Temporal Error:       {fmt(tmp_mean, tmp_std, 'min')} (n={len(tmp)})")
        print()

    if out_dir is None:
        model_embed_dir = pred_path.parent.name
        out_dir = Path("./data_vlm/test_results") / model_embed_dir

    save_summary(
        per_cat, out_dir,
        extra_meta={"prediction_file": str(pred_path), "questions_file": str(qa_path)},
    )

# ---------------------------------------------------------------------------
# NavQA-specific helpers
# ---------------------------------------------------------------------------

def parse_model_name_from_filename(filename: str) -> Optional[str]:
    """Extract ``vlm+captioner+text_embedder`` from a NavQA prediction filename.

    Examples::

        'remembr+gemini-2.5-flash__captions_gpt-4o-mini_3_secs_0.json'
            -> 'gemini-2.5-flash+gpt-4o-mini+mxbai'
        'remembr+gpt-4o+hf+QQMM-embed-v2+0+3584__captions_Qwen2.5_5_secs_0.json'
            -> 'gpt-4o+Qwen2.5+QQMM-embed-v2'
    """
    base = filename.replace('.json', '')
    base = re.sub(r'_\d+_secs_\d+$', '', base)
    base = re.sub(r'_\d+$', '', base)
    if base.startswith('remembr+'):
        base = base[8:]

    if '__' not in base:
        return base.split('_')[0] if '_' in base else base

    parts = base.split('__', 1)
    model_part, caption_part = parts[0], parts[1]

    vlm = None
    text_embedder = "mxbai"
    if '+' in model_part:
        model_parts = model_part.split('+')
        vlm = model_parts[0]
        if len(model_parts) >= 3 and 'QQMM' in model_parts[2]:
            text_embedder = "QQMM-embed-v2"
    else:
        vlm = model_part

    captioner = None
    if caption_part.startswith('captions_'):
        caption_rest = caption_part[9:]
        caption_rest = re.sub(r'_\d+$', '', caption_rest)
        captioner = caption_rest
    elif 'captions' in caption_part:
        caption_rest = caption_part.split('captions', 1)[1]
        if caption_rest.startswith('_'):
            caption_rest = caption_rest[1:]
        caption_rest = re.sub(r'_\d+$', '', caption_rest)
        captioner = caption_rest

    if not vlm or not captioner:
        return model_part
    return f"{vlm}+{captioner}+{text_embedder}"


# ===================================================================
# Mode: caption
# ===================================================================

def _run_caption(args) -> None:
    """Caption mode: group predictions by model name, aggregate per model."""
    out_root = Path(args.out_root)
    qa_root = Path(args.qa_root)
    result_root = Path(args.result_root)

    if not out_root.exists():
        raise FileNotFoundError(f"Output directory not found: {out_root}")

    # Group prediction files by model name
    model_files: Dict[str, List[Tuple[Path, int]]] = {}
    for seq_dir in sorted(out_root.iterdir()):
        if not seq_dir.is_dir():
            continue
        try:
            seq_id = int(seq_dir.name)
        except ValueError:
            print(f"[SKIP] Cannot parse seq_id from directory: {seq_dir.name}")
            continue
        human_qa_dir = seq_dir / "human_qa"
        if not human_qa_dir.exists():
            continue
        for json_file in human_qa_dir.glob("*.json"):
            model_name = parse_model_name_from_filename(json_file.name)
            if model_name is None:
                print(f"[SKIP] Cannot parse model name: {json_file.name}")
                continue
            model_files.setdefault(model_name, []).append((json_file, seq_id))

    if not model_files:
        print("No valid prediction files found.")
        return

    print(f"Found {len(model_files)} unique models:")
    for mn, fl in model_files.items():
        print(f"  {mn}: {len(fl)} files")

    for model_name, file_list in model_files.items():
        model_out_dir = result_root / model_name
        model_out_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = model_out_dir / f"eval_acc_{timestamp}.log"

        original_stdout = sys.stdout
        tee = TeeOutput(log_file)
        sys.stdout = tee
        try:
            print(f"Logging output to: {log_file}")
            print(f"\nProcessing model: {model_name}")
            per_file_lists: List[Dict] = []
            for pred_path, seq_id in file_list:
                qa_path = qa_root / str(seq_id) / 'human_qa.json'
                if not qa_path.exists():
                    print(f"  [SKIP] human_qa.json not found for seq_id={seq_id}")
                    continue
                print(f"  Processing seq {seq_id}: {pred_path.name}")
                try:
                    per_file = compute_per_file_lists(
                        pred_path, qa_path,
                        mode="caption",
                        recompute_errors=args.recompute_errors,
                    )
                    per_file_lists.append(per_file)
                except Exception as e:
                    print(f"  [ERROR] {pred_path.name}: {e}")
            if not per_file_lists:
                print(f"  No valid files processed for model {model_name}")
                continue
            agg = aggregate_across_files(per_file_lists)
            header = f"Aggregated across {len(per_file_lists)} files for model {model_name}"
            save_summary(agg, model_out_dir,
                         model_name=model_name, header_note=header)
            print(f"  Saved results to: {model_out_dir}")
        finally:
            sys.stdout = original_stdout
            tee.close()
            print(f"  Log saved to: {log_file}")

# ===================================================================
# Mode: vlm 
# ===================================================================

def _run_vlm(args) -> None:
    """VLM mode: single-file (--pred_path) or batch directory (--result_dir)."""
    if args.pred_path:
        # --- single-file mode ---
        if args.seq_id is None:
            raise ValueError("--seq_id is required when using --pred_path")
        pred_path = Path(args.pred_path)
        if not pred_path.exists():
            raise FileNotFoundError(f"Prediction file not found: {pred_path}")
        qa_path = (Path(args.qa_path) if args.qa_path
                   else Path(args.qa_root) / str(args.seq_id) / 'human_qa.json')
        if not qa_path.exists():
            raise FileNotFoundError(f"Questions file not found: {qa_path}")

        out_dir = Path(args.out_dir) if args.out_dir else None
        postprocess_single(
            pred_path, qa_path, out_dir=out_dir,
            mode="vlm",
            recompute_errors=args.recompute_errors,
        )
        print("\n" + "=" * 80)
        print("Evaluation completed successfully!")
        print("=" * 80)

    elif args.result_dir:
        # --- batch directory mode ---
        result_dir = Path(args.result_dir)
        if not result_dir.exists() or not result_dir.is_dir():
            raise FileNotFoundError(f"Result directory not found: {result_dir}")

        qa_root = Path(args.qa_root)
        per_file_lists: List[Dict] = []
        json_files = sorted([
            p for p in result_dir.glob("*.json")
            if "human_qa_" in p.name
        ])
        if not json_files:
            print(f"No JSON prediction files found in: {result_dir}")
            return

        for json_path in json_files:
            base = json_path.stem
            seq_id = None
            if base.rsplit('_', 1)[-1].isdigit():
                seq_id = int(base.rsplit('_', 1)[-1])
            if seq_id is None:
                print(f"[SKIP] Cannot infer seq_id from filename: {json_path.name}")
                continue
            qa_path = qa_root / str(seq_id) / 'human_qa.json'
            if not qa_path.exists():
                print(f"[SKIP] human_qa.json not found for seq_id={seq_id}")
                continue

            print(f"Processing seq {seq_id}: {json_path.name}")
            per_file = compute_per_file_lists(
                json_path, qa_path,
                mode="vlm",
                recompute_errors=args.recompute_errors,
            )
            per_file_lists.append(per_file)

        if not per_file_lists:
            print("No valid files processed; nothing to summarize.")
            return

        out_dir = Path(args.out_dir) if args.out_dir else result_dir
        header = f"Aggregated across {len(per_file_lists)} files in {result_dir}"
        agg = aggregate_across_files(per_file_lists)
        save_summary(agg, out_dir, header_note=header)
        print("\n" + "=" * 80)
        print(f"Batch evaluation completed! {len(per_file_lists)} files from {result_dir}")
        print(f"Summary saved to: {out_dir}/acc_cat_summmary.json and {out_dir}/acc_cat_summmary.txt")
        print("=" * 80)
    else:
        raise ValueError("Either --pred_path or --result_dir is required in vlm mode")

# ===================================================================
# CLI
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        prog='navqa_acc',
        description='Unified accuracy computation for NavQA and VLM evaluation outputs.',
    )
    parser.add_argument('--mode', type=str, required=True, choices=['caption', 'vlm'],
                        help="'caption' (caption-mode outputs) or 'vlm' (VLM-mode outputs)")

    # --- shared flags ---
    parser.add_argument('--qa_root', type=str, default=None,
                        help='Root directory with per-seq human_qa.json (default depends on mode)')
    parser.add_argument('--recompute_errors', action='store_true',
                        help='Recompute errors from GT instead of using precomputed values')

    # --- caption-specific ---
    parser.add_argument('--out_root', type=str, default='./out',
                        help='[caption] Root dir with out/{seq_id}/human_qa/*.json')
    parser.add_argument('--result_root', type=str, default='./data/test_results',
                        help='[caption] Output dir for per-model summaries')

    # --- vlm-specific ---
    parser.add_argument('--pred_path', type=str, default=None,
                        help='[vlm] Single prediction JSON file')
    parser.add_argument('--result_dir', type=str, default=None,
                        help='[vlm] Directory with prediction JSONs (batch mode)')
    parser.add_argument('--seq_id', type=int, default=None,
                        help='[vlm] Sequence ID (required for single-file mode)')
    parser.add_argument('--qa_path', type=str, default=None,
                        help='[vlm] Explicit path to human_qa.json (single-file mode)')
    parser.add_argument('--out_dir', type=str, default=None,
                        help='[vlm] Output directory for summaries')
    args = parser.parse_args()

    # Set mode-specific defaults for qa_root
    if args.qa_root is None:
        args.qa_root = './data/questions' if args.mode == 'caption' else './data_vlm/questions'

    if args.mode == 'caption':
        _run_caption(args)
    else:
        _run_vlm(args)

if __name__ == '__main__':
    main()
