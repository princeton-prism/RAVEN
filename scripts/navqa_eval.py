"""
Unified evaluation runner.

Usage examples:
  # Caption mode
  python scripts/navqa_eval.py --dataset navqa_caption --agent raven \
      --embedder siglip --vlm gpt4o --sequence_id 3 --postfix _0

  # VLM mode
  python scripts/navqa_eval.py --dataset navqa_vlm --agent raven \
      --embedder siglip --vlm gpt4o --sequence_id 3 --postfix _0

  # VLM-only (no embedder)
  python scripts/navqa_eval.py --dataset navqa_vlm --agent vlm_only \
      --embedder none --vlm gf25 --sequence_id 0 --postfix _0
"""

import json
import numpy as np
import tqdm
import re
import time
import os
import sys
import pickle as pkl
from PIL import Image as PILImage
import glob
from pathlib import Path

from dataclasses import asdict
import argparse
import traceback
import torch
import gc

# For LLM-based text evaluation
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from raven.agents.remembr_agent import ReMEmbRAgent
from raven.agents.raven_agent import RAVENAgent
from raven.agents.vlm_non_agent import VLMNonAgent
from raven.memory.memory import MemoryItem, VLMMemoryItem
from raven.memory.faiss_memory import FAISSMemory
from raven.memory.text_memory import TextMemory
from raven.memory.video_memory import VideoMemory, ImageMemoryItem
from raven.embedder.embedders import VLMEmbeddings
from raven.memory.memory_factory import MemoryFactory
from raven.utils.util import (
    predefined_filename,
    instantiate_from_yaml,
    print_cfg,
    int_or_json,
    hms_to_seconds,
    parse_json,
)

# ===================================================================
# Default prompt template (used by VLM mode)
# ===================================================================
DEFAULT_QUESTION_MESSAGE_NAVQA = """```json
        {{
            "type_reasoning": "-input your reasoning in here for the type of question-", 
            "type": "-input the type of answer that is expected based only on the question: position, binary, time, or text. Be sure to then fill in that selected category.",
            "answer_reasoning", "-input your reasoning in here for the answer. If you do not know the answer, provide your best guess for the answer type you provide.-", 
            "text": "--a text answer here--",
            "binary": "yes/no",
            "position": "[x,y,z]",
            "orientation": "[-.92]", 
            "time": 5.3,
            "duration": 2.4,
        }}
        ```
        """

# ===================================================================
# Evaluate output  (unified)
# ===================================================================

def evaluate_output(qa_instance, predicted, *, mode="caption",
                    scorer_llm=None, question=None, debug=False):
    """Score a single prediction against ground truth.

    Args:
        mode: "caption" uses eval.py semantics (time in seconds, duration direct).
              "vlm" uses preprocess_vlm_eval.py semantics (time relative to clock,
              duration seconds→minutes).
    """
    out_error = {}
    q_type = qa_instance['type']

    # --- position ---
    if 'position' in q_type:
        answer = np.array(qa_instance['answers']['position'])
        if type(predicted['position']) == str:
            predicted['position'] = eval(predicted['position'])
        pred_pos = np.array(predicted['position'])
        out_error['position_error'] = float(np.linalg.norm(answer - pred_pos))

    # --- binary ---
    elif 'binary' in q_type:
        answer = qa_instance['answers']['text'][1]
        if 'binary' in predicted and predicted['binary'].lower() in ("yes", "no"):
            correct = 1 if predicted['binary'].lower() == answer.lower() else 0
            out_error['binary_iscorrect'] = correct

    # --- time ---
    elif 'time' in q_type:
        answer = np.array(qa_instance['answers']['time'])

        if predicted['time'] is None:
            predicted['time'] = float('inf')
        elif type(predicted['time']) == str:
            if mode == "vlm":
                # VLM mode: convert predicted absolute time to relative minutes
                predicted_time_seconds = float(hms_to_seconds(predicted['time']))
                time_match = re.search(
                    r'The current time is \d{4}-\d{2}-\d{2} (\d{2}):(\d{2}):(\d{2})',
                    qa_instance['question'])
                if time_match:
                    h, m, s = map(int, time_match.groups())
                    curr_seconds = float(hms_to_seconds(f"{h}:{m}:{s}"))
                    predicted['time'] = (predicted_time_seconds - curr_seconds) / 60.0
                else:
                    raise ValueError("Could not find current time in the question")
            else:
                # Caption mode: convert hms to seconds directly
                predicted['time'] = float(hms_to_seconds(predicted['time']))

        pred_time = np.array(predicted['time'])
        out_error['time_error'] = float(abs(answer - pred_time))

    # --- duration ---
    elif 'duration' in q_type:
        answer = np.array(qa_instance['answers']['duration'])

        if type(predicted['duration']) == str:
            if mode == "vlm":
                # VLM mode: predicted is in seconds, convert to minutes
                predicted['duration'] = eval(predicted['duration']) / 60.0
            else:
                predicted['duration'] = eval(predicted['duration'])
        pred_time = np.array(predicted['duration'])
        out_error['duration_error'] = float(abs(answer - pred_time))

    # --- text ---
    elif 'text' in q_type:
        answer = qa_instance['answers']['text']
        out_error['answer'] = answer

        if scorer_llm is not None and question is not None:
            model_answer = predicted.get('text')
            correctness = judge_text_correctness(question, answer, model_answer, scorer_llm, debug=debug)
            if correctness is not None:
                out_error['text_iscorrect'] = correctness
                print(f"[TEXT] pred={str(model_answer)[:60]}..., gt={str(answer)[:60]}..., correct={correctness}")
            else:
                print(f"[TEXT] Evaluation failed for question: {question[:80]}...")

    else:
        raise Exception("We do not support question type " + q_type)

    return out_error


def judge_text_correctness(question, gt_answer, model_answer, scorer_llm, debug=False):
    """LLM-based semantic correctness judge for text answers."""
    if scorer_llm is None:
        return None
    if isinstance(gt_answer, list):
        gt_answer = str(gt_answer[0]) if gt_answer else ""
    else:
        gt_answer = str(gt_answer)
    model_answer = str(model_answer) if model_answer else ""

    prompt = f"""You are an expert evaluator for text question-answering.

    Determine whether the MODEL ANSWER correctly answers the QUESTION,
    based solely on semantic equivalence with the GROUND TRUTH ANSWER.

    Respond with EXACTLY one word: "correct" or "incorrect".

    QUESTION:
    {question}

    GROUND TRUTH ANSWER:
    {gt_answer}

    MODEL ANSWER:
    {model_answer}
    """
    if debug:
        print(f"########################################################")
        print(f"question: {question}")
        print(f"gt_answer: {gt_answer}")
        print(f"model_answer: {model_answer}")
        print(f"########################################################")
    try:
        messages = [HumanMessage(content=prompt)]
        response = scorer_llm.invoke(messages)
        response_text = response.content.strip().lower()
        print(f"Response text: {response_text}")
        if "correct" in response_text and "incorrect" not in response_text:
            return 1
        elif "incorrect" in response_text:
            return 0
        else:
            print(f"Warning: Unclear LLM response for text evaluation: {response_text}")
            return 0
    except Exception as e:
        print(f"Error in judge_text_correctness: {e}")
        traceback.print_exception(*sys.exc_info())
        return None


# ===================================================================
# Answer a question (unified)
# ===================================================================

def answer_question(model, question, qa_instance, *, mode="caption",
                    scorer_llm=None, debug=False):
    """Query the model and evaluate the response."""
    print(f'Question: {question}')
    parsed = None
    while True:
        try:
            start_time = time.time()
            response = model.query(question)
            end_time = time.time()
            elapsed = end_time - start_time

            if debug:
                print(f"##########################")
                print(f"response: {response}")
                print(f"##########################")

            parsed = asdict(response)

            if debug:
                print(f"##########################")
                print(f"parsed: {parsed}")
                print(f"##########################")

            out_error = evaluate_output(
                qa_instance, parsed,
                mode=mode,
                scorer_llm=scorer_llm,
                question=question,
                debug=debug,
            )
            print("Time elapsed", elapsed)

        except Exception as e:
            print(parsed)
            print(e)
            traceback.print_exception(*sys.exc_info())
            continue

        return_dict = {"response": parsed}
        return_dict.update(parsed)
        return_dict['error'] = out_error
        return_dict['elapsed'] = elapsed
        return return_dict


# ===================================================================
# Memory loading: caption mode 
# ===================================================================

def load_caption_memory(args, qa_instance, embedder=None, use_optimal_context=False):
    """Load memory from caption JSON files (eval.py path)."""
    start_time = qa_instance['start_time']
    end_time = qa_instance['end_time']

    # Generate collection name with embedder info if using VLM embedder
    if embedder is not None and args.backend is not None:
        if args.backend == 'oc':
            embedder_id = f"{args.backend}_{args.oc_model}_{args.emb_dim}"
        elif args.backend == 'hf':
            model_name = args.hf_model_id.split('/')[-1] if '/' in args.hf_model_id else args.hf_model_id
            embedder_id = f"{args.backend}_{model_name}_{args.emb_dim}"
        elif args.backend == 'ol':
            embedder_id = f"{args.backend}_{args.online_model_nickname}_{args.emb_dim}"
        else:
            embedder_id = f"{args.backend}_{args.emb_dim}"
        db_collection_name = f"eval_memory_{args.sequence_id}_{embedder_id}_v2"
    else:
        db_collection_name = f"eval_memory_{args.sequence_id}_v2"

    # Check and clean up dimension-mismatched FAISS indices
    if embedder is not None:
        import faiss as faiss_lib
        storage_path = args.memory_storage
        vlm_index_path = os.path.join(storage_path, f"{db_collection_name}_vlm.index")
        os.makedirs(storage_path, exist_ok=True)

        print(f"Checking FAISS index at: {vlm_index_path}")
        print(f"Expected dimension: {args.emb_dim}, Collection name: {db_collection_name}")

        if os.path.exists(vlm_index_path):
            try:
                existing_index = faiss_lib.read_index(vlm_index_path)
                print(f"Found existing FAISS index with dimension: {existing_index.d}")
                if existing_index.d != args.emb_dim:
                    print(f"Warning: dimension mismatch ({existing_index.d} vs {args.emb_dim}). Deleting old index files.")
                    for suffix in ['_vlm.index', '_position.index', '_time.index', '_metadata.json', '_id_counter.json']:
                        fp = os.path.join(storage_path, f"{db_collection_name}{suffix}")
                        if os.path.exists(fp):
                            os.remove(fp)
                            print(f"Deleted: {fp}")
                else:
                    print(f"FAISS index dimension matches ({existing_index.d}), keeping.")
            except Exception as e:
                print(f"Warning: Failed to load FAISS index: {e}. Deleting corrupted files.")
                for suffix in ['_vlm.index', '_position.index', '_time.index', '_metadata.json', '_id_counter.json']:
                    fp = os.path.join(storage_path, f"{db_collection_name}{suffix}")
                    if os.path.exists(fp):
                        os.remove(fp)
                        print(f"Deleted: {fp}")
        else:
            print(f"FAISS index file does not exist, will create new index.")

        # Check old collection names 
        old_collection_name = f"eval_memory_{args.sequence_id}_v2"
        if old_collection_name != db_collection_name:
            old_vlm_index_path = os.path.join(storage_path, f"{old_collection_name}_vlm.index")
            if os.path.exists(old_vlm_index_path):
                try:
                    old_index = faiss_lib.read_index(old_vlm_index_path)
                    if old_index.d != args.emb_dim:
                        print(f"Warning: old FAISS index '{old_collection_name}' dim mismatch. Deleting.")
                        for suffix in ['_vlm.index', '_position.index', '_time.index', '_metadata.json', '_id_counter.json']:
                            fp = os.path.join(storage_path, f"{old_collection_name}{suffix}")
                            if os.path.exists(fp):
                                os.remove(fp)
                                print(f"Deleted: {fp}")
                except Exception as e:
                    print(f"Warning: Failed to check old FAISS index: {e}")

    if embedder is not None:
        # Force delete dimension-mismatched indices
        import faiss as faiss_lib
        storage_path = args.memory_storage
        for suffix in ['_vlm.index', '_position.index', '_time.index', '_metadata.json', '_id_counter.json']:
            fp = os.path.join(storage_path, f"{db_collection_name}{suffix}")
            if os.path.exists(fp):
                if suffix == '_vlm.index':
                    try:
                        existing_index = faiss_lib.read_index(fp)
                        if existing_index.d != args.emb_dim:
                            print(f"Force deleting mismatched index: {fp}")
                            os.remove(fp)
                        else:
                            continue
                    except:
                        print(f"Force deleting corrupted index: {fp}")
                        os.remove(fp)
                else:
                    vlm_idx = os.path.join(storage_path, f"{db_collection_name}_vlm.index")
                    if not os.path.exists(vlm_idx):
                        os.remove(fp)
                        print(f"Deleted related file: {fp}")

        memory = MemoryFactory.create_memory(
            backend=args.memory_backend,
            db_collection_name=db_collection_name,
            embedder=embedder,
            storage_path=args.memory_storage,
            use_vlm_embedding=True,
            time_offset=start_time,
            dim=args.emb_dim,
            retriever_k=args.top_k,
            respond_with_score=args.add_score_info,
        )
    else:
        from langchain_huggingface import HuggingFaceEmbeddings
        mxbai_embedder = HuggingFaceEmbeddings(
            model_name='mixedbread-ai/mxbai-embed-large-v1',
            model_kwargs={'device': 'cpu'}
        )
        if args.framework == 'vlm_only':
            memory = VideoMemory()
        else:
            memory = FAISSMemory(
                db_collection_name, embedder=mxbai_embedder,
                storage_path=args.memory_storage,
                time_offset=start_time)

    # Check dimension before reset
    if embedder is not None and args.memory_backend == 'faiss_vlm':
        import faiss as faiss_lib
        storage_path = args.memory_storage
        vlm_index_path = os.path.join(storage_path, f"{memory.db_collection_name}_vlm.index")
        if os.path.exists(vlm_index_path):
            try:
                existing_index = faiss_lib.read_index(vlm_index_path)
                if existing_index.d != args.emb_dim:
                    print(f"Warning: Before reset(), dim mismatch. Deleting old index files.")
                    for suffix in ['_vlm.index', '_position.index', '_time.index', '_metadata.json', '_id_counter.json']:
                        fp = os.path.join(storage_path, f"{memory.db_collection_name}{suffix}")
                        if os.path.exists(fp):
                            os.remove(fp)
                            print(f"Deleted: {fp}")
            except Exception as e:
                print(f"Warning: Failed to check FAISS index before reset(): {e}")

    memory.reset()

    captions_path = os.path.join(
        args.data_dir, 'captions', str(args.sequence_id), 'captions', f'{args.caption_file}.json')
    with open(captions_path, 'r') as f:
        out = json.load(f)

    outputs = []

    all_start_times = np.array([float(x['file_start'][:-4]) for x in out])
    start_idx = np.argmin(np.abs(all_start_times - start_time))
    all_end_times = np.array([float(x['file_end'][:-4]) for x in out])
    end_idx = np.argmin(np.abs(all_end_times - end_time))

    pkl_files = glob.glob(os.path.join(args.coda_dir, str(args.sequence_id), '*.pkl'))
    pkl_files.sort(key=lambda x: float(x.split('/')[-1][:-4]))

    # Handle text embedding caching for caption embeddings
    dict_caption_to_emb = {}
    if embedder is not None and out:
        cache_dir = os.path.join(args.coda_dir, str(args.sequence_id), ".cache")
        os.makedirs(cache_dir, exist_ok=True)

        if args.backend == 'ol':
            cache_emb = os.path.join(cache_dir, f"{args.online_model_nickname}_{args.emb_dim}_caption_embeds.npy")
            cache_list = os.path.join(cache_dir, f"{args.online_model_nickname}_{args.emb_dim}_captions.json")
        elif args.hf_model_id == 'youzexue/QQMM-embed-v2':
            cache_emb = os.path.join(cache_dir, f"QQMM-embed-v2_{args.emb_dim}_caption_embeds.npy")
            cache_list = os.path.join(cache_dir, f"QQMM-embed-v2_{args.emb_dim}_captions.json")
        else:
            if args.backend == 'oc':
                mn = args.oc_model.replace('/', '_').replace('-', '_')
                cache_emb = os.path.join(cache_dir, f"{args.backend}_{mn}_{args.emb_dim}_caption_embeds.npy")
                cache_list = os.path.join(cache_dir, f"{args.backend}_{mn}_{args.emb_dim}_captions.json")
            else:
                cache_emb = os.path.join(cache_dir, f"{args.backend}_{args.emb_dim}_caption_embeds.npy")
                cache_list = os.path.join(cache_dir, f"{args.backend}_{args.emb_dim}_captions.json")

        if os.path.exists(cache_emb) and os.path.exists(cache_list):
            with open(cache_list, "r", encoding="utf-8") as f:
                cached_captions = json.load(f)
            arr = np.load(cache_emb)
            if len(arr) > 0 and len(arr[0]) != args.emb_dim:
                print(f"Warning: Cached embeddings dim {len(arr[0])} != {args.emb_dim}. Regenerating...")
                os.remove(cache_emb)
                os.remove(cache_list)
                dict_caption_to_emb = {}
            else:
                print(f"Loading {len(cached_captions)} caption embeddings from cache...")
                for i, caption in enumerate(cached_captions):
                    dict_caption_to_emb[caption] = arr[i]
                print(f"Loaded {len(cached_captions)} caption embeddings from cache!")
        else:
            print(f"Caching caption embeddings...")
            unique_captions = set()
            for entity in out:
                if 'caption' in entity and entity['caption']:
                    unique_captions.add(entity['caption'])
            captions_list = list(unique_captions)
            if captions_list:
                print(f"Embedding {len(captions_list)} captions...")
                for i in range(0, len(captions_list), args.batch_size):
                    batch = captions_list[i:i+args.batch_size]
                    text_prompts = [f"[TXT]{c}" for c in batch]
                    embs = embedder.embed_documents(text_prompts)
                    for j, cap in enumerate(batch):
                        dict_caption_to_emb[cap] = embs[j]
                    print(f"Processed {min(i+args.batch_size, len(captions_list))}/{len(captions_list)}")
            with open(cache_list, "w", encoding="utf-8") as f:
                json.dump(list(dict_caption_to_emb.keys()), f)
            np.save(cache_emb, np.array(list(dict_caption_to_emb.values())))
            print(f"Cached {len(dict_caption_to_emb)} caption embeddings")

    for i in range(start_idx, end_idx + 1):
        item = out[i]
        if embedder is not None:
            entity = {
                'position': item['position'],
                'theta': item['theta'],
                'time': item['time'],
                'caption': item['caption'],
                'image_file_path': None,
                'image_filenames': None,
            }
            outputs.append(entity)
            entity = VLMMemoryItem.from_dict(entity)
            caption_embedding = None
            if dict_caption_to_emb and 'caption' in item and item['caption']:
                caption_embedding = dict_caption_to_emb.get(item['caption'])
            elif 'caption' in item and item['caption']:
                caption_embedding = embedder.embed_query(f"[TXT]{item['caption']}")
            memory.insert(entity, vlm_embedding=caption_embedding)
        else:
            entity = {
                'position': item['position'],
                'theta': item['theta'],
                'time': item['time'],
                'caption': item['caption'],
            }
            outputs.append(entity)

            if type(memory) == VideoMemory:
                qa_start_path = os.path.join(args.coda_dir, str(args.sequence_id), out[i]['file_start'])
                qa_end_path = os.path.join(args.coda_dir, str(args.sequence_id), out[i+1]['file_start'])
                qa_start_idx = pkl_files.index(qa_start_path)
                qa_end_idx = pkl_files.index(qa_end_path)
                idxs = np.linspace(qa_start_idx, qa_end_idx, 6, dtype=int)
                for pkl_idx in idxs:
                    pkl_path = pkl_files[pkl_idx]
                    with open(pkl_path, 'rb') as f:
                        pkl_data = pkl.load(f)
                    entity['image'] = PILImage.fromarray(pkl_data['cam0'].astype('uint8'), 'RGB')
                entity = ImageMemoryItem.from_dict(entity)
            else:
                entity = MemoryItem.from_dict(entity)
            memory.insert(entity, text_embedding=item.get('text_embedding', None))

    if use_optimal_context:
        memory = TextMemory()
        memory.insert(qa_instance['context'])

    return memory, outputs


# ===================================================================
# Memory loading: VLM mode  
# ===================================================================

def load_vlm_memory(args, qa_instance, embedder, unit='seconds'):
    """Load memory from convert JSON files with image embeddings."""
    start_time = qa_instance['start_time']
    end_time = qa_instance['end_time']
    convert_file = args.convert_file

    if args.framework == 'raven':
        memory = MemoryFactory.create_memory(
            backend=args.memory_backend,
            db_collection_name="eval_memory_for_vlm",
            embedder=embedder,
            storage_path=args.memory_storage,
            use_vlm_embedding=True,
            time_offset=start_time if unit == 'seconds' else start_time / 1e6,
            dim=args.emb_dim,
            retriever_k=args.top_k,
            respond_with_score=args.add_score_info,
        )
    elif args.framework == 'vlm_only':
        memory = VideoMemory(start_time=start_time if unit == 'seconds' else start_time / 1e6)
    else:
        raise Exception(f"Unsupported agent framework: {args.framework}")

    memory.reset()
    with open(convert_file, 'r') as f:
        out = json.load(f)

    outputs = []

    all_start_times = np.array([float(x['start_time']) for x in out])
    start_idx = np.argmin(np.abs(all_start_times - start_time))
    all_end_times = np.array([float(x['end_time']) for x in out])
    end_idx = np.argmin(np.abs(all_end_times - end_time))

    pkl_files = glob.glob(os.path.join(args.coda_dir, str(args.sequence_id), '*.pkl'))
    pkl_files.sort(key=lambda x: float(x.split('/')[-1][:-4]))

    # Image embedding cache
    dict_fns_to_emb = {}
    if args.backend == 'ol' and out:
        cache_dir = os.path.join(args.coda_dir, str(args.sequence_id), ".cache")
        os.makedirs(cache_dir, exist_ok=True)
        cache_emb = os.path.join(cache_dir, f"{args.online_model_nickname}_image_embeds.npy")
        cache_list = os.path.join(cache_dir, f"{args.online_model_nickname}_image_filenames.json")
        if os.path.exists(cache_emb) and os.path.exists(cache_list):
            with open(cache_list, "r", encoding="utf-8") as f:
                cached_fns = json.load(f)
            arr = np.load(cache_emb)
            print(f"Loading {len(cached_fns)} embeddings from {cache_emb}...")
            for i, fn in enumerate(cached_fns):
                dict_fns_to_emb[fn] = arr[i]
        else:
            print(f"Caching {len(out)} image embeddings...")
            for entity_raw in out:
                if 'image_filenames' in entity_raw and entity_raw['image_filenames'] is not None:
                    embs = embedder.embed_documents(["[IMG]" + p for p in entity_raw['image_filenames']])
                    mean_emb = [sum(e) / len(e) for e in zip(*embs)]
                    dict_fns_to_emb[Path(entity_raw['image_filenames'][0]).name] = mean_emb
            with open(cache_list, "w", encoding="utf-8") as f:
                json.dump(list(dict_fns_to_emb.keys()), f)
            np.save(cache_emb, np.array(list(dict_fns_to_emb.values())))
            print(f"Cached {len(dict_fns_to_emb)} image embeddings")
    elif args.hf_model_id == 'youzexue/QQMM-embed-v2' and out:
        cache_dir = os.path.join(args.coda_dir, str(args.sequence_id), ".cache")
        os.makedirs(cache_dir, exist_ok=True)
        cache_emb = os.path.join(cache_dir, "QQMM-embed-v2_image_embeds.npy")
        cache_list = os.path.join(cache_dir, "QQMM-embed-v2_image_filenames.json")
        if os.path.exists(cache_emb) and os.path.exists(cache_list):
            with open(cache_list, "r", encoding="utf-8") as f:
                cached_fns = json.load(f)
            arr = np.load(cache_emb)
            print(f"Loading {len(cached_fns)} QQMM embeddings from cache...")
            for i, fn in enumerate(cached_fns):
                dict_fns_to_emb[fn] = arr[i]
        else:
            print(f"Caching {len(out)} QQMM image embeddings...")
            for entity_raw in out:
                if 'image_filenames' in entity_raw and entity_raw['image_filenames'] is not None:
                    embs = embedder.embed_documents(["[IMG]" + p for p in entity_raw['image_filenames']])
                    mean_emb = [sum(e) / len(e) for e in zip(*embs)]
                    dict_fns_to_emb[Path(entity_raw['image_filenames'][0]).name] = mean_emb
            with open(cache_list, "w", encoding="utf-8") as f:
                json.dump(list(dict_fns_to_emb.keys()), f)
            np.save(cache_emb, np.array(list(dict_fns_to_emb.values())))
            print(f"Cached {len(dict_fns_to_emb)} QQMM image embeddings")

    for i in range(start_idx, end_idx + 1):
        item = out[i]
        entity = {
            'position': item['position'],
            'time': item['time'] if unit == "seconds" else item['time'] / 1e6,
            'caption': item['caption'],
            'theta': item['theta'],
            'image_file_path': item.get('image_file_path'),
            'image_filenames': item['image_filenames'],
        }
        outputs.append(entity)
        entity = VLMMemoryItem.from_dict(entity)

        if args.framework == 'raven':
            vlm_emb = dict_fns_to_emb.get(Path(item['image_filenames'][0]).name) if dict_fns_to_emb else None
            memory.insert(entity, vlm_embedding=vlm_emb)
        elif args.framework == 'vlm_only':
            memory.insert(entity)
        else:
            raise Exception(f"Unsupported agent framework: {args.framework}")

    return memory, outputs


# ===================================================================
# Build legacy model string (for backward-compatible filenames)
# ===================================================================

def _build_model_string(agent_cfg, vlm_cfg, embedder_cfg):
    """Construct a legacy model string for backward-compatible filenames."""
    agent_type = agent_cfg['type']
    vlm_name = vlm_cfg['full_name']
    backend = embedder_cfg['backend']

    if backend is None:
        return f"{agent_type}+{vlm_name}"
    elif backend == 'oc':
        return (f"{agent_type}+{vlm_name}+oc"
                f"+{embedder_cfg['oc_model']}+{embedder_cfg['oc_pretrained']}+{embedder_cfg['emb_dim']}")
    elif backend == 'hf':
        return (f"{agent_type}+{vlm_name}+hf"
                f"+{embedder_cfg['hf_model_id']}+{embedder_cfg['vlm_layer']}+{embedder_cfg['emb_dim']}")
    elif backend == 'ol':
        return (f"{agent_type}+{vlm_name}+ol"
                f"+{embedder_cfg['online_model_nickname']}+{embedder_cfg['emb_dim']}")
    return f"{agent_type}+{vlm_name}"


# ===================================================================
# Main loop: caption mode  
# ===================================================================

def main_caption(args, embedder=None):
    """Run evaluation in caption mode."""
    use_optimal_context = False

    if args.framework == 'remembr':
        if embedder is not None:
            print(f"Initialized embedder backend={args.backend}, emb_dim={args.emb_dim}")
        agent = ReMEmbRAgent(
            llm_type=args.base_llm, num_ctx=args.num_ctx,
            temperature=args.temperature, debug=args.debug,
            prompt_dir=args.prompt_dir)
    else:
        raise Exception(f"Unsupported framework for caption mode: {args.framework}")

    # Initialize text scorer (always enabled)
    scorer_llm = None
    try:
        scorer_llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-pro", temperature=0.001, max_output_tokens=999999)
        print("Initialized gemini-2.5-pro as scorer LLM for text evaluation")
    except Exception as e:
        print(f"Warning: Failed to initialize scorer LLM: {e}")

    data_path = os.path.join(args.data_dir, 'questions', str(args.sequence_id), args.qa_file + '.json')
    data = json.load(open(data_path, 'r'))['data']

    running_successes = 0;          num_binary = 0
    running_pos_error = 0;          num_position = 0
    running_time_error = 0;         num_time = 0
    running_duration_error = 0;     num_duration = 0
    running_text_successes = 0;     num_text = 0
    running_description_successes = 0; num_description = 0

    responses = []
    for i in tqdm.tqdm(range(len(data)), total=len(data)):
        print(f"Evaluating {i} out of {len(data)}")
        qa_instance = data[i]
        question = qa_instance['question']
        qid = qa_instance['id']

        memory, instance_captions = load_caption_memory(
            args, data[i], embedder=embedder, use_optimal_context=use_optimal_context)
        if len(instance_captions) == 0:
            print("Length of Instance Captions is 0. It should not be")
            import pdb; pdb.set_trace()

        print("HISTORY LENGTH", len(instance_captions))
        agent.set_memory(memory)

        out_dict = answer_question(
            agent, question, qa_instance,
            mode="caption", scorer_llm=scorer_llm,
            debug=args.debug)

        out_dict['question'] = question
        out_dict['id'] = qid
        error_dict = out_dict['error']

        if qa_instance['type'] == 'position':
            num_position += 1
            if 'position_error' in error_dict:
                running_pos_error += error_dict['position_error']
        elif qa_instance['type'] == 'binary':
            num_binary += 1; num_description += 1
            if 'binary_iscorrect' in error_dict:
                running_successes += error_dict['binary_iscorrect']
                running_description_successes += error_dict['binary_iscorrect']
        elif qa_instance['type'] == 'time':
            num_time += 1
            if 'time_error' in error_dict:
                running_time_error += error_dict['time_error']
        elif qa_instance['type'] == 'duration':
            num_duration += 1
            if 'duration_error' in error_dict:
                running_duration_error += error_dict['duration_error']
        elif qa_instance['type'] == 'text':
            num_text += 1; num_description += 1
            if 'text_iscorrect' in error_dict:
                running_text_successes += error_dict['text_iscorrect']
                running_description_successes += error_dict['text_iscorrect']

        print("Question:", question)
        if 'response' in out_dict:
            print("Response:", out_dict['response'])
        print("Running Binary QA accuracy", running_successes / (num_binary + 1))
        print("Running Spatial Error", running_pos_error / (num_position + 1))
        print("Running Temporal Error", running_time_error / (num_time + 1))
        print("Running Duration Error", running_duration_error / (num_duration + 1))
        print("Running Text QA accuracy", running_text_successes / (num_text + 1))
        print("Running Description QA accuracy", running_description_successes / (num_description + 1))
        print()

        responses.append(out_dict)

        del memory; del instance_captions
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        if 'gemini' in args.base_llm.lower():
            time.sleep(2)

    # Save outputs
    out_json = {"version": 0.1, "responses": responses}
    out_path = os.path.join(args.out_dir, str(args.sequence_id), args.qa_file)
    os.makedirs(out_path, exist_ok=True)

    embedder_name_for_filename = None
    if embedder is not None and args.backend is not None:
        if args.backend == 'oc':
            embedder_name_for_filename = f"oc+{args.oc_model}+{args.oc_pretrained}+{args.emb_dim}"
        elif args.backend == 'hf':
            mn = args.hf_model_id.split('/')[-1] if '/' in args.hf_model_id else args.hf_model_id
            embedder_name_for_filename = f"hf+{mn}+{args.vlm_layer}+{args.emb_dim}"
        elif args.backend == 'ol':
            embedder_name_for_filename = f"ol+{args.online_model_nickname}+{args.emb_dim}"

    model_safe = args.model.replace('/', '_')
    if embedder_name_for_filename:
        emb_safe = embedder_name_for_filename.replace('/', '_')
        name = f"{model_safe}__{args.caption_file}_{emb_safe}{args.postfix}"
    else:
        name = f"{model_safe}__{args.caption_file}{args.postfix}"

    with open(os.path.join(out_path, f'{name}.json'), 'w') as f:
        json.dump(out_json, f, indent=4)
    print(f"Saved {name}.json")


# ===================================================================
# Main loop: VLM mode  
# ===================================================================

def main_vlm(args, embedder=None):
    """Run evaluation in VLM mode."""
    if args.framework == 'raven':
        agent = RAVENAgent(
            llm_type=args.base_llm, num_ctx=args.num_ctx,
            temperature=args.temperature, debug=args.debug,
            prompt_dir=args.vlm_prompt_dir)
    elif args.framework == 'vlm_only':
        agent = VLMNonAgent(
            llm_type=args.base_llm, num_ctx=args.num_ctx,
            temperature=args.temperature,
            question_message=args.question_message,
            prompt_dir=args.vlm_prompt_dir,
            max_message_length=args.max_message_length)
    else:
        raise Exception(f"Unsupported agent framework: {args.framework}")

    # Initialize text scorer (always enabled)
    scorer_llm = None
    try:
        scorer_llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-pro", temperature=0.001, max_output_tokens=999999)
        print("Initialized gemini-2.5-pro as scorer LLM for text evaluation (VLM mode)")
    except Exception as e:
        print(f"Warning: Failed to initialize scorer LLM: {e}")

    data = json.load(open(args.qa_file, 'r'))['data']

    running_successes = 0;         num_binary = 0
    running_pos_error = 0;         num_position = 0
    running_time_error = 0;        num_time = 0
    running_duration_error = 0;    num_duration = 0
    running_text_successes = 0;    num_text = 0
    running_description_successes = 0; num_description = 0

    responses = []
    for i in tqdm.tqdm(range(len(data)), total=len(data)):
        print(f"Evaluating {i} out of {len(data)}")
        qa_instance = data[i]
        question = qa_instance['question']
        qid = qa_instance['id']
        q_type = qa_instance['type']

        memory, instance_captions = load_vlm_memory(args=args, qa_instance=data[i], embedder=embedder)
        if len(instance_captions) == 0:
            print("Length of Instance Captions is 0. It should not be")
            import pdb; pdb.set_trace()

        print("HISTORY LENGTH", len(instance_captions))
        agent.set_memory(memory)

        out_dict = answer_question(
            agent, question, qa_instance, mode="vlm",
            scorer_llm=scorer_llm, debug=args.debug)

        out_dict['question'] = question
        out_dict['id'] = qid
        error_dict = out_dict['error']

        if q_type == 'position':
            num_position += 1
            if 'position_error' in error_dict:
                running_pos_error += error_dict['position_error']
        elif q_type == 'binary':
            num_binary += 1; num_description += 1
            if 'binary_iscorrect' in error_dict:
                running_successes += error_dict['binary_iscorrect']
                running_description_successes += error_dict['binary_iscorrect']
        elif q_type == 'time':
            num_time += 1
            if 'time_error' in error_dict:
                running_time_error += error_dict['time_error']
        elif q_type == 'duration':
            num_duration += 1
            if 'duration_error' in error_dict:
                running_duration_error += error_dict['duration_error']
        elif q_type == 'text':
            num_text += 1; num_description += 1
            if 'text_iscorrect' in error_dict:
                running_text_successes += error_dict['text_iscorrect']
                running_description_successes += error_dict['text_iscorrect']

        print("Question:", question)
        if 'response' in out_dict:
            print("Response:", out_dict['response'])
        print("Running Binary QA accuracy", running_successes / (num_binary + 1))
        print("Running Spatial Error", running_pos_error / (num_position + 1))
        print("Running Temporal Error", running_time_error / (num_time + 1))
        print("Running Duration Error", running_duration_error / (num_duration + 1))
        print("Running Text QA accuracy", running_text_successes / (num_text + 1))
        print("Running Description QA accuracy", running_description_successes / (num_description + 1))

        responses.append(out_dict)

    # Save
    out_json = {"version": 0.1, "responses": responses}
    os.makedirs(args.out_dir, exist_ok=True)

    model_safe = args.model.replace('/', '_')
    name = f"{model_safe}__{Path(args.convert_file).name[:-5]}__{Path(args.qa_file).name[:-5]}{args.postfix}"

    flag_overwrite = args.overwrite
    out_file = os.path.join(args.out_dir, f'{name}.json')
    if os.path.exists(out_file) and not flag_overwrite:
        print(f"Output file {out_file} already exists! Skipping...")
        return
    with open(out_file, 'w') as f:
        json.dump(out_json, f, indent=4)
        print(f"Saved {name}.json")


# ===================================================================
# CLI
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        prog='navqa_eval',
        description='Unified NaVQA evaluation runner (caption + VLM modes).',
    )
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset config name in cfgs/datasets/, e.g. navqa_caption, navqa_vlm")
    parser.add_argument("--agent", type=str, required=True,
                        help="Agent config name in cfgs/agents/, e.g. raven, vlm_only, remembr_original")
    parser.add_argument("--embedder", type=str, default="none",
                        help="Embedder config name in cfgs/embedders/, e.g. siglip, qqmm, none")
    parser.add_argument("--vlm", type=str, required=True,
                        help="VLM config name in cfgs/vlms/, e.g. gpt4o, gf25")
    parser.add_argument("--postfix", type=str, default="_0",
                        help="Postfix for output filename")
    parser.add_argument("--sequence_id", type=int, default=0,
                        help="Sequence ID to evaluate")
    parser.add_argument("--overwrite", action='store_true',
                        help="Overwrite existing output files")
    args = parser.parse_args()

    # ---- Load YAML configs ----
    dataset_cfg = instantiate_from_yaml(
        cfg_path=os.path.join('cfgs', 'datasets', f'{args.dataset}.yaml'), cls=None)
    agent_cfg = instantiate_from_yaml(
        cfg_path=os.path.join('cfgs', 'agents', f'{args.agent}.yaml'), cls=None)
    vlm_cfg = instantiate_from_yaml(
        cfg_path=os.path.join('cfgs', 'vlms', f'{args.vlm}.yaml'), cls=None)

    # ---- Instantiate embedder ----
    embedder_cfg_path = os.path.join('cfgs', 'embedders', f'{args.embedder}.yaml')
    if agent_cfg['type'] == 'remembr' and agent_cfg['text_emb_model'] == 'original':
        from langchain_huggingface import HuggingFaceEmbeddings
        embedder = HuggingFaceEmbeddings(model_name='mixedbread-ai/mxbai-embed-large-v1')
        embedder_cfg = instantiate_from_yaml(cfg_path=embedder_cfg_path, cls=None)
    elif agent_cfg['type'] in ['raven', 'remembr', 'embedder_only']:
        embedder, embedder_cfg = instantiate_from_yaml(cfg_path=embedder_cfg_path, cls=VLMEmbeddings)
    else:  # vlm_only
        embedder, embedder_cfg = None, instantiate_from_yaml(cfg_path=embedder_cfg_path, cls=None)

    print_cfg("Dataset", dataset_cfg)
    print_cfg("Agent", agent_cfg)
    print_cfg("VLM", vlm_cfg)
    print_cfg("Embedder", embedder_cfg)

    # ---- Populate args namespace from configs ----
    # Dataset
    args.data_dir       = dataset_cfg['data_dir']
    args.coda_dir       = dataset_cfg['coda_dir']
    args.out_dir        = dataset_cfg['out_dir']
    args.qa_file        = dataset_cfg['qa_file']
    args.caption_file   = dataset_cfg['caption_file']
    args.memory_storage = dataset_cfg['memory_storage']
    args.memory_backend = dataset_cfg['memory_backend']
    args.top_k          = dataset_cfg['top_k']
    args.temperature    = dataset_cfg['temperature']
    args.num_ctx        = dataset_cfg['num_ctx']
    args.debug          = dataset_cfg['debug']
    args.max_message_length = dataset_cfg['non_agent_max_image_len']
    args.question_message   = DEFAULT_QUESTION_MESSAGE_NAVQA
    args.prompt_dir         = dataset_cfg['prompt_folder']
    args.vlm_prompt_dir     = dataset_cfg['vlm_prompt_folder']

    # Agent
    args.framework      = agent_cfg['type']
    args.add_score_info = agent_cfg['add_score_info']

    # VLM
    args.base_llm = vlm_cfg['full_name']

    # Embedder
    args.backend              = embedder_cfg['backend']
    args.oc_model             = embedder_cfg['oc_model']
    args.oc_pretrained        = embedder_cfg['oc_pretrained']
    args.hf_model_id          = embedder_cfg['hf_model_id']
    args.vlm_layer            = embedder_cfg['vlm_layer']
    args.emb_dim              = embedder_cfg['emb_dim']
    args.batch_size           = embedder_cfg['batch_size']
    args.fp_16                = embedder_cfg['fp_16']
    args.vlm_text_prompts     = embedder_cfg['vlm_text_prompts']
    args.vlm_image_prompts    = embedder_cfg['vlm_image_prompts']
    args.online_model_nickname = embedder_cfg['online_model_nickname']
    args.device               = embedder_cfg['device']

    # Backward-compatible model string (used in filenames)
    args.model = _build_model_string(agent_cfg, vlm_cfg, embedder_cfg)

    # Build out_dir with VLM and embedder subdirectory
    vlm_nick = vlm_cfg['nick_name']
    embedder_nick = args.embedder  # config file name, e.g. qqmm, siglip, none
    args.out_dir = os.path.join(dataset_cfg['out_dir'], f"{vlm_nick}_{embedder_nick}")

    # ---- Dispatch based on dataset mode ----
    mode = dataset_cfg['mode']
    if mode == 'caption':
        main_caption(args, embedder)
    else:
        # VLM mode: auto-discover convert/qa files
        qa_files = sorted(glob.glob(
            os.path.join(args.data_dir, 'questions', str(args.sequence_id), 'human_qa.json')))
        convert_files = sorted(glob.glob(
            os.path.join(args.data_dir, 'convert', str(args.sequence_id), '*.json')))
        print(f"Loaded {len(qa_files)} QA files and {len(convert_files)} convert files")

        for qa_file, convert_file in zip(qa_files, convert_files):
            args.qa_file = qa_file
            args.convert_file = convert_file
            model_safe = args.model.replace('/', '_')
            out_name = os.path.join(
                args.out_dir,
                f"{model_safe}__{Path(convert_file).name[:-5]}__{Path(qa_file).name[:-5]}{args.postfix}.json")
            if os.path.exists(out_name) and not args.overwrite:
                print(f"Output file {out_name} already exists! Skipping...")
                continue
            print(f"Processing {qa_file} and {convert_file}")
            main_vlm(args, embedder)


if __name__ == "__main__":
    main()
