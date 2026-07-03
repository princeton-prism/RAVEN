import sys, os, re
import time, json
import argparse, traceback
from pathlib import Path
from dataclasses import asdict
from collections import defaultdict
import tqdm
import cv2
from PIL import Image as PILImage
import numpy as np

# load this directory
sys.path.append(sys.path[0] + '/..')
from datasets import load_dataset
from langchain_huggingface import HuggingFaceEmbeddings
from raven.agents.embedder_only_agent import EmbedderOnlyAgent 
from raven.agents.raven_agent import RAVENAgent
from raven.agents.remembr_agent import ReMEmbRAgent
from raven.agents.vlm_non_agent import VLMNonAgent
from raven.agents.agent import AgentOutput
from raven.embedder.embedders import VLMEmbeddings
from raven.memory.memory import VLMMemoryItem, MemoryItem
from raven.memory.memory_factory import MemoryFactory
from raven.memory.video_memory import VideoMemory
from raven.captioners.captioner import OpenAICaptioner
from raven.utils.util import int_or_json, instantiate_from_yaml, print_cfg

INGAME_FPM = 4.168
START_TIME = -46800 # -13 hours in seconds to keep it in the same date

def answer_question(model, question):

    print(f'Question: {question}')

    parsed = None
    while True:
        try:

            start_time = time.time()
            response = model.query(question)
            end_time = time.time()

            elapsed = end_time - start_time

            # ##### embedder_only specific parsing start #############
            if isinstance(response, int) or response is None:
                if response is None:
                    response = AgentOutput(None, None, None, None, None, None, None)
                else:
                    time_frame = response
                    time_second = time_frame / INGAME_FPM * 60 + START_TIME
                    time_str = time.strftime('%H:%M:%S', time.localtime(time_second))
                    response = AgentOutput(None, None, None, None, None, None, time_str)
            # ##### embedder_only specific parsing end ############### 


            parsed = asdict(response)

            print("Time elapsed", elapsed)

        except Exception as e:
            print(parsed)
            print(e)
            traceback.print_exception(*sys.exc_info()) 
            continue

        return_dict = {"response": parsed}
        return_dict.update(parsed)
        return_dict['elapsed'] = elapsed

        return return_dict
    

def convert_video_frames_and_make_annotations(agent_cfg,
                                              dataset_cfg,
                                              embedder_cfg,
                                              video_path,
                                              output_dir,
                                              embedder=None):
    """
    Convert video frames to images and create annotations for each frame.
    """
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception(f"Cannot open video file {video_path}")
    assert time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(START_TIME)) == '1969-12-31 06:00:00'
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    caption_file_path = Path(video_path).parent / ".cache" / Path(video_path).stem / "captions.json"
    
    have_to_caption = agent_cfg['type'] == 'remembr'
    load_captions = caption_file_path.exists()
    captions = None
    if have_to_caption and load_captions:
        # a list of dict with keys: caption, txt_embed_original, txt_embed_sota
        with open(caption_file_path, 'r') as f:
            captions_data = json.load(f)
            step_num = captions_data['step']
            captions = captions_data['captions']
            length_needed = len(range(0, frame_count, step_num))
            if len(captions) < length_needed:
                print(f"Warning: Loaded captions length {len(captions)} is less than needed {length_needed}, will make up the rest captions.")
                load_captions = False
            else:
                print(f"Loaded {len(captions)} captions from {caption_file_path}, step num {step_num}")

    annotations = []
    embed_mode = 'txt_embed_original' if agent_cfg['text_emb_model'] == 'original' else 'txt_embed_sota'
    for i in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break
        frame_time = 60 * i / INGAME_FPM + START_TIME  # in seconds, since video starts at 6:00 AM
        frame_filename = os.path.join(output_dir, f"frame_{i:04d}.jpg")
        cv2.imwrite(frame_filename, frame)

        annotations.append({
            "time": frame_time, # in seconds
            "caption": (captions[i // step_num]['caption'] if i % step_num == 0 else "No caption") if have_to_caption and load_captions else None, # placeholder caption
            "text_embedding": captions[i // step_num][embed_mode] if have_to_caption and load_captions else None, # placeholder for text embedding
            "image_file_path": frame_filename
        })
    

    if have_to_caption and (not load_captions):
        def caption_embed_video_frames(frame_path, captioner=None, embedder_sota=None, embedder_original=None):
            try:
                im = PILImage.open(frame_path).convert("RGB")
            except Exception as e:
                print(f"Skip unreadable image: {frame_path} ({e})")
            # caption (single image list to reuse your captioner API)
            cap_text = captioner.caption([im])
            # embed text
            text_emb_sota = embedder_sota.embed_query("[TXT]" + cap_text)
            text_emb_original = embedder_original.embed_query("[TXT]" + cap_text)
            return {"caption": cap_text, "txt_embed_original": text_emb_original, "txt_embed_sota": text_emb_sota}

        class ConfigCap:
            openai_api_key: str = ""
            openai_model: str = "gpt-4o-mini"
            query: str = "You are wandering around a house. Please describe in detail what you see in the image. Specifically focus on the objects, environmental features, events/activities, people, and other interesting details. Think step by step about these details and be very specific. Thank you!"
            max_new_tokens: int = 5120
            temperature: float = 0.2

        args_cap = ConfigCap()
        args_cap.openai_api_key = os.getenv("OPENAI_API_KEY")
        captioner = OpenAICaptioner(args_cap)
        
        if agent_cfg["text_emb_model"] == 'original':
            embedder_original = embedder
            embedder_sota = VLMEmbeddings(
                embedder_cfg["device"],
                backend="hf",
                oc_model=None,
                oc_pretrained=None,
                hf_model_id="youzexue/QQMM-embed-v2",
                vlm_layer=0,
                batch_size=embedder_cfg["batch_size"],
                fp_16=embedder_cfg["fp_16"],
                emb_dim=3584,
                vlm_text_prompts=embedder_cfg["vlm_text_prompts"],
                vlm_image_prompts=embedder_cfg["vlm_image_prompts"],
                online_model_nickname=None
            )
        else:
            embedder_original = HuggingFaceEmbeddings(model_name='mixedbread-ai/mxbai-embed-large-v1')
            embedder_sota = embedder
        if captions is None:
            captions = []
            step_num = dataset_cfg["caption_step_num"]
        for i in tqdm.tqdm(range(0, frame_count, step_num)):
            if i // step_num  < len(captions):
                continue
            frame_filename = os.path.join(output_dir, f"frame_{i:04d}.jpg")
            captions.append(caption_embed_video_frames(frame_filename, captioner, embedder_sota, embedder_original))
            # save every 50 captions to avoid data loss
            if len(captions) % 20 == 0:
                os.makedirs(caption_file_path.parent, exist_ok=True)
                with open(caption_file_path, 'w') as f:
                    json.dump({"step": step_num, "captions": captions}, f, indent=4)
        for i in range(frame_count):
            annotations[i]['caption'] = captions[i // step_num]['caption'] if i % step_num == 0 else "No caption"
            annotations[i]['text_embedding'] = captions[i // step_num][embed_mode]

        # save captions
        os.makedirs(caption_file_path.parent, exist_ok=True)
        with open(caption_file_path, 'w') as f:
            json.dump({"step": step_num, "captions": captions}, f, indent=4)
    
    cap.release()
    
    return annotations


def load_memory(agent_cfg, dataset_cfg, embedder_cfg, output_dir_basename,
                embedder:VLMEmbeddings|HuggingFaceEmbeddings|None, 
                video_path):
    if agent_cfg['type'] in ['raven', 'embedder_only', 'remembr']:
        memory = MemoryFactory.create_memory(
            backend=dataset_cfg['memory_backend'],
            db_collection_name=agent_cfg['type'],
            embedder=embedder,
            storage_path=os.environ.get('RAVEN_MEMORY_STORAGE', dataset_cfg.get('memory_storage_path', './output/memory_storage')),
            use_vlm_embedding=agent_cfg['type'] != 'remembr',
            time_offset=0,
            dim=embedder_cfg['emb_dim'],
            retriever_k=dataset_cfg['top_k'],
            respond_with_score=agent_cfg['add_score_info'],
        )
    elif agent_cfg['type'] == 'vlm_only':
        memory = VideoMemory(start_time=0)
    else:
        raise Exception("Unsupported agent type for memory creation!")
    
    memory.reset()
    # import ipdb; ipdb.set_trace()
    image_cache_root = os.environ.get('RAVEN_IMAGE_CACHE', dataset_cfg.get('image_cache_path', './temp_image_cache'))
    out_info = convert_video_frames_and_make_annotations(agent_cfg,
                                                         dataset_cfg,
                                                         embedder_cfg,
                                                         video_path,
                                                         output_dir=os.path.join(image_cache_root, output_dir_basename),
                                                         embedder=embedder)

    outputs = []

    emb_arr = None
    if embedder_cfg['backend'] in ['ol', 'hf']:
        cache_dir = os.path.join(Path(video_path).parent, ".cache", str(ep_id))
        os.makedirs(cache_dir, exist_ok=True)
        cache_name = embedder_cfg['online_model_nickname'] if embedder_cfg['backend'] == 'ol' else embedder_cfg['hf_model_id'].replace('/', '_')
        cache_emb = os.path.join(cache_dir, f"{cache_name}_image_embeds.npy")
        if os.path.exists(cache_emb):
            emb_arr = np.load(cache_emb)
        else:
            emb_arr = []
            for entity in out_info:
                print("Caching image embeddings to memory. This may take a while...")
                emb_arr.append(embedder.embed_query("[IMG]" + entity['image_file_path']))
            np.save(cache_emb, np.array(emb_arr))

    for i in range(len(out_info)):

        item = out_info[i]
        entity = {
            'position': [0., 0., 0.], # we don't have position info in the captions right now
            'time': item['time'],
            'caption': item['caption'],
            'theta': 3.14, # we don't have theta info in the captions right now
            'image_file_path': item['image_file_path']
        }

        outputs.append(entity)

        entity = VLMMemoryItem.from_dict(entity) if agent_cfg['type'] != 'remembr' else MemoryItem.from_dict(entity)

        if agent_cfg['type'] == 'raven' or agent_cfg['type'] == 'embedder_only':
            memory.insert(entity, vlm_embedding=emb_arr[i] if emb_arr is not None else None) # if None, will make up embeddings inside memory
        elif agent_cfg['type'] == 'vlm_only':
            memory.insert(entity)
        elif agent_cfg['type'] == 'remembr':
            memory.insert(entity, text_embedding=item['text_embedding'])
        else:
            raise Exception("We only support [raven, vlm_only, remembr, embedder_only] for now")
        
    return memory, outputs


def question_special_formatting(question: str) -> str:
    # for finding dory, we need to ensure the question ends with a question mark
    assert isinstance(question, str), "Question must be a string"
    if "The robot's goal is: " in question:
        question = question.replace("The robot's goal is: ", "")
    if ":" in question:
        def normalize_time(match):
            time_str = match.group()
            hour, minute = time_str.split(":")
            hour = hour.zfill(2)      # ensure two digits
            minute = minute.zfill(2)  # ensure two digits
            return f"{hour}:{minute}:00"
        
        pattern = r'\b(?:[0-1]?\d|2[0-3]):[0-5]?\d\b'
        question = re.sub(pattern, normalize_time, question)
    return question.strip()


def load_data(dataset_cfg):
    '''
        Raw FindingDory data structure:
            ## features: ['ep_id', 'video', 'question', 'answer', 'task_id', 'high_level_category', 'low_level_category', 'num_interactions'],
            ## num_rows: 5870
    '''
    dataset = load_dataset("yali30/findingdory", cache_dir="./data/.cache")["validation"]
    data_length = len(dataset)
    dict_qa_data = defaultdict(list)
    dict_video_name = defaultdict(list)

    eps = None
    if dataset_cfg["eval_ep_ids"] != "-1":
        eps = dataset_cfg["eval_ep_ids"].split(',')
    # how many examples to eval
    if dataset_cfg["eval_num"] > 0:
        data_length = min(data_length, dataset_cfg["eval_num"])
    elif dataset_cfg["eval_num"] == 0:
        raise Exception("eval_num cannot be 0")
    for i in range(data_length):
        instance = dataset[i]
        instance['question'] = question_special_formatting(instance['question'])
        if eps is not None and instance['ep_id'] not in eps:
            continue
        dict_qa_data[instance['ep_id']].append(instance)
        dict_video_name[instance['ep_id']] = Path(dataset_cfg["input_path"]) / instance['video']
    ids = list(dict_qa_data.keys())
    print(f"Loaded {len(ids)} videos with total {data_length} QA pairs.")
    qa_data, video_names = [], []
    for id in ids:
        qa_data.append(dict_qa_data[id])
        video_names.append(dict_video_name[id])

    return ids, qa_data, video_names
    

def main(dataset_cfg, 
         agent_cfg, 
         vlm_cfg, 
         embedder_cfg, 
         ep_id,
         qa_data,
         video_path,
         output_path,
         embedder=None):
    print("started!")

    name = os.path.join(output_path, (f'{ep_id}_output.json').replace("/", "_"))
    responses = []
    if os.path.exists(name):
        with open(name, 'r') as f:
            data = json.load(f)
            responses = data['responses']
        if len(responses) == len(qa_data):
            print(f"Output file {name} already completed! Skipping...")
            return

    if agent_cfg["type"] in ['raven', 'remembr']:
        cls = RAVENAgent if agent_cfg["type"] == 'raven' else ReMEmbRAgent
        prompts = dataset_cfg['vlm_prompt_folder'] if agent_cfg['type'] == 'raven' else dataset_cfg['prompt_folder']
        agent = cls(llm_type=vlm_cfg['full_name'], 
                    num_ctx=dataset_cfg['num_ctx'], 
                    num_gen_tokens=dataset_cfg['max_gen_tokens'], 
                    temperature=dataset_cfg['temperature'], 
                    debug=dataset_cfg['debug'], 
                    prompt_dir=prompts, 
                    max_tool_calls=dataset_cfg['max_tool_calls'])
    elif agent_cfg['type'] == 'vlm_only':
        agent = VLMNonAgent(llm_type=vlm_cfg['full_name'], 
                            num_ctx=dataset_cfg['num_ctx'], 
                            num_gen_tokens=dataset_cfg['max_gen_tokens'], 
                            temperature=dataset_cfg['temperature'], 
                            prompt_dir=dataset_cfg['vlm_prompt_folder'], 
                            max_message_length=dataset_cfg['non_agent_max_image_len'])
    elif agent_cfg['type'] == 'embedder_only':
        agent = EmbedderOnlyAgent(embedder=embedder)
    else:
        raise Exception("An unsupported agent!")


    memory, instance_captions = load_memory(agent_cfg, dataset_cfg, embedder_cfg, 
                                            Path(output_path).name,
                                            embedder, video_path)
    
    agent.set_memory(memory)

    if len(instance_captions) == 0: # ISSUE
        print("Length of Instance Captions is 0. It should not be")
        import pdb; pdb.set_trace()

    print("HISTORY LENGTH", len(instance_captions))
    print("COMPLETED QA:", len(responses), "OUT OF", len(qa_data))

    for i in tqdm.tqdm(range(len(responses), len(qa_data)), total=len(qa_data)-len(responses)):

        print(f"Evaluating {i} out of {len(qa_data)}")

        qa_instance = qa_data[i]
        question = qa_instance['question']
        
        out_dict = answer_question(agent, question)

        out_dict['question'] = qa_instance['question']
        out_dict['id'] = str(i)

        print("Question:", question)
        if 'response' in out_dict:
            print("Response:", out_dict['response'])

        responses.append(out_dict)
        out_json = {
            "version": 0.1,
            "responses": responses
        }
        # save the outputs
        os.makedirs(output_path, exist_ok=True)
        with open(name, 'w') as f:
            json.dump(out_json, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                        prog='FindingDory Runner',
                        description='Runs various robotic retrieval methods on the FindingDory dataset',)
    
    parser.add_argument("--dataset", type=str, default="findingdory", help="The dataset name (in cfgs/datasets) to evaluate on, findingdory for now")
    parser.add_argument("--agent", type=str, default=None, help="agent name (in cfgs.agents) in yaml, e.g., raven, vlm_only, remembr_sota, remembr_original, embedder_only")
    parser.add_argument("--embedder", type=str, default="none", help="name of the embedder (in cfgs.embedders) in yaml")
    parser.add_argument("--vlm", type=str, default="none", help="name of the embedder (in cfgs.vlms) in yaml")
    parser.add_argument("--postfix", type=str, default="_0", help="postfix of output file name")



    args = parser.parse_args()

    # dataset cfg
    dataset_cfg_path = os.path.join('cfgs', 'datasets', f'{args.dataset}.yaml')
    dataset_cfg = instantiate_from_yaml(cfg_path=dataset_cfg_path, cls=None)

    # agent cfg
    agent_cfg_path = os.path.join('cfgs', 'agents', f'{args.agent}.yaml')
    agent_cfg = instantiate_from_yaml(cfg_path=agent_cfg_path, cls=None)

    # vlm cfg
    vlm_cfg_path = os.path.join('cfgs', 'vlms', f'{args.vlm}.yaml')
    vlm_cfg = instantiate_from_yaml(cfg_path=vlm_cfg_path, cls=None)

    # embedder and cfg
    embedder = None
    embedder_cfg_path = os.path.join('cfgs', 'embedders', f'{args.embedder}.yaml')
    if agent_cfg['type'] == 'remembr' and agent_cfg['text_emb_model'] == 'original':
        embedder = HuggingFaceEmbeddings(model_name='mixedbread-ai/mxbai-embed-large-v1')
        embedder_cfg = instantiate_from_yaml(cfg_path=embedder_cfg_path, cls=None)
    elif agent_cfg['type'] in ['raven', 'remembr', 'embedder_only']:
        embedder, embedder_cfg = instantiate_from_yaml(cfg_path=embedder_cfg_path, cls=VLMEmbeddings)
    else: # vlm_only
        embedder, embedder_cfg = None, instantiate_from_yaml(cfg_path=embedder_cfg_path, cls=None)

    dataset_cfg["top_k"] = int_or_json(dataset_cfg["top_k"])

    print_cfg("Dataset CFG", dataset_cfg)
    print_cfg("Agent CFG", agent_cfg)
    print_cfg("VLM CFG", vlm_cfg)
    print_cfg("Embedder CFG", embedder_cfg)


    ids, qa_data, video_names = load_data(dataset_cfg)
    output_path = dataset_cfg["output_path"].format(method_and_models=args.agent + "_" + args.vlm + "_" + args.embedder + args.postfix)
    for ep_id, qa_dp, video_name in zip(ids, qa_data, video_names):
        
        print(f"Processing {video_name}...")

        main(dataset_cfg, 
             agent_cfg, 
             vlm_cfg, 
             embedder_cfg, 
             ep_id, 
             qa_dp, 
             video_name, 
             output_path, 
             embedder)