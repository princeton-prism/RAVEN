import os, sys
import glob
from pathlib import Path
from dataclasses import asdict
import json, time
import argparse
import traceback 

# load this directory
sys.path.append(sys.path[0] + '/..')
import numpy as np
import tqdm
from langchain_huggingface import HuggingFaceEmbeddings
from raven.agents.raven_agent import RAVENAgent
from raven.agents.remembr_agent import ReMEmbRAgent
from raven.agents.vlm_non_agent import VLMNonAgent
from raven.memory.memory import VLMMemoryItem
from raven.agents.embedder_only_agent import EmbedderOnlyAgent 
from raven.utils.util import predefined_filename, instantiate_from_yaml, print_cfg
from raven.embedder.embedders import VLMEmbeddings
from raven.memory.memory_factory import MemoryFactory
from raven.memory.video_memory import VideoMemory
from raven.memory.memory import MemoryItem
from raven.agents.agent import AgentOutput

def answer_question(model, question):

    print(f'Question: {question}')

    parsed = None
    while True:
        try:

            start_time = time.time()
            response = model.query(question) # model == agent
            end_time = time.time()

            elapsed = end_time - start_time

            # ##### embedder_only specific parsing start #############
            if isinstance(response, int) or response is None:
                if response is None:
                    response = AgentOutput(None, None, None, None, None, None, None)
                else:
                    time_str = f"frame_{response:03d}.jpg"
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


def load_memory(agent_cfg, 
                dataset_cfg, 
                embedder_cfg, 
                embedder:VLMEmbeddings|HuggingFaceEmbeddings|None, 
                start_time, 
                caption_file, 
                unit='microseconds'):
    if agent_cfg['type'] in ['raven', 'embedder_only', 'remembr']:
        memory = MemoryFactory.create_memory(
            backend=dataset_cfg['memory_backend'],
            db_collection_name=agent_cfg['type'],
            embedder=embedder,
            storage_path=os.environ.get('RAVEN_MEMORY_STORAGE', dataset_cfg.get('memory_storage_path', './output/memory_storage')),
            use_vlm_embedding=agent_cfg['type'] != 'remembr',
            time_offset=start_time if unit=='seconds' else start_time/1e6,
            dim=embedder_cfg['emb_dim'],
            retriever_k=dataset_cfg['top_k'],
            respond_with_score=agent_cfg['add_score_info'],
        )
    elif agent_cfg['type'] == 'vlm_only':
        memory = VideoMemory(start_time=start_time if unit=='seconds' else start_time/1e6)
    else:
        raise Exception("Unsupported agent type for loading memory!")
    
    memory.reset()
    with open(caption_file, 'r') as f:
        out = json.load(f)


    outputs = []

    start_idx = 0
    end_idx = len(out) - 1
    
    dict_fn_to_emb = {}

    # for api use, we do caching of image embeddings to save time and api credits
    if embedder_cfg['backend'] == 'ol' and out:
        cache_dir = os.path.join(Path(out[0]["image_file_path"]).parent.parent, ".cache")
        os.makedirs(cache_dir, exist_ok=True)
        cache_emb = os.path.join(cache_dir, f"{embedder_cfg['online_model_nickname']}_image_embeds.npy")
        cache_list = os.path.join(cache_dir, f"{embedder_cfg['online_model_nickname']}_image_filenames.json")
        if os.path.exists(cache_emb) and os.path.exists(cache_list):
            with open(cache_list, "r", encoding="utf-8") as f:
                cached_fns = json.load(f)
            arr = np.load(cache_emb)
            for i, fn in enumerate(cached_fns):
                dict_fn_to_emb[fn] = arr[i]
        else:
            to_embed = [e for e in out if e.get('image_file_path') is not None]
            for entity in tqdm.tqdm(to_embed, desc="Caching image embeddings", unit="img"):
                dict_fn_to_emb[Path(entity['image_file_path']).name] = embedder.embed_query("[IMG]" + entity['image_file_path'])
            with open(cache_list, "w", encoding="utf-8") as f:
                json.dump(list(dict_fn_to_emb.keys()), f)
            np.save(cache_emb, np.array(list(dict_fn_to_emb.values())))
        
    for i in range(start_idx, end_idx+1):

        item = out[i]
        if "position" not in item or "unknown" in item['position']:
            item['position'] = [0., 0., 0.]
        item['position'] = [round(x, 2) for x in item['position']]
        entity = {
            'position': item['position'], # we don't have position info in the captions right now
            'time': item['time'] if unit == "seconds" else item['time'] / 1e6,
            'caption': item['caption'],
            'theta': 3.14, # we don't have theta info in the captions right now
            'image_file_path': item['image_file_path'] if 'image_file_path' in item else None
        }

        outputs.append(entity)

        entity = VLMMemoryItem.from_dict(entity) if agent_cfg['type'] != 'remembr' else MemoryItem.from_dict(entity)

        if agent_cfg['type'] in ['raven', 'embedder_only']:
            memory.insert(entity, vlm_embedding=dict_fn_to_emb[Path(item['image_file_path']).name] if dict_fn_to_emb else None) # will make up embeddings inside
        elif agent_cfg['type'] == 'vlm_only':
            memory.insert(entity)
        else: # remembr
            memory.insert(entity, text_embedding=item['text_embedding'] if agent_cfg['text_emb_model'] == 'original' else None)
        
    return memory, outputs

def main(embedder, 
         qa_path:str, 
         caption_path:str, 
         dataset_cfg:dict, 
         agent_cfg:dict, 
         embedder_cfg:dict, 
         vlm_cfg:dict, 
         output_filename:str):
    print(dataset_cfg)
    if agent_cfg['type'] in ['raven', 'remembr']:
        cls = RAVENAgent if agent_cfg['type'] == 'raven' else ReMEmbRAgent
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

    # qa questions / queries
    # frames images == memories items
    data = json.load(open(qa_path, 'r'))
    start_time = data['start_time']

    data = data['data']

    responses = []
    
    memory, instance_captions = load_memory(agent_cfg, dataset_cfg, embedder_cfg, 
                                            embedder, start_time, caption_path)
    
    agent.set_memory(memory)

    assert len(instance_captions) > 0, "Length of Instance Captions is 0. It should not be"
    print("HISTORY LENGTH", len(instance_captions))

    for i in tqdm.tqdm(range(0, len(data)), total=len(data)):

        print(f"Evaluating {i} out of {len(data)}")

        qa_instance = data[i]
        question = qa_instance['question']
        qid = qa_instance['id']

        out_dict = answer_question(agent, question)

        out_dict['question'] = qa_instance['question']
        out_dict['id'] = qid

        print("Question:", question)
        if 'response' in out_dict:
            print("Response:", out_dict['response'])

        responses.append(out_dict)


    # save all_questions into json
    out_json = {
        "version": 0.1,
        "responses": responses
    }

    # save the outputs
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)

    with open(output_filename, 'w') as f:
        json.dump(out_json, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                        prog='RAVEN-QA Runner',
                        description='Runs various robotic retrieval methods on the RAVEN-QA dataset',)
    
    # choice 1: using input folder flag: doing video batch 
    parser.add_argument("--dataset", type=str, default=None, help="The dataset name (in cfgs/datasets) to evaluate on, e.g., real_world, simulation, simple, hard")
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

    print_cfg("Dataset CFG", dataset_cfg)
    print_cfg("Agent CFG", agent_cfg)
    print_cfg("VLM CFG", vlm_cfg)
    print_cfg("Embedder CFG", embedder_cfg)

    qa_files = sorted(glob.glob(os.path.join(dataset_cfg["input_path"], '*_qa.json')))
    caption_files = sorted(glob.glob(os.path.join(dataset_cfg["input_path"], '*_frames.json')))
    assert len(qa_files) == len(caption_files), "Warning: The number of QA files and caption files do not match!"

    for qa_path, caption_path in zip(qa_files, caption_files):
        assert qa_path.split('/')[-1].split('_qa.json')[0] == caption_path.split('/')[-1].split('_frames.json')[0], \
            f"QA file {qa_path} and caption file {caption_path} do not match!"

        output_filename = os.path.join(dataset_cfg["output_path"].format(method_and_models=args.agent + "_" + args.vlm + "_" + args.embedder + args.postfix), 
                            predefined_filename(caption_path, qa_path))

        if os.path.exists(output_filename):
            print(f"Output file {output_filename} already exists! Skipping...")
            continue
        
        print(f"Processing {qa_path} and {caption_path}")          

        main(embedder, qa_path, caption_path, dataset_cfg, agent_cfg, embedder_cfg, vlm_cfg, output_filename)