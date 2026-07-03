
# Evaluation on RAVEN-QA
Activate the `raven` conda environment. Data are released [HERE](https://huggingface.co/datasets/zzcnewly/RAVEN_QA), please place them under `./data` as follows.

```
data/
├── habitat_sim/
├── real_world/
├── irs_hard/
└── irs/
```

## Data Convertion and (Optional) Captioning
If use ReMEmbR, we will first caption data. Otherwise, we can skip the next 1 and 2 steps.

1. (optional) parser the image dataset
```bash
python remembr_data_parser.py -i ./data/real_world -o ./data/real_world_output/pic_parse --pic_parser
```

2. (optional) caption the image dataset.
```bash
export OPENAI_API_KEY = $your_api_key

python remembr_data_captioner.py --input_dir ./data/real_world_output/pic_parse --output_dir ./data/real_world_output/pic_caption --seconds_per_caption 1 --num_video_frames 1 --api_delay 4.0 --captioner_type openai --openai_model gpt-4o-mini --query "You are wandering around an office. Please describe in detail what you see in the image. Specifically focus on the objects, environmental features, events/activities, people, and other interesting details. Think step by step about these details and be very specific. Thank you!" --from_images # --overwrite
```

3. do a convertion to adapt to a structured format

```bash
# if skipped captioning (RAVEN)
python raven_qa_format_convert.py \
  --data_root  data/real_world \
  --out_dir  data/real_world_converted

# if captioned data (ReMEmbR)
python raven_qa_format_convert.py \
  --data_root  data/real_world \
  --out_dir  data/real_world_converted \
  --caption_path data/real_world_output/pic_caption
```

4. Run the methods as below.


## Running on Closed APIs

> Note: Please make sure to use `timeout`. With very low temperature, or if a VLM fails to follow instructions, the agent may retry multiple times and consume excessive API credits.

One can specify any combinations of the experimental components, as those listed under `./cfgs`. In the arguments, just fill in the `.yaml` file names. For example, if we are going to test different agents with `qqmm` as an embedder and `gemini-3-pro-preview` as the base VLM, the commands look like:


```bash
export GOOGLE_API_KEY=$your_api_key
## For OpenAI models
# export OPENAI_API_KEY=$your_api_key
## For seed models
# export SEED_API_KEY=$your_api_key

# - RAVEN
timeout 5400 python raven_qa_run.py --dataset real_world --agent raven --embedder qqmm --vlm gp3
# or using seed and gemini-2.5-flash 
timeout 5400 python raven_qa_run.py --dataset real_world --agent raven --embedder seed --vlm gf25

# - ReMEmbR
# qqmm as an embedder ('sota' means using the state of the art embedders)
timeout 5400 python raven_qa_run.py --dataset real_world --agent remembr_sota --embedder qqmm --vlm gp3

# mxbai as an embedder ('original' means using mxbai as the embedder)
timeout 5400 python raven_qa_run.py --dataset real_world --agent remembr_original --embedder mxbai --vlm gp3

# - VLM Only
timeout 5400 python raven_qa_run.py --dataset real_world --agent vlm_only --vlm gp3

# - Embedder Only
python raven_qa_run.py --dataset real_world --agent embedder_only --embedder qqmm
```

### Calculate the evaluation statistics
The output results can now be checked under `./output`. Get the accuracy (incl. overall, per-category, per-query, and per-type accuracies) by running:

```bash
# - RAVEN
python raven_qa_eval.py  --pred_dir output/real_world/raven_gp3_qqmm_0 --gt_dir data/real_world_converted

## - ReMEmbR
## qqmm as an embedder
# --pred_dir output/real_world/remembr_sota_gp3_qqmm_0/
## - VLM Only
# --pred_dir output/real_world/vlm_only_gf25_none_0
## - Embedder Only
# --pred_dir output/real_world/embedder_only_none_qqmm_0
```

## Running on Open Source VLMs

Take Gemma3 27B + QQMM as an example here.

Before start, 

1. Better to clone a new virtual environment; make sure `transformers==4.44.2`
2. Pip install neccessary packages, e.g., ollama
3. Launch your Ollama server via the following commands.

Launch two shells. One runs 
```bash
~/.cache/ollama/bin/ollama serve
```

or, keep everything in one shell

```bash
set -euo pipefail
~/.cache/ollama/bin/ollama serve > ollama.log 2>&1 & SERV_PID=$!

# Wait until server is ready
for i in {1..120}; do
if curl -sSf http://127.0.0.1:11434/api/tags >/dev/null; then
    break
fi
sleep 1
done
```

Then pull the model from remote, 

```bash
~/.cache/ollama/bin/ollama pull gemma3:27b
# or
~/.cache/ollama/bin/ollama pull qwen2.5vl:32b-q4_K_M
# or
~/.cache/ollama/bin/ollama pull qwen3-vl:32b-instruct-q4_K_M
# or ...
```


Take Qwen2.5-VL 32B (quantized) and Gemma3 27B (Ollama) as examples: Make sure you can access a machine with at least 40GB compute for inference. If you use QQMM (~20GB memory use under fp32) as the embedder, the memory limit should be expected larger (60 GB). 

> Note: If you encounter issues when running the following command, you may need to downscale the frames to approximately 360 px in height or width. The resized frames are also provided in the dataset. That is why we use a new dataset named `real_world_open_models`.

Run open agents, like, 

```bash 
# example 1: raven + gemma3-27b + qqmm
python raven_qa_run.py --dataset real_world_open_models --agent raven --embedder qqmm --vlm gemma3-27b

# example 2: remembr + qwen3vl-32b + mxbai
python raven_qa_run.py --dataset real_world_open_models --agent remembr_original --embedder mxbai --vlm qwen3vl-32b
```

> Note: For VLM-only settings, please constrain the maximum number of images fed to the VLM in a single call. Also limit the maximum number of tool calls and top-k number for RAVEN to avoid exceeding the context window; otherwise, inference may get stuck. Relevant hyperparameters are specified in `./cfgs`.

Properly close the server,

```bash
kill $SERV_PID || true
wait $SERV_PID || true
```

## Customized model recipe

If people want to add and try out their specified models (VLMs or embedders),

1. make sure the model configuration is under `./cfgs` where its config can inherit from a `base.yaml` class. VLMs and embedder's configurations are under `./cfgs/vlms` and `./cfgs/embedders` folders, repectively.

2. If there does not exist your model. Then, in the codespace, one may manually add the implementation to adapt for the customized embedding models and VLMs. Their main code files are under `./raven/embedder` and `./raven/agents`. Also one may expect modifying the main running file. Similarly, one can also add new agents aside from existing `raven`, `remembr`, `vlm_only`, and `embedder_only`.
