
# Evaluation on NaVQA

NaVQA is a navigation dataset built on CODa from ReMEmbR. A more detailed description of NaVQA data generation can be found in the [NaVQA setup](https://github.com/NVIDIA-AI-IOT/remembr/blob/main/eval.md).

Below we provide the simplest instructions for downloading, preprocessing, and evaluating RAVEN, ReMEmbR and VLM Only on NaVQA.

## Download and preprocess the NavQA dataset

First download the relevant subsets of the [CODa dataset](https://amrl.cs.utexas.edu/coda/), which consists of 22 sequences. 

We only need 7 of them which are `0, 3, 4, 6, 16, 21, 22`. These numbers will be referred to as sequence IDs. Each sequence ID has 30 questions associated with it.

> Because of the number of videos, be sure to have a large amount of storage. The processed dataset is ~335GB, but since the pre-processing phase also downloads LiDAR and other outputs, we would recommend having ~500GB extra storage.

Download the CODa devkit to some directory not inside RAVEN. Ideally place this in a larger HDD that has enough storage for all the data.

```bash
git clone https://github.com/ut-amrl/coda-devkit.git
cd coda-devkit && mkdir data
```

Then set two environment variables. Fill them with the appropriate paths. The `RAVEN_PATH` is the folder where the `scripts` folder is accessible.

You <span style="color:red">
NEED
</span> add these to your `~/.bashrc`

```bash
export CODA_ROOT_DIR=/path/to/coda-devkit/data
export RAVEN_PATH=/path/to/RAVEN
```
Then `source ~/.bashrc` to load the environment variables.

We have to first install their `coda` environment:

```bash
# while in the coda-devkit directory
conda env create -f environment.yml
```

Then run the following command which will preprocess the data in the appropriate format from the `RAVEN` directory:

```bash
conda activate coda
cd RAVEN
bash scripts/bash_scripts/preprocess_coda_all.sh
```

## Evaluate ReMEmbR on NaVQA

> :raising_hand: we provide our processed `captions` and `questions` folders under `RAVEN/data` folder, so running steps in [Caption](#caption) and [Form Questions](#form-questions) is not necessary to run RAVEN.

### Caption

Ensure the location of your preprocessed coda data is located in `/path/to/RAVEN/coda_data`.

Caption the NavQA dataset. Given the dataset, run the following command for each (for example, you need to caption the seq_id 0) with gpt-4o-mini:

```bash
# Using gpt-4o-mini:
python scripts/navqa_scripts/preprocess_captions.py \
    --seq_id 0 \
    --seconds_per_caption 3 \
    --openai_model gpt-4o-mini \
    --captioner_type openai \
    --out_path data/captions/0/captions \
    --overwrite \
    --api_delay 4.0
```

- `seq_id`: The sequence ID from the CODa dataset (of the 7 listed in the previous section)
- `seconds_per_caption`: The number of seconds of frames aggregated for generating a caption
- `openai_model`: The OpenAI model to use for captioning (e.g., `gpt-4o-mini`)
- `captioner_type`: The type of captioner to use (currently only `openai` is supported)
- `out_path`: The format of the captions must be: `data/captions/{seq_id}/captions`
- `api_delay`: Delay in seconds between API calls to avoid rate limiting

The captions for each frame should be put into a JSON file located in `data/captions/{seq_id}/captions`.

We provide an example to preprocess all captions as above in `scripts/bash_scripts/preprocess_captions_all.sh`.

### Form Questions

#### 1. Ensure `data/navqa/data.csv` exists

This folder contains the questions and answers that must be converted into the proper format. 

#### 2. Form the questions in the proper format

Run the following script, providing it a base captioner file that you ran previously. 

```bash
python scripts/question_scripts/form_question_jsons.py --caption_file captions_{captioner_name}_{seconds_per_caption}_secs
```

This is meant to also aggregate the "optimal" context required to answer the question based on the captioner and seconds per caption, so you must set `captioner_name` and `seconds_per_caption`. We recommend using a 3 seconds per caption value. For example, if `gpt-4o-mini` was used in the [Caption](#caption) step, you should run:

```bash
python scripts/question_scripts/form_question_jsons.py --caption_file captions_gpt-4o-mini_3_secs
```

After this step, a folder called `data/questions` should exist. Previously, NaVQA has a timestamp shift [issue](https://github.com/NVIDIA-AI-IOT/remembr/issues/21), we fix this in [form_question_jsons.py](../scripts/question_scripts/form_question_jsons.py) by fixing the time zone as UTC-5.

### Evaluation

We provide instruction to evaluate ReMEmbR per sequence ID. Text type questions are evaluated by querying VLM. Here we use Gemini-2.5-pro (requires `GOOGLE_API_KEY`). To use GPT-4o as the agent, you need `OPENAI_API_KEY`.

```bash
export GOOGLE_API_KEY=<your-google-api-key>
export OPENAI_API_KEY=<your-openai-api-key>
```

The CLI uses YAML configs under `cfgs/` for dataset, agent, embedder, and VLM settings:

```bash
python scripts/navqa_eval.py \
    --dataset {dataset_config} \
    --agent {agent_config} \
    --embedder {embedder_config} \
    --vlm {vlm_config} \
    --sequence_id {seq_id} \
    --postfix {postfix}
```

| Flag | Config directory | Examples |
|------|-----------------|----------|
| `--dataset` | `cfgs/datasets/` | `navqa_caption`, `navqa_vlm` |
| `--agent` | `cfgs/agents/` | `raven`, `remembr_original`, `remembr_sota`, `vlm_only`, `embedder_only` |
| `--embedder` | `cfgs/embedders/` | `siglip`, `clip`, `qqmm`, `mxbai`, `paligemma2`, `none` |
| `--vlm` | `cfgs/vlms/` | `gpt4o`, `gf25`, `gp3`, `gpt52` |

Because of how the code is written, if `seconds_per_caption` is changed, we would recommend re-running `questions/form_question_jsons.py`.

An example with gpt-4o and SigLIP embedder:

```bash
python scripts/navqa_eval.py \
    --dataset navqa_caption \
    --agent remembr_sota \
    --embedder siglip \
    --vlm gpt4o \
    --sequence_id 3 \
    --postfix _0
```

Another example with gemini-2.5-flash and mxbai embedder:

```bash
python scripts/navqa_eval.py \
    --dataset navqa_caption \
    --agent remembr_original \
    --embedder mxbai \
    --vlm gf25 \
    --sequence_id 0 \
    --postfix _0
```

### Accuracy Analyses

To analyze the accuracy and error, `navqa_acc.py` processes evaluation results from the `out/{vlm}_{embedder}` folder (generated by `navqa_eval.py`) and computes accuracy metrics grouped by base model name. All question types including text accuracy are always computed.

```bash
python scripts/navqa_acc.py \
    --mode caption \
    --out_root ./out/{vlm}_{embedder} \
    --qa_root ./data/questions \
    --result_root ./data/test_results
```

For example

```bash
python scripts/navqa_acc.py \
    --mode caption \
    --out_root ./out/gf25_mxbai \
    --qa_root ./data/questions \
    --result_root ./data/test_results
```

**Output:**

- Creates `data/test_results/{model_name}/acc_cat_summmary.json` for each base model
- Creates `data/test_results/{model_name}/acc_cat_summmary.txt` for each base model

**Note:** This script groups results by base model name (e.g., `gemini-2.5-flash`, `gpt-4o`) across different captioner types, rather than keeping them separate.

## Evaluate RAVEN on NaVQA

> :raising_hand: we provide our processed `convert` and `questions` folders under `RAVEN/data` folder, so running steps in [Convertion](#convertion) and [Form Questions](#form-questions-1) is not necessary to run RAVEN.

### Convertion

First, convert the video to frames, in `raven` folder, run

```bash
for seq_id in 0 3 4 6 16 21 22;
do
python scripts/navqa_scripts/preprocess_vlm_convert.py --seq_id $seq_id --overwrite
done
```

This will create the `data_vlm/convert/{seq_id}/convert_3_secs.json` files.

### Form Questions

Second, ensure `data_vlm/navqa/data.csv` exists. This folder contains the questions and answers that must be converted into the proper format. 

Then, in `RAVEN` folder, run

```bash
python scripts/navqa_scripts/preprocess_vlm_questions.py --convert_file convert_3_secs --overwrite
```

This will create the `data_vlm/questions/{seq_id}/human_qa.json` files. The questions are provided with context which is the optimal context required to answer the question based on the captioner and seconds per caption.

### Evaluation

Text type questions requires `GOOGLE_API_KEY`. To use GPT-4o as the agent, you need `OPENAI_API_KEY`.

```bash
export GOOGLE_API_KEY=<your-google-api-key>
export OPENAI_API_KEY=<your-openai-api-key>
```

Run the evaluation. Multimodal embedder would embed images directly from `.pkl` files, so please ensure that you have `coda_data/` folder under `RAVEN` folder. All question types (including text) are evaluated in a single run.

Here we provide an example using gpt-4o as the base llm and CLIP as the image embedder:

```bash
python scripts/navqa_eval.py \
    --dataset navqa_vlm \
    --agent raven \
    --embedder clip \
    --vlm gpt4o \
    --sequence_id 0 \
    --postfix _0 \
    --overwrite
```

We provide another example of using QQMM with Gemini-2.5-flash:

```bash
python scripts/navqa_eval.py \
    --dataset navqa_vlm \
    --agent raven \
    --embedder qqmm \
    --vlm gf25 \
    --sequence_id 0 \
    --postfix _0 \
    --overwrite
```

### Accuracy Analyses

Use `navqa_acc.py --mode vlm` to compute accuracy/errors grouped by `SHORT, MEDIUM, LONG` categories.

**Batch mode** — process all sequences in a result directory:

```bash
python scripts/navqa_acc.py \
    --mode vlm \
    --result_dir ./data_vlm/test_results/{model_name}_{embedder}
```

For example

```bash
python scripts/navqa_acc.py \
    --mode vlm \
    --result_dir ./data_vlm/test_results/gf25_qqmm
```

**Output:**
- `acc_cat_summmary.json`: Structured JSON summary with metrics per category
- `acc_cat_summmary.txt`: Human-readable text summary

**Metrics computed:**
- **Descriptive Accuracy**: Accuracy including binary and text questions correctness
- **Positional Error**: Mean ± std L2 norm error in meters
- **Temporal Error**: Mean ± std absolute difference in minutes (combining time and duration errors)

## Evaluate VLM Only on NaVQA

VLM only evaluation shares exactly same steps as RAVEN. The difference is that `--agent vlm_only` and `--embedder none` are used. For example:

```bash
python scripts/navqa_eval.py \
    --dataset navqa_vlm \
    --agent vlm_only \
    --embedder none \
    --vlm {vlm_config} \
    --sequence_id 0 \
    --postfix _0 \
    --overwrite
```

Where `{vlm_config}` is the VLM config name: `gpt4o`, `gf25`, `gp3`, etc.

For example

```bash
python scripts/navqa_eval.py \
    --dataset navqa_vlm \
    --agent vlm_only \
    --embedder none \
    --vlm gf25 \
    --sequence_id 0 \
    --postfix _0 \
    --overwrite
```

Steps of [convertion](#convertion), [Form Questions](#form-questions-1) has been implemented inside [Evaluate RAVEN on NaVQA](#evaluate-raven-on-navqa). No need to run again. Accuracy analyses of VLM only are the same as [Accuracy Analyses](#accuracy-analyses-1).

For example,

```bash
python scripts/navqa_acc.py \
    --mode vlm \
    --result_dir ./data_vlm/test_results/gf25_none
```