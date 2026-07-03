
# Evaluation on FindingDory

FindingDory (Data: https://huggingface.co/datasets/yali30/findingdory) is a long-term loco-manipulation dataset and object retrieval benchmark built on Habitat.

Below we provide an example of evaluating high-level retrieval success rates on long-horizon loco-manipulation videos.
Please refer to the FindingDory paper ([arXiv:2506.15635](https://arxiv.org/abs/2506.15635)) for full dataset details.

This evaluation is training-free. Since no official test split is provided for high-level evaluation, we report results on the validation set.

## Steps

> Note: If you plan to use commercial APIs to run the full FindingDory validation set (approximately 6,000 queries), please ensure that your account has sufficient API credits !

1. First download video from https://huggingface.co/datasets/yali30/findingdory/blob/main/videos.zip and place `videos/` folder under `./data/findingdory/`, as follows.

    ```
    data/
    ├── findingdory/
    │   └── videos/
    │       ├── train/
    │       │   └── ...
    │       └── val/
    │           └── ...
    ```

2. Activate conda environment, `conda activate raven`

3. Run evaluation. The evaluation pipeline is resumable and can continue from the last completed query sample, under the same command. 

    For ReMEmbR experiment, considering the cost and efficiency over captioning a large number of frames (~180,000 requests), we are going to use GPT-4o-mini as the captioner and caption 1 image for every 5, which is a sweet spot between the information retention and efficiency. 

    You may expect that one experiment can run for up to **2 weeks** if in a single thread. For ReMEmbR which has an additional captioning phase, the process may take longer.




```bash

export GOOGLE_API_KEY=$your_api_key

# Here we only test 5 queries (to change the evaluation range, modify `eval_num` in the `findingdory_test` yaml file)

# - raven
timeout 5400 python findingdory_run.py --dataset findingdory_test --agent raven  --vlm gf25 --embedder qqmm

# - remembr (captioning is done here)
export OPENAI_API_KEY=$your_api_key  # for captioning
timeout 5400 python findingdory_run.py --dataset findingdory_test --agent remembr_original  --vlm gf25 --embedder mxbai 
timeout 5400 python findingdory_run.py --dataset findingdory_test --agent remembr_sota  --vlm gf25 --embedder qqmm 

# - vlm only
timeout 5400 python findingdory_run.py --dataset findingdory_test --agent vlm_only  --vlm gf25

# - embedder only
python findingdory_run.py --dataset findingdory_test --agent embedder_only  --embedder qqmm 
```


4. Calculate numbers of success rates

```bash
## raven
python findingdory_eval.py --out_dir output/fd/raven_gf25_qqmm_0
## remembr original
# python findingdory_eval.py --out_dir output/fd/remembr_original_gf25_mxbai_0
## remembr sota
# python findingdory_eval.py --out_dir output/fd/remembr_sota_gf25_qqmm_0
## vlm only
# python findingdory_eval.py --out_dir output/fd/vlm_only_gf25_none_0
## embedder only
# python findingdory_eval.py --out_dir output/fd/embedder_only_none_qqmm_0  
```

If tested on the extra-long subset (from `ep_91` to `ep_100`), only evaluate on those episodes by running:
```bash
## raven ep_91-ep_100 (a long-range subset of findingdory)
python findingdory_eval.py --out_dir output/fd/raven_gf25_qqmm_0  --eval_ep_ids ep_91,ep_92,ep_93,ep_94,ep_95,ep_96,ep_97,ep_98,ep_99,ep_100
```

If only check the first `eval_num` queries, add a `--eval_num $num` flag.
