for i in 0 3 4 6 16 21 22;
do
    python scripts/navqa_scripts/preprocess_captions.py \
    --seq_id $i \
    --seconds_per_caption 3 \
    --openai_model gpt-4o-mini \
    --captioner_type openai \
    --out_path data/captions/$i/captions \
    --overwrite \
    --api_delay 4.0
done




