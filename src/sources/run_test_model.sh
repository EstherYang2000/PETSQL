python src/sources/sql_gen/call_llm.py \
    --path_generate data/process/PPL_DEV.JSON-3_SHOT_Euclidean_mask_1034 \
    --model deepseekapi \
    --model_version r1_distill_llama_70b \
    --out_file deepseek_r1_distill_llama_70b_api.txt \
    --start_num_prompts 0 \
    --num_prompts 500 \
    --batch_size 1 \
    --call_mode append

python src/sources/sql_gen/call_llm.py \
    --path_generate data/process/PPL_DEV.JSON-3_SHOT_Euclidean_mask_1034 \
    --model deepseekapi \
    --model_version r1_distill_llama_70b \
    --out_file deepseek_r1_distill_llama_70b_api.txt \
    --start_num_prompts 500 \
    --num_prompts 1034 \
    --batch_size 1 \
    --call_mode append