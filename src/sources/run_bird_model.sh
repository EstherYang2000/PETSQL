    # {"name": "llamaapi_3.3", "model": "llamaapi", "version": "3.3"},
#     {"name": "gpt-4", "model": "gptapi", "version": "gpt-4"},
    # {"name": "gpt-4o", "model": "gptapi", "version": "chatgpt-4o-latest"},
#     {"name": "gpt-4.1-2025-04-14", "model": "gptapi", "version": "gpt-4.1-2025-04-14"},
    # {"name": "o1-preview", "model": "gptapi", "version": "o1-preview"},
    # {"name": "o1", "model": "gptapi", "version": "o1"},
    # {"name": "gpt-4.5", "model": "gptapi", "version": "gpt-4.5-preview"},
    #{"name": "o3-mini", "model": "gptapi", "version": "o3-mini"},
    # {"name": "qwen_api_32b-instruct-fp16", "model": "qwen_api", "version": "32b-instruct-fp16"},
    # {"name": "mistralapi_small_24b", "model": "mistralapi", "version": "small_24b"},
    # {"name": "qwen_api_2_5_72b", "model": "qwen_api", "version": "2_5_72b"},
    # {"name": "gemini", "model": "googlegeminiapi", "version": "gemini-2.5-pro-exp-03-25"},
    #{'name': 'grok3', 'model': 'grokapi', 'version': 'grok-3-beta'},


python src/sources/sql_gen/prompt_gen.py \
  --data_type bird \
  --dataset ppl_dev_bird.json 
  --dataset_type dev \
  --n 1534 \
  --kshot 9 \
  --select_type Euclidean_mask

python src/sources/sql_gen/call_llm.py \
    --path_generate bird/process/bird/PPL_DEV_BIRD.JSON-9_SHOT_Euclidean_mask_1534_base \
    --model grokapi \
    --model_version grok-3-beta \
    --out_file grokapi_grok-3-beta.json \
    --data_type bird \
    --dataset_type dev \
    --start_num_prompts 0 \
    --end_num_prompts 1534 \
    --batch_size 1 \
    --call_mode append \
    --n_samples 1

python src/sources/sql_gen/call_llm.py \
    --path_generate bird/process/bird/PPL_DEV_BIRD.JSON-9_SHOT_Euclidean_mask_1534 \
    --model llamaapi \
    --model_version 3.3 \
    --out_file llamaapi_3.3.json \
    --data_type bird \
    --dataset_type dev \
    --start_num_prompts 0 \
    --end_num_prompts 1534 \
    --batch_size 1 \
    --call_mode append \
    --n_samples 1

python src/sources/post_process.py \
    --file bird/process/bird/PPL_DEV_BIRD.JSON-9_SHOT_Euclidean_mask_1534_base/grokapi_grok-3-beta.json \
    --output bird/process/bird/PPL_DEV_BIRD.JSON-9_SHOT_Euclidean_mask_1534_base/grokapi_grok-3-beta_output.json \
    --llm sensechat

python src/sources/bird_evaluation/process_sql.py \
    --file bird/process/bird/PPL_DEV_BIRD.JSON-9_SHOT_Euclidean_mask_1534_base/grokapi_grok-3-beta_output.json \
    --output bird/process/bird/PPL_DEV_BIRD.JSON-9_SHOT_Euclidean_mask_1534_base/grokapi_grok-3-beta_output_eval.json

python src/sources/extract_sql_output.py \
    --file bird/process/bird/PPL_DEV_BIRD.JSON-9_SHOT_Euclidean_mask_1534_base/grokapi_grok-3-beta_output.json \
    --output bird/process/bird/PPL_DEV_BIRD.JSON-9_SHOT_Euclidean_mask_1534_base/grokapi_grok-3-beta_output.txt



python src/sources/bird_evaluation/evaluation.py \
    --predicted_sql_path bird/process/bird/PPL_DEV_BIRD.JSON-9_SHOT_Euclidean_mask_1534_base/grokapi_grok-3-beta_output_eval.json \
    --ground_truth_path bird/bird/dev.sql \
    --db_root_path bird/bird/database/ \
    --num_cpus 4 \
    --meta_time_out 30 \
    --diff_json_path bird/bird/dev.json \
    --sql_dialect SQLite \
    --output_log_path bird/process/bird/PPL_DEV_BIRD.JSON-9_SHOT_Euclidean_mask_1534_base/evaluation_log.txt


python src/sources/schemalink.py --output ppl_dev_add_sl_bird.json --file bird/process/vote/PPL_DEV_BIRD.JSON-9_SHOT_Euclidean_mask_1534_rf_naive/final_sql_1.txt

python src/sources/sql_gen/prompt_gen.py \
  --data_type bird \
  --dataset ppl_dev_add_sl_bird.json \
  --dataset_type dev \
  --n 1534 \
  --kshot 9 \
  --select_type Euclidean_mask \
  --sl


python src/sources/sql_gen/call_llm.py \
    --path_generate bird/process/bird/PPL_DEV_ADD_SL_BIRD.JSON-9_SHOT_Euclidean_mask_1534_base \
    --model googlegeminiapi \
    --model_version gemini-2.5-pro-exp-03-25 \
    --out_file googlegeminiapi_gemini-2.5-pro-exp-03-25.json \
    --data_type bird \
    --dataset_type dev \
    --start_num_prompts 998 \
    --end_num_prompts 1534 \
    --batch_size 1 \
    --call_mode append \
    --n_samples 1


python src/sources/post_process.py \
    --file bird/process/bird/PPL_DEV_ADD_SL_BIRD.JSON-9_SHOT_Euclidean_mask_1534_base/googlegeminiapi_gemini-2.5-pro-exp-03-25.json \
    --output bird/process/bird/PPL_DEV_ADD_SL_BIRD.JSON-9_SHOT_Euclidean_mask_1534_base/googlegeminiapi_gemini-2.5-pro-exp-03-25_output.json \
    --llm sensechat

python src/sources/bird_evaluation/process_sql.py \
    --file bird/process/bird/PPL_DEV_ADD_SL_BIRD.JSON-9_SHOT_Euclidean_mask_1534_base/googlegeminiapi_gemini-2.5-pro-exp-03-25_output.json \
    --output bird/process/bird/PPL_DEV_ADD_SL_BIRD.JSON-9_SHOT_Euclidean_mask_1534_base/googlegeminiapi_gemini-2.5-pro-exp-03-25_output_eval.json

python src/sources/extract_sql_output.py \
    --file bird/process/bird/PPL_DEV_ADD_SL_BIRD.JSON-9_SHOT_Euclidean_mask_1534_base/googlegeminiapi_gemini-2.5-pro-exp-03-25_output.json \
    --output bird/process/bird/PPL_DEV_ADD_SL_BIRD.JSON-9_SHOT_Euclidean_mask_1534_base/googlegeminiapi_gemini-2.5-pro-exp-03-25_output.txt



python src/sources/bird_evaluation/evaluation.py \
    --predicted_sql_path bird/process/bird/PPL_DEV_ADD_SL_BIRD.JSON-9_SHOT_Euclidean_mask_1534_base/googlegeminiapi_gemini-2.5-pro-exp-03-25_output_eval.json \
    --ground_truth_path bird/bird/dev.sql \
    --db_root_path bird/bird/database/ \
    --num_cpus 4 \
    --meta_time_out 30 \
    --diff_json_path bird/bird/dev.json \
    --sql_dialect SQLite \
    --output_log_path bird/process/bird/PPL_DEV_ADD_SL_BIRD.JSON-9_SHOT_Euclidean_mask_1534_base/evaluation_log_googlegeminiapi_gemini-2.5-pro-exp-03-25.txt
