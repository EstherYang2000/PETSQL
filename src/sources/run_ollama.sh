export PYTHONPATH=src/sources
# echo "## Start generating sqls by codellamaapi ..."
# python src/sources/sql_gen/call_llm.py \
#     --path_generate /home/yyj/Desktop/yyj/thesis/code/PETSQL/data/process/PPL_DEV.JSON-3_SHOT_Euclidean_mask_1034 \
#     --model codellamaapi \
#     --model_version 34b-instruct \
#     --out_file codellama_34b-instruct_api.txt \
#     --num_prompts 1034 \
#     --batch_size 1

python src/sources/sql_gen/call_llm.py \
    --path_generate data/process/PPL_DEV.JSON-3_SHOT_Euclidean_mask_1034 \
    --model codellamaapi \
    --model_version 70b \
    --out_file codellama_70b_api.txt \
    --num_prompts 1034 \
    --batch_size 1
python src/sources/sql_gen/call_llm.py \
    --path_generate data/process/PPL_DEV.JSON-9_SHOT_Euclidean_mask_1034 \
    --model codellamaapi \
    --model_version 70b \
    --out_file codellama_70b_api.txt \
    --num_prompts 1034 \
    --batch_size 1
# echo "## Start generating sqls by deepseekapi ..."
# sleep 1
# python src/sources/sql_gen/call_llm.py \
#     --path_generate /home/yyj/Desktop/yyj/thesis/code/PETSQL/data/process/PPL_DEV.JSON-3_SHOT_Euclidean_mask_1034 \
#     --model deepseekapi \
#     --model_version r1_70b \
#     --out_file deepseek_r1_70b_api.txt \
#     --num_prompts 1034 \
#     --batch_size 1
python src/sources/sql_gen/call_llm.py \
    --path_generate data/process/PPL_DEV.JSON-9_SHOT_Euclidean_mask_1034 \
    --model deepseekapi \
    --model_version r1_distill_llama_70b \
    --out_file deepseek_r1_distill_llama_70b_api.txt \
    --num_prompts 1034 \
    --batch_size 1

python src/sources/sql_gen/call_llm.py \
    --path_generate data/process/PPL_DEV.JSON-9_SHOT_Euclidean_mask_1034 \
    --model deepseekapi \
    --model_version coder-v2:16b \
    --out_file deepseek_coder-v2:16b_api.txt \
    --start_num_prompts 0 \
    --num_prompts 10 \
    --batch_size 1 \
    --call_mode append
python src/sources/sql_gen/call_llm.py \
    --path_generate data/process/PPL_DEV_ADD_SL.JSON-9_SHOT_Euclidean_mask_1034_sl \
    --model qwen_api \
    --model_version 32b-instruct-fp16 \
    --out_file qwen_api_32b-instruct-fp16_api.txt \
    --start_num_prompts 0 \
    --num_prompts 1034 \
    --batch_size 1 \
    --call_mode append
python src/sources/sql_gen/call_llm.py \
    --path_generate data/process/PPL_DEV.JSON-9_SHOT_Euclidean_mask_1034 \
    --model deepseekapi \
    --model_version r1_distill_llama_70b_specdec \
    --out_file deepseek_r1_distill_llama_70b_specdec_api.txt \
    --num_prompts 1034 \
    --batch_size 1

# python src/sources/sql_gen/call_llm.py \
#     --path_generate data/process/PPL_DEV.JSON-9_SHOT_Euclidean_mask_1034 \
#     --model deepseekapi \
#     --model_version llm_67b \
#     --out_file deepseek_coder_llm_67b_api.txt \
#     --num_prompts 1034 \
#     --batch_size 1
# echo "## Start generating sqls by llama3.3:latest ..."
# sleep 1
python src/sources/sql_gen/call_llm.py \
    --path_generate /home/yyj/Desktop/yyj/thesis/code/PETSQL/data/process/PPL_DEV.JSON-9_SHOT_Euclidean_mask_1034 \
    --model llamaapi \
    --model_version 3.3 \
    --out_file llama3.3:latest_api.txt \
    --start_num_prompts 0 \
    --num_prompts 1034 \
    --batch_size 1
python src/sources/sql_gen/call_llm.py \
    --path_generate data/process/PPL_DEV.JSON-9_SHOT_Euclidean_mask_1034 \
    --model llamaapi \
    --model_version 3.3_70b_specdec \
    --out_file llama3.3_70b_specdec_api.txt \
    --start_num_prompts 0 \
    --num_prompts 100 \
    --batch_size 1
# echo "## Start generating sqls by phind-codellama-api ..."
# sleep 1
# python src/sources/sql_gen/call_llm.py \
#     --path_generate /home/yyj/Desktop/yyj/thesis/code/PETSQL/data/process/PPL_DEV.JSON-3_SHOT_Euclidean_mask_1034 \
#     --model phind-codellamaapi \
#     --out_file phind-codellama_api.txt \
#     --num_prompts 1034 \
#     --batch_size 1
# echo "## Start generating sqls by qwen2.5-coderaapi ..."
# sleep 1
# python src/sources/sql_gen/call_llm.py \
#     --path_generate /home/yyj/Desktop/yyj/thesis/code/PETSQL/data/process/PPL_DEV_ADD_SL.JSON-9_SHOT_Euclidean_mask_1034 \
#     --model qwen2.5-coderaapi \
#     --out_file qwen2.5-coder_32b_instruct_api.txt \
#     --num_prompts 1034 \
#     --batch_size 1

# python src/sources/sql_gen/call_llm.py \
#     --path_generate data/process/PPL_DEV.JSON-9_SHOT_Euclidean_mask_1034 \
#     --model mistralapi \
#     --model_version small_24b \
#     --out_file mistralapi_small_24b_api.txt \
#     --start_num_prompts 0 \
#     --num_prompts 10 \
#     --batch_size 1 \
#     --call_mode append

# ollama run phind-codellama
# ollama run qwen2.5-coder:32b-instruct-fp16