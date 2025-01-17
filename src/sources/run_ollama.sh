echo "## Start generating sqls by codellamaapi ..."
python src/sources/sql_gen/call_llm.py \
    --path_generate /home/yyj/Desktop/yyj/thesis/code/PETSQL/data/process/PPL_DEV.JSON-9_SHOT_Euclidean_mask_1034 \
    --model codellamaapi \
    --model_version 34b-instruct \
    --out_file codellama_34b-instruct_api.txt \
    --num_prompts 1034
echo "## Start generating sqls by deepseekapi ..."
sleep 1
python src/sources/sql_gen/call_llm.py \
    --path_generate /home/yyj/Desktop/yyj/thesis/code/PETSQL/data/process/PPL_DEV.JSON-9_SHOT_Euclidean_mask_1034 \
    --model deepseekapi \
    --model_version v2-16b \
    --out_file deepseek_v2-16b_api.txt \
    --num_prompts 1034

echo "## Start generating sqls by deepseekapi ..."
sleep 1
python src/sources/sql_gen/call_llm.py \
    --path_generate /home/yyj/Desktop/yyj/thesis/code/PETSQL/data/process/PPL_DEV.JSON-9_SHOT_Euclidean_mask_1034 \
    --model llamaapi \
    --out_file llama_3-70b_api.txt \
    --num_prompts 1034 \
    --batch_size 1
echo "## Start generating sqls by phind-codellama-api ..."
sleep 1
python src/sources/sql_gen/call_llm.py \
    --path_generate /home/yyj/Desktop/yyj/thesis/code/PETSQL/data/process/PPL_DEV.JSON-9_SHOT_Euclidean_mask_1034 \
    --model phind-codellamaapi \
    --out_file phind-codellam_api.txt \
    --num_prompts 1034 \
    --batch_size 1
echo "## Start generating sqls by qwen2.5-coderaapi ..."
sleep 1
python src/sources/sql_gen/call_llm.py \
    --path_generate /home/yyj/Desktop/yyj/thesis/code/PETSQL/data/process/PPL_DEV.JSON-9_SHOT_Euclidean_mask_1034 \
    --model qwen2.5-coderaapi \
    --out_file qwen2.5-coder_32b_instruct_api.txt \
    --num_prompts 1034 \
    --batch_size 1


# ollama run phind-codellama
# ollama run qwen2.5-coder:32b-instruct-fp16