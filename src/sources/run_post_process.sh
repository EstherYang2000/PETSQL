echo "## Start post processing for codellama_34b-instruct ..."
sleep 1
python src/sources/post_process.py \
    --file data/process/PPL_DEV.JSON-3_SHOT_Euclidean_mask_1034/codellama_34b-instruct_api.txt \
    --output data/process/PPL_DEV.JSON-3_SHOT_Euclidean_mask_1034/codellama_34b-instruct_api_output.txt \
    --llm sensechat \
echo "## Start evaluation for codellama_34b-instruct ..."

python src/sources/evaluation.py \
    --gold ./data/spider/dev_gold.sql  \
    --pred data/process/PPL_DEV.JSON-9_SHOT_Euclidean_mask_1034/codellama_34b-instruct_api_output.txt \
    --etype all \
    --db ./data/spider/database \
    --table ./data/spider/tables.json \

echo "## Start post processing for deepseek_v2-16b ..."
sleep 1
python src/sources/post_process.py \
    --file data/process/PPL_DEV.JSON-3_SHOT_Euclidean_mask_1034/deepseek_v2-16b_api.txt \
    --output data/process/PPL_DEV.JSON-3_SHOT_Euclidean_mask_1034/deepseek_v2-16b_api_output.txt \
    --llm sensechat \

echo "## Start evaluation for deepseek_v2-16b ..."

python src/sources/evaluation.py \
    --gold ./data/spider/dev_gold.sql  \
    --pred data/process/PPL_DEV.JSON-9_SHOT_Euclidean_mask_1034/deepseek_v2-16b_api_output.txt \
    --etype all \
    --db ./data/spider/database \
    --table ./data/spider/tables.json \

echo "## Start post processing for llama3.3:latest ..."
sleep 1
python src/sources/post_process.py \
    --file data/process/PPL_DEV.JSON-3_SHOT_Euclidean_mask_1034/llama3.3:latest_api.txt \
    --output data/process/PPL_DEV.JSON-3_SHOT_Euclidean_mask_1034/llama3.3:latest_api_output.txt \
    --llm sensechat \

python src/sources/evaluation.py \
    --gold ./data/spider/dev_gold.sql  \
    --pred data/process/PPL_DEV.JSON-3_SHOT_Euclidean_mask_1034/llama3.3:latest_api_output.txt \
    --etype all \
    --db ./data/spider/database \
    --table ./data/spider/tables.json \

echo "## Start post processing for phind-codellama ..."
sleep 1
python src/sources/post_process.py \
    --file data/process/PPL_DEV.JSON-3_SHOT_Euclidean_mask_1034/phind-codellama_api.txt \
    --output data/process/PPL_DEV.JSON-3_SHOT_Euclidean_mask_1034/phind-codellama_api_output.txt \
    --llm sensechat \

echo "## Start evaluation for phind-codellama ..."
python src/sources/evaluation.py \
    --gold ./data/spider/dev_gold.sql  \
    --pred data/process/PPL_DEV.JSON-3_SHOT_Euclidean_mask_1034/phind-codellama_api_output.txt \
    --etype all \
    --db ./data/spider/database \
    --table ./data/spider/tables.json \

echo "## Start post processing for qwen2.5-coder_32b_instruct ..."
sleep 1
python src/sources/post_process.py \
    --file data/process/PPL_DEV.JSON-3_SHOT_Euclidean_mask_1034/qwen2.5-coder_32b_instruct_api.txt \
    --output data/process/PPL_DEV.JSON-3_SHOT_Euclidean_mask_1034/qwen2.5-coder_32b_instruct_api_output.txt \
    --llm sensechat \

echo "## Start evaluation for qwen2.5-coder_32b_instruct ..."
python src/sources/evaluation.py \
    --gold ./data/spider/dev_gold.sql  \
    --pred data/process/PPL_DEV.JSON-3_SHOT_Euclidean_mask_1034/qwen2.5-coder_32b_instruct_api_output.txt \
    --etype all \
    --db ./data/spider/database \
    --table ./data/spider/tables.json \