#1. Data Preprocessing

# echo "## Data processing ..."
# START_TIME=`date +%s`
python src/sources/data_preprocess.py --type dev
# END_TIME=`date +%s`
# EXECUTING_TIME=`expr $END_TIME - $START_TIME`
# echo "data preprocess time consume: $EXECUTING_TIME s"


# 2. First Round of SQL Generation
echo "## Start 1st generation of prompting ..."
START_TIME=`date +%s`
python src/sources/sql_gen/prompt_gen.py \
  --dataset ppl_dev.json \
  --dataset_type dev \
  --n 1034 \
  --kshot 9 \
  --select_type Euclidean_mask 
  
# echo "## Start 1st calling LLM ..."
python src/sources/sql_gen/call_llm.py \
    --path_generate data/process/PPL_DEV.JSON-9_SHOT_Euclidean_mask_1034 \
    --model qwen_api \
    --model_version 32b-instruct-fp16 \
    --out_file qwen_api_32b-instruct-fp16.json \
    --dataset_type dev \
    --start_num_prompts 100 \
    --end_num_prompts 1034 \
    --batch_size 1 \
    --call_mode append \
    --n_samples 1

python src/sources/post_process.py \
    --file data/process/PPL_DEV.JSON-9_SHOT_Euclidean_mask_1034/qwen_api_32b-instruct-fp16.json \
    --output data/process/PPL_DEV.JSON-9_SHOT_Euclidean_mask_1034/qwen_api_32b-instruct-fp16_output.json \
    --llm sensechat
python src/sources/extract_sql_output.py \
    --file data/process/PPL_DEV.JSON-9_SHOT_Euclidean_mask_1034/qwen_api_32b-instruct-fp16_output.json \
    --output data/process/PPL_DEV.JSON-9_SHOT_Euclidean_mask_1034/qwen_api_32b-instruct-fp16_output.txt


python src/sources/evaluation.py \
    --gold ./data/spider/dev_gold.sql  \
    --pred data/process/PPL_DEV.JSON-9_SHOT_Euclidean_mask_1034/qwen_api_32b-instruct-fp16_output.txt \
    --etype all \
    --db ./data/spider/database \
    --table ./data/spider/tables.json \
    --num 1034

# sleep 1
# echo "1st round done!"
# END_TIME=`date +%s`
# EXECUTING_TIME=`expr $END_TIME - $START_TIME`
# echo "1st round time consume: $EXECUTING_TIME s"


# # # 3. schema linking for the first round
echo "## Start schema linking for the first round ..."
python src/sources/schemalink.py --output ppl_dev_add_sl.json --file data/process/PPL_DEV.JSON-9_SHOT_Euclidean_mask_1034_qwen_32b/qwen_api_32b-instruct-fp16_output.txt


# 4. Second Round of SQL Generation
echo "## Start 2nd generation of prompting ..."
START_TIME=`date +%s`
python src/sources/sql_gen/prompt_gen.py \
    --dataset ppl_dev_add_sl.json \
    --dataset_type dev \
    --n 1034 \
    --kshot 9 \
    --select_type Euclidean_mask \
    --sl
# echo "## Start 2nd calling LLM ..."
python src/sources/sql_gen/call_llm.py \
    --path_generate data/process/PPL_DEV_ADD_SL.JSON-9_SHOT_Euclidean_mask_1034 \
    --model qwen_api \
    --model_version 32b-instruct-fp16 \
    --out_file qwen_api_32b-instruct-fp16.json \
    --dataset_type dev \
    --start_num_prompts 0 \
    --end_num_prompts 1034 \
    --batch_size 1 \
    --call_mode append \
    --n_samples 1


python src/sources/post_process.py \
    --file data/process/PPL_DEV_ADD_SL.JSON-9_SHOT_Euclidean_mask_1034/qwen_api_32b-instruct-fp16.json \
    --output data/process/PPL_DEV_ADD_SL.JSON-9_SHOT_Euclidean_mask_1034/qwen_api_32b-instruct-fp16_output.json  \
    --llm sensechat

python src/sources/extract_sql_output.py \
    --file data/process/PPL_DEV_ADD_SL.JSON-9_SHOT_Euclidean_mask_1034/qwen_api_32b-instruct-fp16_output.json \
    --output data/process/PPL_DEV_ADD_SL.JSON-9_SHOT_Euclidean_mask_1034/qwen_api_32b-instruct-fp16_output.txt

python src/sources/evaluation.py \
    --gold ./data/spider/dev_gold.sql  \
    --pred data/process/PPL_DEV_ADD_SL.JSON-9_SHOT_Euclidean_mask_1034/qwen_api_32b-instruct-fp16_output.txt \
    --etype all \
    --db ./data/spider/database \
    --table ./data/spider/tables.json


# python src/sources/evaluation.py \
#     --gold ./data/spider/dev_gold.sql  \
#     --pred data/process/PPL_DEV_ADD_SL.JSON-9_SHOT_Euclidean_mask_1034_simple_sl_rf_naive/final_sql_1.txt \
#     --etype all \
#     --db ./data/spider/database \
#     --table ./data/spider/tables.json \
#     --num 1034



