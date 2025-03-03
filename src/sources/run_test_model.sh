#1. Data Preprocessing

echo "## Data processing ..."
START_TIME=`date +%s`
python src/sources/data_preprocess.py --type test
END_TIME=`date +%s`
EXECUTING_TIME=`expr $END_TIME - $START_TIME`
echo "data preprocess time consume: $EXECUTING_TIME s"


# 2. First Round of SQL Generation
echo "## Start 1st generation of prompting ..."
START_TIME=`date +%s`
python src/sources/sql_gen/prompt_gen.py \
  --dataset ppl_test.json \
  --dataset_type test \
  --n 1034 \
  --kshot 9 \
  --select_type Euclidean_mask 
  
echo "## Start 1st calling LLM ..."
python src/sources/sql_gen/call_llm.py \
    --path_generate data/process/PPL_TEST.JSON-9_SHOT_Euclidean_mask_1034 \
    --model qwen_api \
    --model_version 2_5_72b \
    --out_file qwen_api_2_5_72b.txt \
    --start_num_prompts 0 \
    --num_prompts 1034 \
    --batch_size 1 \
    --call_mode append


python src/sources/post_process.py \
    --file data/process/PPL_TEST.JSON-9_SHOT_Euclidean_mask_1034/llamaapi_3.3.txt \
    --output data/process/PPL_TEST.JSON-9_SHOT_Euclidean_mask_1034/llamaapi_3.3_output.txt \
    --llm sensechat

python src/sources/evaluation.py \
    --gold ./data/spider/test_gold.sql  \
    --pred data/process/PPL_TEST.JSON-9_SHOT_Euclidean_mask_1034/llamaapi_3.3_output.txt \
    --etype all \
    --db ./data/spider/test_database \
    --table ./data/spider/test_tables.json

sleep 1
echo "1st round done!"
END_TIME=`date +%s`
EXECUTING_TIME=`expr $END_TIME - $START_TIME`
echo "1st round time consume: $EXECUTING_TIME s"


# # 3. schema linking for the first round
echo "## Start schema linking for the first round ..."
python src/sources/schemalink.py --output ppl_test_add_sl.json --file data/process/PPL_TEST.JSON-9_SHOT_Euclidean_mask_1034/llamaapi_3.3_output.txt


# 4. Second Round of SQL Generation
echo "## Start 2nd generation of prompting ..."
START_TIME=`date +%s`
python src/sources/sql_gen/prompt_gen.py \
    --dataset ppl_test_add_sl.json \
    --dataset_type test \
    --n 1034 \
    --kshot 9 \
    --select_type Euclidean_mask \
    --sl
# echo "## Start 2nd calling LLM ..."
python src/sources/sql_gen/call_llm.py \
    --path_generate data/process/PPL_TEST_ADD_SL.JSON-9_SHOT_Euclidean_mask_1034 \
    --model llamaapi \
    --model_version 3.3 \
    --out_file llamaapi_3.3.txt \
    --start_num_prompts 0 \
    --num_prompts 1034 \
    --batch_size 1 \
    --call_mode append

python src/sources/post_process.py \
    --file data/process/PPL_TEST_ADD_SL.JSON-9_SHOT_Euclidean_mask_1034/llamaapi_3.3.txt \
    --output data/process/PPL_TEST_ADD_SL.JSON-9_SHOT_Euclidean_mask_1034/llamaapi_3.3_output.txt \
    --llm sensechat

python src/sources/evaluation.py \
    --gold ./data/spider/test_gold.sql  \
    --pred data/process/PPL_TEST_ADD_SL.JSON-9_SHOT_Euclidean_mask_1034/llamaapi_3.3_output.txt \
    --etype all \
    --db ./data/spider/test_database \
    --table ./data/spider/test_tables.json