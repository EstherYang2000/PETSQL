#1. Data Preprocessing

echo "## Data processing ..."
START_TIME=`date +%s`
# python src/sources/data_preprocess.py --type test
END_TIME=`date +%s`
EXECUTING_TIME=`expr $END_TIME - $START_TIME`
echo "data preprocess time consume: $EXECUTING_TIME s"


# # 2. First Round of SQL Generation
# echo "## Start 1st generation of prompting ..."
# START_TIME=`date +%s`
# python src/sources/sql_gen/prompt_gen.py \
#   --dataset ppl_test.json \
#   --dataset_type test \
#   --n 1034 \
#   --kshot 9 \
#   --select_type Euclidean_mask
  
# echo "## Start 1st calling LLM ..."
# python src/sources/sql_gen/call_llm.py \
#     --path_generate data/process/PPL_TEST.JSON-9_SHOT_Euclidean_mask_1034\
#     --model gptapi \
#     --model_version o3-mini \
#     --out_file gptapi_o3-mini.json \
#     --dataset_type test \
#     --start_num_prompts 0 \
#     --end_num_prompts 1034 \
#     --batch_size 1 \
#     --call_mode append \
#     --n_samples 1

# python src/sources/post_process.py \
#     --file data/process/PPL_TEST.JSON-9_SHOT_Euclidean_mask_1034/gptapi_o3-mini.json \
#     --output data/process/PPL_TEST.JSON-9_SHOT_Euclidean_mask_1034/gptapi_o3-mini_output.json \
#     --llm sensechat
# python src/sources/extract_sql_output.py \
#     --file data/process/PPL_TEST.JSON-9_SHOT_Euclidean_mask_1034/gptapi_o3-mini_output.json \
#     --output data/process/PPL_TEST.JSON-9_SHOT_Euclidean_mask_1034/gptapi_o3-mini_output.txt


# python src/sources/evaluation.py \
#     --gold ./data/spider/test_gold.sql  \
#     --pred data/process/PPL_TEST.JSON-9_SHOT_Euclidean_mask_1034_o3_mini/gptapi_o3-mini_output.txt \
#     --etype all \
#     --db ./data/spider/test_database \
#     --table ./data/spider/test_tables.json

# # sleep 1
# # echo "1st round done!"
# # END_TIME=`date +%s`
# # EXECUTING_TIME=`expr $END_TIME - $START_TIME`
# echo "1st round time consume: $EXECUTING_TIME s"


# # 3. schema linking for the first round
# echo "## Start schema linking for the first round ..."
python src/sources/schemalink.py --output ppl_test_add_sl.json --dataset_type test --file data/process/PPL_TEST.JSON-9_SHOT_Euclidean_mask_1034_o3_mini/gptapi_o3-mini_output.txt


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
    --model gptapi \
    --model_version o3-mini \
    --out_file gptapi_o3-mini.json \
    --dataset_type test \
    --start_num_prompts 0 \
    --end_num_prompts 1034 \
    --batch_size 1 \
    --call_mode append \
    --n_samples 1


python src/sources/post_process.py \
    --file data/process/PPL_TEST_ADD_SL.JSON-9_SHOT_Euclidean_mask_1034/gptapi_o3-mini.json \
    --output data/process/PPL_TEST_ADD_SL.JSON-9_SHOT_Euclidean_mask_1034/gptapi_o3-mini_output.json \
    --llm sensechat

python src/sources/extract_sql_output.py \
    --file data/process/PPL_TEST_ADD_SL.JSON-9_SHOT_Euclidean_mask_1034/gptapi_o3-mini_output.json \
    --output data/process/PPL_TEST_ADD_SL.JSON-9_SHOT_Euclidean_mask_1034/gptapi_o3-mini_output.txt

python src/sources/evaluation.py \
    --gold ./data/spider/test_gold.sql  \
    --pred data/process/PPL_TEST_ADD_SL.JSON-9_SHOT_Euclidean_mask_1034_o3_mini_rf/final_sql_1.txt\
    --etype all \
    --db ./data/spider/test_database \
    --table ./data/spider/test_tables.json > output.txt





"""
100
qwen 2.5
    old   0.82 50筆  , 0.81  100 筆 , 0.806 500筆

    1. new 0.73 (0.5.0 / ...unners/cuda_v12/ollama_llama_server)2024/12/05 temperature None / 100 筆
    2. new 0.73 (0.5.0 / ...unners/cuda_v12/ollama_llama_server)2024/12/05 temperature 0.0 / 100 筆
    (2) 25 new 0.74 (0.4.2 / ...unners/cuda_v12/ollama_llama_server) 2024/11/15 temperature None, "num_ctx": None / 50 筆  w/  qwen2.5:72b
    26.1 new 0.70 (0.4.1 / ...unners/cuda_v12/ollama_llama_server) 2024/11/9 temperature None, "num_ctx": None / 50 筆  w/  qwen2.5:72b
    27 new 0.720 (0.4.1 / ...unners/cuda_v12/ollama_llama_server) 2024/10/21 temperature 0.0, "num_ctx": None / 50 筆  w/  qwen2.5:72b
    28 new 0.720 (0.4.1 / ...unners/cuda_v12/ollama_llama_server) 2024/10/21 temperature 0.5, "num_ctx": None / 50 筆  w/  qwen2.5:72b
llama 3.3:latest 
    old 0.82 50筆 / 0.76 100 筆 / 0,804 500  筆
    33 new 0.680 (0.5.0 / ...unners/cuda_v12/ollama_llama_server) 2024/12/05 temperature 0.0 / 50 筆 w/  qwen2.5:72b docker image

    new 0.670 (0.5.0)
    1. new 0.670 (0.5.5 / ...rs/cuda_v12_avx/ollama_llama_server) 2025/01/09 temperature None / 100 筆
    1.1 new 0.760 (0.5.5 / ...rs/cuda_v12_avx/ollama_llama_server) 2025/01/09 temperature 0.0 / 50 筆
    1.2 new 0.760 (0.5.5 / ...rs/cuda_v12_avx/ollama_llama_server) 2025/01/09 temperature 0.0 / 50 筆 w/ llama 3.3 70b-instruct-q6_K    
    12 new 0.760 (0.5.5 / ...rs/cuda_v12_avx/ollama_llama_server) 2025/01/09 temperature 0.0 / 50 筆 w/ llama 3.3 70b-instruct-q6_K

    2. new 0.660 (0.5.4 / ...rs/cuda_v12_avx/ollama_llama_server) 2024/12/18 temperature None / 100 筆
    3. new 0.660 (0.5.3 / ...rs/cuda_v12_avx/ollama_llama_server) 2024/12/17 temperature None / 100 筆
    4. new 0.650 (0.5.2 / ...rs/cuda_v12_avx/ollama_llama_server) 2024/12/12 temperature None / 100 筆
    5. new 0.670 (0.5.1 / ...unners/cuda_v12/ollama_llama_server)  2024/12/07 temperature None / 100 筆
    13 new 0.696 (0.5.1 / ...unners/cuda_v12/ollama_llama_server)  2024/12/07 temperature None / 500 筆 w/  70b-instruct-q6_K
    14 new 0.660 (0.5.1 / ...unners/cuda_v12/ollama_llama_server)  2024/12/07 temperature None / 50 筆 w/  llama 3.1:70 b
    15 new 0.680 (0.5.1 / ...unners/cuda_v12/ollama_llama_server)  2024/12/07 temperature 0.0 / 50 筆 w/  llama 3.1:70 b
    16 new 0.760 (0.5.1 / ...unners/cuda_v12/ollama_llama_server)  2024/12/07 temperature 0.0 / 50 筆 w/  llama 3.3:latest
    6. new 0.670 (0.5.0 / ...unners/cuda_v12/ollama_llama_server)2024/12/05 temperature None / 100 筆
    17. new 0.760 (0.5.0 / ...unners/cuda_v12/ollama_llama_server) 2024/12/05 temperature 0.0 / 50 筆 w/  llama 3.3:latest
    18. new 0.670 (0.5.0 / ...unners/cuda_v12/ollama_llama_server) 2024/12/05 temperature 0.0 / 100 筆 w/  llama 3.3:latest
    19 new 0.760 (0.5.0 / ...unners/cuda_v12/ollama_llama_server) 2024/12/05 temperature 0.0 / 50 筆 w/  llama3.3_tp0:latest modelfile temperature 0.0
    20 new 0.760 (0.5.0 / ...unners/cuda_v12/ollama_llama_server) 2024/12/05 temperature 0.0 / 50 筆 w/  llama3.3_tp0:latest modelfile temperature 0.0 new chat
    7. new 0.650 (0.4.7 / ...unners/cuda_v12/ollama_llama_server) 2024/12/01 temperature None / 100 筆  w/  llama 3.3:latest
    8. new 0.650 (0.4.6 / ...unners/cuda_v12/ollama_llama_server) 2024/11/28 temperature None / 100 筆  w/  llama 3.3:latest
    9. new 0.760 (0.4.5 / ...unners/cuda_v12/ollama_llama_server) 2024/11/26 temperature 0.0, "num_ctx": 4096 / 50 筆  w/  llama 3.3:latest
    9.1 new 0.760 (0.4.5 / ...unners/cuda_v12/ollama_llama_server) 2024/11/26 temperature 0.0, "num_ctx": 8192 / 50 筆  w/  llama 3.3:latest
    10 new 0.670 (0.4.4 / ...unners/cuda_v12/ollama_llama_server) 2024/11/23 temperature None
    11 new 0.670 (0.4.4 / ...unners/cuda_v12/ollama_llama_server) 2024/11/23 temperature 0.0 / 100 筆  w/  llama 3.3:latest
   21 new 0.76 (0.4.3 / ...unners/cuda_v12/ollama_llama_server) 2024/11/21 temperature 0.0, "num_ctx": 8192 / 50 筆  w/  llama 3.3:latest
   21.1 new 0.760 (0.4.3 / ...unners/cuda_v12/ollama_llama_server) 2024/11/21 temperature 0.0, "num_ctx": 16384 / 50 筆  w/  llama 3.3:latest
   22 new 0.76. (0.4.2 / ...unners/cuda_v12/ollama_llama_server) 2024/11/15 temperature 0.0, "num_ctx": 4096 / 50 筆  w/  llama 3.3:latest
   23 new 0.68. (0.4.2 / ...unners/cuda_v12/ollama_llama_server) 2024/11/15 temperature 0.0, "num_ctx": 4096 / 50 筆  w/  llama 3.1:70b
   24 new 0.760 (0.4.2 / ...unners/cuda_v12/ollama_llama_server) 2024/11/15 temperature 0.0, "num_ctx": 4096 / 50 筆  w/  llama3.3:70b-instruct-q6_K
    29 new 0.760 (0.4.1 / ...unners/cuda_v12/ollama_llama_server) 2024/10/21 temperature 0.0, "num_ctx": None / 50 筆  w/  llama 3.3:latest
    30 new 0.760 (0.4.1 / ...unners/cuda_v12/ollama_llama_server) 2024/10/21 temperature 0.0, "num_ctx": None / 50 筆  w/  llama 3.3:latest
    31 new 0.700 (0.4.1 / ...unners/cuda_v12/ollama_llama_server) 2024/10/21 temperature 0.0, "num_ctx": None / 50 筆  w/  llama-3.3-70b-specdec Groq
    32 new 0.760 (0.4.1 / ...unners/cuda_v12/ollama_llama_server) 2024/10/21 temperature 0.0, "num_ctx": None / 50 筆  w/  llama3-70b-8192	Groq
    33 new 0.74 (0.4.6 / ...unners/cuda_v12/ollama_llama_server) 2024/10/21 temperature 0.0, "num_ctx": None / 50 筆  w/ llama 3.3:latest	docker image latest
    34 new 0.682 (0.4.6 / ...unners/cuda_v12/ollama_llama_server) 2024/10/21 "temperature": 0.0,"top_p": 0.9,"top_k": 50 / 1034 筆  w/ llama 3.3:latest	docker image latest


docker run --gpus all --rm -it \
  -e OLLAMA_CUDA=1 \
  -e OLLAMA_PRECISION=fp16 \
  -v ollama_data:/root/.ollama \
  -p -p 11434:11434 \
  ollama/ollama

"""