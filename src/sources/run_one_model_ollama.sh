#1. Data Preprocessing

echo "## Data processing ..."
START_TIME=`date +%s`
python src/sources/data_preprocess.py
END_TIME=`date +%s`
EXECUTING_TIME=`expr $END_TIME - $START_TIME`
echo "data preprocess time consume: $EXECUTING_TIME s"

# 2. First Round of SQL Generation
echo "## Start generation of prompting ..."
START_TIME=`date +%s`
python src/sources/sql_gen/prompt_gen.py --kshot 9 --pool 1 --select_type Euclidean_mask --n 1034 --sl --dataset ppl_dev_add_sl.json
sleep 1
echo "1st round done!"
END_TIME=`date +%s`
EXECUTING_TIME=`expr $END_TIME - $START_TIME`
echo "1st round time consume: $EXECUTING_TIME s"

# python src/sources/sql_gen/call_llm.py \
#     --path_generate /home/yyj/Desktop/yyj/thesis/code/PETSQL/data/process/PPL_DEV.JSON-9_SHOT_Euclidean_mask_1034 \
#     --model phind-codellamaapi \
#     --out_file phind-codellama_api.txt \
#     --num_prompts 1034 \
#     --batch_size 1

