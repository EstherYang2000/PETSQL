#1. Data Preprocessing

echo "## Data processing ..."
source /root/venv/bin/activate
export NLTK_DATA=src/sources/nltk_data
export CORENLP_HOME=src/sources/third_party/stanford-corenlp-full-4.5.7
export CACHE_DIR=/root
START_TIME=`date +%s`
python src/sources/data_preprocess.py
END_TIME=`date +%s`
EXECUTING_TIME=`expr $END_TIME - $START_TIME`
echo "data preprocess time consume: $EXECUTING_TIME s"
# 2. First Round of SQL Generation
echo "## Start generation of prediction file: 'predicted_sql.txt' ..."
START_TIME=`date +%s`
python src/sources/sql_gen/main.py --model "$1api" --kshot 9 --pool 1  --out_file src/sources/raw.txt --select_type Euclidean_mask
sleep 1
echo "1st round done!"
END_TIME=`date +%s`
EXECUTING_TIME=`expr $END_TIME - $START_TIME`
echo "1st round time consume: $EXECUTING_TIME s"

# 3. Post-Processing After the First Round
python src/sources/post_process.py --file src/sources/raw.txt --llm $1
mv src/sources/raw_out.txt src/sources/"intermediate_results_only_dont_use_$1.txt"
python src/sources/schemalink.py --output ppl_dev_add_sl.json

#4. Second Round of SQL Generation
START_TIME=`date +%s`
python src/sources/sql_gen/main.py --model "$1api" --kshot 9 --pool 1  --out_file src/sources/raw.txt --select_type Euclidean_mask --sl --dataset ppl_dev_add_sl.json
sleep 1
echo "2nd round done!"
END_TIME=`date +%s`
EXECUTING_TIME=`expr $END_TIME - $START_TIME`
echo "2nd round time consume: $EXECUTING_TIME s"

#5. Final Post-Processing
python src/sources/post_process.py --file src/sources/raw.txt --llm $1
mv src/sources/raw_out.txt src/sources/"predicted_sql_$1.txt"
echo "File 'predicted_sql.txt' generated."