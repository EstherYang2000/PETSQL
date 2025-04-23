result_dir="data/process/vote/202504/PPL_DEV.JSON-9_SHOT_Euclidean_mask_1034_base_rwma"
strategy="RWMA"
title="((1) Baseline PETSQL - No schema linking, no self‑refinement"  # Properly quoted
number="1_2"
dataset_type="spider_dev"
data_type="Dev"

python src/sources/wma/weight_plotting.py \
    --result_dir "${result_dir}" \
    --strategy "${strategy}" \
    --title "${title}" \
    --number "${number}" \
    --dataset_type "${dataset_type}" \
    --data_type "${data_type}"

python src/sources/wma/model_error_plotting.py \
    --result_dir "${result_dir}" \
    --strategy "${strategy}" \
    --title "${title}" \
    --number "${number}" \
    --dataset_type "${dataset_type}" \
    --data_type "${data_type}"

python src/sources/wma/error_rate_plotting.py \
    --result_dir "${result_dir}" \
    --strategy "${strategy}" \
    --title "${title}" \
    --number "${number}" \
    --dataset_type "${dataset_type}" \
    --data_type "${data_type}"