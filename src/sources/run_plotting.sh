output_dir="data/pic/1_baseline"
# strategy="RWMA"
# title="((1) Baseline PETSQL - No schema linking, no self‑refinement"  # Properly quoted
# number="1_2"
dataset_type="Spider 1.0"
data_type="Dev"
wma_path = "data/vote/202504/PPL_DEV.JSON-9_SHOT_Euclidean_mask_1034_base_wma"
rwma_path = "data/vote/202504/PPL_DEV.JSON-9_SHOT_Euclidean_mask_1034_base_rwma"
navie_path = "data/vote/202504/PPL_DEV.JSON-9_SHOT_Euclidean_mask_1034_base_naive"
# python src/sources/wma/weight_plotting.py \
#     --result_dir "${result_dir}" \
#     --strategy "${strategy}" \
#     --title "${title}" \
#     --number "${number}" \
#     --dataset_type "${dataset_type}" \
#     --data_type "${data_type}"

# python src/sources/wma/model_error_plotting.py \
#     --result_dir "${result_dir}" \
#     --strategy "${strategy}" \
#     --title "${title}" \
#     --number "${number}" \
#     --dataset_type "${dataset_type}" \
#     --data_type "${data_type}"

# python src/sources/wma/error_rate_plotting.py \
#     --result_dir "${result_dir}" \
#     --strategy "${strategy}" \
#     --title "${title}" \
#     --number "${number}" \
#     --dataset_type "${dataset_type}" \
#     --data_type "${data_type}"

python src/sources/wma/weight_plotting.py \
--wma_dir "${wma_path}" \
--rwma_dir "${rwma_path}" \
--naive_dir "${naive_path}" \
--dataset_type "${dataset_type}" \
--data_type  "${data_type}" \
--output_dir "${output_dir}" \
--figure_number "1"

python src/sources/wma/error_rate_plotting.py \
    --wma_dir "${wma_path}" \
    --rwma_dir "${rwma_path}" \
    --naive_dir "${naive_path}" \
    --dataset_type "${dataset_type}" \
    --data_type  "${data_type}" \
    --output_dir "${output_dir}" \
    --figure_number "1"