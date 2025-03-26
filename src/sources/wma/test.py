from evaluation import evaluate_cc,build_foreign_key_map_from_json
gold = "./data/spider/dev_gold.sql"
# with open(gold) as f:
#     glist = [l.strip().split('\t') for l in f.readlines() if len(l.strip()) > 0]
# print(glist[:1])
predict = "data/process/PPL_DEV.JSON-9_SHOT_Euclidean_mask_20/gpt-3.5-turbo.txt"
# with open(predict) as f:
#     plist = [l.strip().split('\t') for l in f.readlines() if len(l.strip()) > 0]
# print(plist[:1])    
gold_sql = ["""SELECT C.city_code,        COALESCE(O.sum_distance, 0) + COALESCE(I.sum_distance, 0) AS total_distance FROM City C LEFT JOIN (SELECT city1_code, SUM(distance) AS sum_distance FROM Direct_distance GROUP BY city1_code) O ON C.city_code = O.city1_code LEFT JOIN (SELECT city2_code, SUM(distance) AS sum_distance FROM Direct_distance GROUP BY city2_code) I ON C.city_code = I.city2_code;""",
            "address_1"]  # This is already in the correct format
sql = ["""
       SELECT city1_code ,  sum(distance) FROM Direct_distance GROUP BY city1_code;
       """]
db = "./data/spider/test_database"
etype = "all"
table = "./data/spider/tables.json"
kmaps = build_foreign_key_map_from_json(table)
is_correct = evaluate_cc(gold_sql, sql, db, etype, kmaps)
print(is_correct)