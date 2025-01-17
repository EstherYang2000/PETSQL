from evaluation import evaluate_cc,build_foreign_key_map_from_json
gold = "./data/spider/dev_gold.sql"
# with open(gold) as f:
#     glist = [l.strip().split('\t') for l in f.readlines() if len(l.strip()) > 0]
# print(glist[:1])
predict = "data/process/PPL_DEV.JSON-9_SHOT_Euclidean_mask_20/gpt-3.5-turbo.txt"
# with open(predict) as f:
#     plist = [l.strip().split('\t') for l in f.readlines() if len(l.strip()) > 0]
# print(plist[:1])    
gold_sql = ["""SELECT DISTINCT T1.Maker 
                FROM CAR_MAKERS AS T1 
                JOIN MODEL_LIST AS T2 ON T1.Id  =  T2.Maker 
                JOIN CAR_NAMES AS T3 ON T2.model  =  T3.model 
                JOIN CARS_DATA AS T4 ON T3.MakeId  =  T4.id 
                WHERE T4.year  =  '1970';""",
            "car_1"]  # This is already in the correct format
sql = ["""
       SELECT DISTINCT cm.Maker 
        FROM car_makers cm
        JOIN model_list ml ON cm.Id = ml.Maker
        JOIN car_names cn ON ml.ModelId = cn.Model
        JOIN cars_data cd ON cn.MakeId = cd.Id
        WHERE cd.Year = 1970;
       """]
db = "./data/spider/database"
etype = "all"
table = "./data/spider/tables.json"
kmaps = build_foreign_key_map_from_json(table)
is_correct = evaluate_cc(gold_sql, sql, db, etype, kmaps)
print(is_correct)