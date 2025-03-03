from evaluation import evaluate_cc,build_foreign_key_map_from_json
gold = "./data/spider/dev_gold.sql"
# with open(gold) as f:
#     glist = [l.strip().split('\t') for l in f.readlines() if len(l.strip()) > 0]
# print(glist[:1])
predict = "data/process/PPL_DEV.JSON-9_SHOT_Euclidean_mask_20/gpt-3.5-turbo.txt"
# with open(predict) as f:
#     plist = [l.strip().split('\t') for l in f.readlines() if len(l.strip()) > 0]
# print(plist[:1])    
gold_sql = ["""SELECT avg(age) ,  min(age) ,  max(age) FROM singer WHERE country  =  'France'""",
            "concert_singer"]  # This is already in the correct format
sql = ["""
       SELECT AVG(Age) as Average_Age,MIN(Age) as Minimum_Age,MAX(Age) as Maximum_Age FROM singer WHERE Country = 'France';
       """]
db = "./data/spider/database"
etype = "all"
table = "./data/spider/tables.json"
kmaps = build_foreign_key_map_from_json(table)
is_correct = evaluate_cc(gold_sql, sql, db, etype, kmaps)
print(is_correct)