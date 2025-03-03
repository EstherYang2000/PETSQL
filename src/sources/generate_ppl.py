import json

result = []
with open('test_schema-linking.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

for index,item in enumerate(data):
    data = {}
    data['id'] = index
    data['db_id'] = item['db_id']
    data['question'] = item['question']
    data['gold_sql'] = item['query']
    
