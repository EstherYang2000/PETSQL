import argparse
import json
import os
import pickle
from pathlib import Path
import sqlite3
from tqdm import tqdm
import random
import sys, os
sys.path.append(os.path.dirname(__file__))
from utils.linking_process import SpiderEncoderV2Preproc
from utils.pretrained_embeddings import GloVe
from utils.datasets.spider import load_tables, Schema
# from dataset.process.preprocess_kaggle import gather_questions
import torch

def schema_linking_producer(test, train, table, db, dataset_dir, compute_cv_link=True):

    # load data
    test_data = json.load(open(os.path.join(dataset_dir, test)))
    train_data = json.load(open(os.path.join(dataset_dir, train)))

    # load schemas
    schemas, _ = load_tables([os.path.join(dataset_dir, table)])
    
    # Backup in-memory copies of all the DBs and create the live connections
    for db_id, schema in schemas.items():
        # sqlite_path = Path(dataset_dir) / db / db_id / f"{db_id}.sqlite"
        sqlite_path = f"{dataset_dir}{db}/{db_id}/{db_id}.sqlite"
        print(sqlite_path)
        source: sqlite3.Connection
        with sqlite3.connect(str(sqlite_path)) as source:
            dest = sqlite3.connect(':memory:')
            dest.row_factory = sqlite3.Row
            source.backup(dest)
        schema.connection = dest

    word_emb = GloVe(kind='42B', lemmatize=True)
    linking_processor = SpiderEncoderV2Preproc(dataset_dir,
            min_freq=4,
            max_count=5000,
            include_table_name_in_column=False,
            word_emb=word_emb,
            fix_issue_16_primary_keys=True,
            compute_sc_link=True,
            compute_cv_link=compute_cv_link,
            device='cuda' if torch.cuda.is_available() else 'cpu'  )

    # build schema-linking
    print("Build test schema-linking ...")
    for data, section in zip([test_data, train_data],['test', 'train']):
        for item in tqdm(data, desc=f"{section} section linking"):
            db_id = item["db_id"]
            schema = schemas[db_id]
            to_add, validation_info = linking_processor.validate_item(item, schema, section)
            if to_add:
                # print(f"Add {item['db_id']} to {section} section")
                # print(item)
                # print(schema)
                # print(validation_info)
                linking_processor.add_item(item, schema, section, validation_info)

        
    # save
    linking_processor.save()


import re, json, os

from sql_metadata import Parser


def bird_pre_process(bird_dir, with_evidence=False):
    new_db_path = os.path.join(bird_dir, "database")
    if not os.path.exists(new_db_path):
        os.system(f"cp -r {os.path.join(bird_dir, 'train/train_databases/*')} {new_db_path}")
        os.system(f"cp -r {os.path.join(bird_dir, 'dev/dev_databases/*')} {new_db_path}")

    def json_preprocess(data_jsons):
        new_datas = []
        for data_json in data_jsons:
            ### Append the evidence to the question
            if with_evidence and len(data_json["evidence"]) > 0:
                data_json['question'] = (data_json['question'] + " " + data_json["evidence"]).strip()
            question = data_json['question']
            tokens = []
            for token in question.split(' '):
                if len(token) == 0:
                    continue
                if token[-1] in ['?', '.', ':', ';', ','] and len(token) > 1:
                    tokens.extend([token[:-1], token[-1:]])
                else:
                    tokens.append(token)
            data_json['question_toks'] = tokens
            data_json['query'] = data_json['SQL']
            new_datas.append(data_json)
        return new_datas
    output_dev = 'dev.json'
    output_train = 'train.json'
    with open(os.path.join(bird_dir, 'dev/dev.json')) as f:
        data_jsons = json.load(f)
        wf = open(os.path.join(bird_dir, output_dev), 'w')
        json.dump(json_preprocess(data_jsons), wf, indent=4)
    with open(os.path.join(bird_dir, 'train/train.json')) as f:
        data_jsons = json.load(f)
        wf = open(os.path.join(bird_dir, output_train), 'w')
        json.dump(json_preprocess(data_jsons), wf, indent=4)
    os.system(f"cp {os.path.join(bird_dir, 'dev/dev.sql')} {bird_dir}")
    os.system(f"cp {os.path.join(bird_dir, 'train/train_gold.sql')} {bird_dir}")
    tables = []
    with open(os.path.join(bird_dir, 'dev/dev_tables.json')) as f:
        tables.extend(json.load(f))
    with open(os.path.join(bird_dir, 'train/train_tables.json')) as f:
        tables.extend(json.load(f))
    with open(os.path.join(bird_dir, 'tables.json'), 'w') as f:
        json.dump(tables, f, indent=4)
        
proj_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))


def add_fk(ppl_test, data_type = "spider",dataset_type='dev'):
    if dataset_type == 'dev' or 'train':
        tables_data = json.load(
            open("data/spider/tables.json" if data_type == "spider" else "bird/bird/tables.json", 'r', encoding='utf-8'))
    elif dataset_type == 'test':
        tables_data = json.load(
            open("data/spider/test_tables.json" if data_type == "spider" else "bird/bird/test_tables.json", 'r', encoding='utf-8'))
        
    forekeys = {}
    anno_simddl = {}
    for db in tables_data:
        tables = db["table_names_original"]
        column_names = db["column_names_original"]
        column_types = db["column_types"]
        foreign_keys = db["foreign_keys"]
        sql_tem_all = []
        sql_tem_sim_ddl = []
        for idx, data in enumerate(tables):
            for j, column in enumerate(column_names):
                sql_tem = []
                if idx == column[0]:
                    sql_tem.append(column[0])
                    sql_tem.append(
                        str(column[1]) + " " + str(column_types[j]).upper())
                    sql_tem_all.append(sql_tem)
                    sql_tem_sim_ddl.append([column[0], column[1]])

        simddl_all = []
        for idx, data in enumerate(tables):
            sql_01 = "# " + str(data) + "("
            sql_final_tem = []
            for j, sql_final in enumerate(sql_tem_sim_ddl):
                if idx == sql_final[0]:
                    sql_final_tem.append(sql_final[1])
            sql_01 += ",".join(sql_final_tem) + ")"
            simddl_all.append(sql_01)
        anno_simddl[db["db_id"]] = simddl_all
        forkey = []
        for foreign in foreign_keys:
            vlaus = str(tables[int(
                    column_names[foreign[0]][0])]) + "(" + str(
                column_names[foreign[0]][1]) + ") REFERENCES " + str(tables[int(
                    column_names[foreign[1]][0])]) + "(" + str(
                        column_names[foreign[1]][1]) + ")"
            forkey.append(vlaus)
        forekeys[db["db_id"]] = forkey
    for i in range(len(ppl_test)):
        ppl_test[i]['foreign_key'] = ["\n".join(forekeys[ppl_test[i]["db"]])]
    return ppl_test



def gen_ppl_from_json(ppl_filename='data/ppl_dev.json',data_type = "spider",dataset_type='dev'):
    # Load database table information and development dataset
    print("Loading data...")
    print(proj_dir)
    print("data_type:", data_type)
    print("dataset_type:", dataset_type)
    if dataset_type == 'dev':
        print("Loading dev data...")
        tables_data = json.load(
            open(proj_dir + "/data/spider/tables.json" if data_type == "spider" else "bird/bird/tables.json", 'r', encoding='utf-8'))
        dev_data = json.load(
            open(proj_dir + "/data/spider/dev.json" if data_type == "spider" else "bird/bird/dev.json", 'r', encoding='utf-8'))
    elif dataset_type == 'train':
        print("Loading train data...")
        tables_data = json.load(
            open(proj_dir + "/data/spider/tables.json" if data_type == "spider" else "bird/bird/tables.json", 'r', encoding='utf-8'))
        dev_data = json.load(
            open(proj_dir + "/data/spider/dev.json" if data_type == "spider" else "bird/bird/train.json", 'r', encoding='utf-8'))
    elif dataset_type == 'test':
        print("Loading test data...")
        tables_data = json.load(
            open(proj_dir + "/data/spider/test_tables.json", 'r', encoding='utf-8'))
        dev_data = json.load(
            open(proj_dir + "/data/spider/test.json", 'r', encoding='utf-8'))
    
    # Initialize list to store processed test data
    ppl_test = []
    for ix, it in enumerate(dev_data):
        # Create a dictionary for each data entry containing id, db_id, question, and SQL query
        ppl_test.append({
            "id": ix,
            "db": it['db_id'],
            "question": it['question'],
            "gold_sql": it['query']
        })
    # Initialize dictionaries to store simplified and full DDL (Data Definition Language) annotations
    anno_simddl = {}
    anno = {}
    # Process each database schema in tables_data
    for db in tables_data:
        tables = db["table_names_original"] # List of table names
        column_names = db["column_names_original"] # List of column names with table indices
        column_types = db["column_types"] # List of column data types
        sql_tem_all = [] # List to store full DDL statements
        sql_tem_sim_ddl = [] # List to store simplified DDL statements
        # Process each table and its columns
        for idx, data in enumerate(tables):
            for j, column in enumerate(column_names):
                sql_tem = []
                if idx == column[0]: # Check if column belongs to the current table
                    # Store table index and column name with type
                    sql_tem.append(column[0])
                    sql_tem.append(
                        str(column[1]) + " " + str(column_types[j]).upper())
                    sql_tem_all.append(sql_tem)
                    sql_tem_sim_ddl.append([column[0], column[1]])

    # 外键
        # Process foreign keys and add them to the full DDL statements
        for foreign in db["foreign_keys"]:
            # Format foreign key constraints
            vlaus = str(tables[int(
                    column_names[foreign[0]][0])]) + "(" + str(
                column_names[foreign[0]][1]) + ") REFERENCES " + str(tables[int(
                    column_names[foreign[1]][0])]) + "(" + str(
                        column_names[foreign[1]][1]) + ")"
            #print(vlaus) Catalog_Contents(catalog_level_number) REFERENCES Catalog_Structure(catalog_level_number)
            sql_tem_all.append([column_names[foreign[0]][0], vlaus])
        
    #DDL语句
        ddl_all = []
        for idx, data in enumerate(tables):
            # 表名
            sql_01 = "\nCREATE TABLE " + str(data) + "("
            sql_final_tem = []
            for j, sql_final in enumerate(sql_tem_all):
                if idx == sql_final[0]: # Collect columns for the current table
                    sql_final_tem.append(sql_final[1])
            sql_01 += ", ".join(sql_final_tem) + ");"
            ddl_all.append(sql_01) 
        anno[db["db_id"]] = ddl_all # Store full DDL statements for the database
    
        #DDL语句
        #Create full DDL statements for each table

        # Create simplified DDL statements (without types) for each table
        simddl_all = []
        for idx, data in enumerate(tables):
            sql_01 = "# " + str(data) + "("
            sql_final_tem = []
            for j, sql_final in enumerate(sql_tem_sim_ddl):
                if idx == sql_final[0]: # Collect column names only
                    sql_final_tem.append(sql_final[1])
            sql_01 += ", ".join(sql_final_tem) + ")"
            simddl_all.append(sql_01)
        anno_simddl[db["db_id"]] = simddl_all # Store simplified DDL statements for the database
    # Add DDL annotations to each entry in ppl_test
    for i in range(len(ppl_test)):
        ppl_test[i]['simplified_ddl'] = ";\n".join(
            anno_simddl[ppl_test[i]["db"]]) + ".\n"
        ppl_test[i]['full_ddl'] = "\n".join(anno[ppl_test[i]["db"]]) + '\n'
    # Add foreign key information (assuming add_fk is a function that does this)
    ppl_test = add_fk(ppl_test, data_type,dataset_type)
    
    if data_type == "bird":
        # Add evidence information to each entry in ppl_test
        for i in range(len(ppl_test)):
            if 'evidence' in dev_data[i]:
                ppl_test[i]['evidence'] = dev_data[i]['evidence']
            else:
                ppl_test[i]['evidence'] = []
    # Save the final data to a JSON file
    json.dump(ppl_test,
              open(ppl_filename, 'w', encoding='utf-8'),
              ensure_ascii=False,
              indent=4)
    return ppl_test

import re, json, os

from sql_metadata import Parser

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="spider",choices=["spider", "bird"])
    parser.add_argument("--data_dir", type=str, default="data/spider/")
    parser.add_argument("--type",  default="dev", type=str, help="dev or test")
    args = parser.parse_args()
    if args.dataset == "spider":
    # merge two training split of Spider
        spider_dir = args.data_dir
        if args.type == "dev":
            spider_dev = "dev.json"
            spider_train = 'train_spider_and_others.json'
            spider_table = 'tables.json'
            spider_db = 'database'
        else:
            spider_dev = "test.json"
            spider_train = 'test_spider_and_others.json'
            spider_table = 'test_tables.json'
            spider_db = 'test_database'
        # merge two training split of Spider
        spider_dir = args.data_dir
        split1 = "train_spider.json"
        split2 = "train_others.json"
        total_train = []
        for item in json.load(open(os.path.join(spider_dir, split1))):
            total_train.append(item)
        for item in json.load(open(os.path.join(spider_dir, split2))):
            total_train.append(item)
        with open(os.path.join(spider_dir, spider_train), 'w') as f:
            json.dump(total_train, f)
        schema_linking_producer(spider_dev, spider_train, spider_table, spider_db, spider_dir)

    elif args.dataset == "bird":
        bird_dir = args.data_dir
        bird_pre_process(bird_dir, with_evidence=True)
        if args.type == "dev":
            bird_dev = "dev.json"
            bird_train = 'train.json'
            bird_table = 'tables.json'
            bird_db = 'database'
        else:
            bird_dev = "test.json"
            bird_train = 'test.json'
            bird_table = 'test_tables.json'
            bird_db = 'test_database'
        schema_linking_producer(bird_dev, bird_train, bird_table, bird_db, bird_dir)


"""

python src/sources/data_preprocess.py --dataset bird --data_dir bird/bird/ --type dev

"""