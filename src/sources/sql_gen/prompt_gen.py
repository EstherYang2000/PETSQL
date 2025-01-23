import json
import logging
import os
import sys
import argparse
import sqlite3
from tqdm import tqdm

# 根據專案結構做路徑調整
proj_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(proj_dir)

# 避免 Tokenizers 產生多工警告
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 專案內部匯入
from get_example_modules import get_examples_ins
from data_preprocess import gen_ppl_from_json

######################################################
# Global variables
######################################################
model_pipeline = None  # 全域 Model Pipeline (若使用 pipeline)


######################################################
# Utility functions
######################################################
def get_example_prefix():
    """
    回傳系統引導用的前綴字串 (for few-shot 提示)
    """
    return ("### Some example pairs of question and corresponding SQL query "
            "are provided based on similar problems:\n\n")

def prompt_prefix():
    """
    回傳系統提示的前綴字串
    """
    return ("### Let's think step by step. Complete sqlite SQL query only "
            "and with no explanation\n")

def task_prefix():
    """
    回傳「任務說明」的字串
    """
    return (
        "### Your task: \n"
        "Answer the final question below by providing **only** the final "
        "SQLite SQL query syntax without commentary and explanation.  "
        "You must minimize SQL execution time while ensuring correctness.\n"
    )

def format_example(example: dict):
    """
    將單一訓練樣本格式化為可供 few-shot 使用的 prompt 片段
    """
    template_qa = "### {}\n{}"
    return template_qa.format(example['question'], example['gold_sql'])

######################################################
# Prompt formatting functions
######################################################
def formatting_prompt(sample):
    """
    預設模式下的 prompt 生成 (不採用schema linking)。
    會動態讀取 sqlite 資料庫，拿前三行數據產生 table schema 提示。
    """
    question = sample['question']
    ddls = sample['simplified_ddl']
    db = sample['db']

    # 連接對應的 SQLite DB
    mydb = sqlite3.connect(f"data/spider/test_database/{db}/{db}.sqlite")
    cur = mydb.cursor()

    # 收集所有 table 及其前三行資料
    simplified_ddl_data = []
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cur.fetchall()  # [(table_name,), (table_name,), ...]

    for tbl in tables:
        table_name = tbl[0]
        cur.execute(f"SELECT * FROM `{table_name}`")
        col_name_list = [desc[0] for desc in cur.description]
        db_data_all = []
        for _ in range(3):
            db_data_all.append(cur.fetchone())

        row_info = ""
        for idx, column_data in enumerate(col_name_list):
            try:
                # 取前三行的欄位值
                first = db_data_all[0][idx]
                second = db_data_all[1][idx]
                third = db_data_all[2][idx]
                row_info += f"{column_data}[{first},{second},{third}],"
            except:
                # 遇到空值或存取失敗就跳過
                pass
        simplified_ddl_data.append(f"{table_name}({row_info[:-1]})")

    ddls_data = "# " + ";\n# ".join(simplified_ddl_data) + ";\n"

    # foreign_key 資訊
    fk_info = ""
    for fk_data in sample["foreign_key"][0].split("\n"):
        fk_info += f'# {fk_data};\n'
    fk_info = fk_info[:-2]  # 移除最後多餘的換行

    prompt = (
        f"{task_prefix()}\n"
        "    ### Sqlite SQL tables, with their properties:\n#\n"
        f"{ddls}#\n"
        "    # ### Here are some data information about database references.\n"
        "    # #\n"
        f"{ddls_data}#\n"
        "### Foreign key information of Sqlite SQL tables, used for table joins: \n"
        f"#\n{fk_info}\n#\n"
        f"### Final Question: {question}\n"
        "### SQL: "
    )

    return prompt

def formatting_prompt_sl(sample):
    """
    schema linking 模式下的 prompt 生成。
    只會收集用到的表的 ddls 與前三行資料，並整合 foreign key。
    """
    # 根據 GPT 做 schema linking (linked_tables_gpt)
    linked_tables = [tbl.lower() for tbl in sample['linked_tables_gpt']]
    sc_tables = []
    all_ddl = sample['simplified_ddl'].split("\n")

    # 收集會用到的 table 名
    for ddl_line in all_ddl:
        table_candidate = ddl_line.split("(")[0].strip("#").strip()
        if table_candidate.lower() in linked_tables:
            sc_tables.append(table_candidate)

    db = sample['db']
    # 剪去最後的 ";."
    ddl_str = sample['simplified_ddl'][:-2]
    split_ddl = ddl_str.split(";\n")

    # 外鍵 (foreign_key)
    fk_all = sample["foreign_key"][0].split("\n")
    fk_sc = []

    # (A) 篩選外鍵
    for fk_line in fk_all:
        matched_count = 0
        for sc_table in sc_tables:
            if f" {sc_table}(" in fk_line.lower():
                matched_count += 1
        if matched_count == 2:
            fk_sc.append(fk_line)
    fk_sc = list(set(fk_sc))

    # (B) 收集只跟 sc_tables 有關的 DDL
    ddl_sc = []
    for ddl_line in split_ddl:
        for sc_table in sc_tables:
            if f" {sc_table}(" in ddl_line.lower():
                ddl_sc.append(ddl_line)
    final_ddl = ";\n".join(ddl_sc) + '.'

    # (C) 連接資料庫，拿 sc_tables 的前三筆資料
    mydb = sqlite3.connect(f"data/spider/test_database/{db}/{db}.sqlite")
    cur = mydb.cursor()

    simplified_ddl_data = []
    for table in sc_tables:
        try:
            cur.execute(f"SELECT * FROM `{table}`")
            col_name_list = [desc[0] for desc in cur.description]
            db_data_all = []
            for _ in range(3):
                db_data_all.append(cur.fetchone())

            row_info = ""
            for idx, column_data in enumerate(col_name_list):
                try:
                    row_info += f"{column_data}[{db_data_all[0][idx]},{db_data_all[1][idx]},{db_data_all[2][idx]}],"
                except:
                    pass
            simplified_ddl_data.append(f"{table}({row_info[:-1]})")
        except:
            pass

    ddls_data = "# " + ";\n# ".join(simplified_ddl_data) + ";\n"

    # foreign key
    fk_info = ""
    if len(fk_sc) > 0:
        for fk_line in fk_sc:
            fk_info += f'# {fk_line};\n'
        fk_info = (
            "\n### Foreign key information of Sqlite SQL tables, used for table joins: \n"
            "#\n" + fk_info[:-2]
        )

    question = sample['question']
    if fk_info:
        prompt = (
            f"{task_prefix()}\n"
            "### Sqlite SQL tables, with their properties:\n#\n"
            f"{final_ddl}\n#\n"
            "# ### Here are some data information about database references.\n"
            "# #\n"
            f"{ddls_data}#{fk_info}\n"
            "# #\n"
            f"### Final Question: {question}\n"
            "### SQL: "
        )
    else:
        prompt = (
            f"{task_prefix()}### Sqlite SQL tables, with their properties:\n#\n"
            f"{final_ddl}\n#\n"
            "### Here are some data information about database references.\n#\n"
            f"{ddls_data}#\n"
            f"### Final Question: {question}\n"
            "### SQL: "
        )
    return prompt

def prompt_generation(sample,dataset, kshot, select_type, sl, n,path_generate):
    # 1) 讀取資料
    if not sl:
        input_data = gen_ppl_from_json(dataset)
    else:
        with open(args.dataset, 'r', encoding='utf-8') as f:
            input_data = json.load(f)

    if kshot != 0:
        examples_libary = get_examples_ins(select_type)
        print(f"select type: {select_type}, k shot: {kshot}")
    else:
        examples_libary = None

    # 2) 依照參數生成 prompt
    all_prompts = []
    prompt_path = os.path.join(path_generate, "prompts.txt")
    with open(prompt_path, "w", encoding='utf-8') as f_out:
        # 3) 寫出 prompts 到檔案
        for i, sample in enumerate(input_data[:n]):
            # schema linking or not
            if args.sl:
                prompt_target = formatting_prompt_sl(sample)
            else:
                prompt_target = formatting_prompt(sample)

            # 若要 few-shot
            if examples_libary and kshot != 0:
                examples = examples_libary.get_examples(sample, kshot, cross_domain=False)
                prompt_example = [format_example(exm) for exm in examples]
                prefix = get_example_prefix()
                final_prompt = prefix + "\n\n".join(prompt_example + [prompt_target])
            else:
                final_prompt = prompt_target

            all_prompts.append(final_prompt)
            f_out.write(final_prompt + "\n\n\n\n")

        print(f"Prompts generated and saved to: {prompt_path}")

    


if __name__ == '__main__':

    # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser()

    # 添加命令行选项
    # parser.add_argument("--model", type=str, default="puyuapi") # codellamaapi, puyuapi, llamaapi, sqlcoderapi, vicunaapi, gptapi
    # parser.add_argument("--model_version", type=str, default="none",
    #                 help="Which GPT version to use with gptapi? Options: o1-preview, gpt-4, gpt-4o")
    parser.add_argument("--dataset", type=str, default="ppl_dev.json")
    # parser.add_argument("--out_file", type=str, default="raw.txt")
    parser.add_argument("--kshot", type=int, default=3)
    parser.add_argument("--pool", type=int, default=1)
    parser.add_argument("--sl", action="store_true")
    parser.add_argument("--select_type", type=str, default="Euclidean_mask")
    # parser.add_argument("--max_seq_len", type=int, default=2048, help="The maximal length that LLM takes") # Larger lengths may include more context but risk exceeding model limits.
    # parser.add_argument("--max_ans_len", type=int, default=200, help="The maximal length that an answer takes") # Sets the maximum token length for the LLM-generated answer.
    parser.add_argument("--n", type=int, default=5, help="Size of self-consistent set") # 自洽集的大小
    # 解析命令行参数
    args = parser.parse_args()
    # Construct the log directory and file path
    log_dir = os.path.join(os.path.dirname(__file__), "logs")
    print(log_dir)
    log_file_path = os.path.join(log_dir, f"{args.dataset.upper()}-{args.kshot}_SHOT_{args.select_type}_{args.n}.log")
    print(log_file_path)
    # Create the logs directory if it does not exist
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    logging.basicConfig(filename=log_file_path,
                        level=logging.INFO,
                        filemode='w')
    logger = logging.getLogger()
    if args.sl == False:
        input_data = gen_ppl_from_json(args.dataset)
    else:
        input_data = json.load(open(args.dataset, 'r'))
    path_generate = f"data/process/{args.dataset.upper()}-{args.kshot}_SHOT_{args.select_type}_{args.n}"
    os.makedirs(path_generate, exist_ok=True)
    json.dump(input_data, open(os.path.join(path_generate, "questions.json"), "w"), indent=4)
    print(f"Input data has been saved to {path_generate}")
    print("schema linking: ", args.sl)
    print(args.dataset)
    prompt_generation(input_data, args.dataset,args.kshot, args.select_type, args.sl, args.n,path_generate)

"""
python src/sources/sql_gen/prompt_gen.py \
  --dataset ppl_dev.json \
  --n 1034 \
  --kshot 9 \
  --select_type Euclidean_mask \

"""
    


# python src/sources/sql_gen/prompt_gen.py --kshot 3 --pool 1 --select_type Euclidean_mask_select --n 1034

