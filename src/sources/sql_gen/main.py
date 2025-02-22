import json
import logging
import os
import re
import sys
import argparse
import sqlite3
import torch
import torch.multiprocessing as mp

from transformers import pipeline
from tqdm import tqdm

# 根據專案結構做路徑調整
proj_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(proj_dir)

# 避免 Tokenizers 產生多工警告
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 專案內部匯入
from llms import CodeLlama, Puyu, Llama2, SQLCoder, vicuna, GPT
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

######################################################
# Model initialization & inference
######################################################
def initialize_model():
    """
    Initializes the HF pipeline-based model in the main process.
    """
    global model_pipeline
    if model_pipeline is None:
        print("Initializing model...")
        model_pipeline = pipeline(
            "text-generation",
            model="meta-llama/Llama-3.2-1B",
            device_map="auto",
            torch_dtype=torch.float16,
            max_length=4096,
        )
        print("Model initialized successfully!")


def llmapi(prompt):
    """
    Worker function to generate a response for a given prompt using model_pipeline.
    """
    global model_pipeline
    if model_pipeline is None:
        raise RuntimeError("Model pipeline is not initialized.")
    response = model_pipeline(prompt, max_length=50, num_return_sequences=1)
    return response[0]["generated_text"]
######################################################
# Main function for SQL generation
######################################################
def run_sql_generation(model,
                       input_data,
                       out_file,
                       k_shot=0,
                       select_type="Euclidean_mask",
                       pool_num=1,
                       sl=False,
                       n=5,
                       gpt_version="o1-preview"):
    """
    主函式：根據使用者指定的 model 產生 SQL 查詢，並將輸出寫入 out_file。

    :param model: str, 指定要使用哪個模型 (codellamaapi, puyuapi, llamaapi, sqlcoderapi, vicunaapi, gptapi)
    :param input_data: list, 包含多筆需要生成 SQL 的問題資料
    :param out_file: str, 輸出檔案路徑
    :param k_shot: int, few-shot 數量
    :param select_type: str, 用什麼方式選取範例 (ex. "Euclidean_mask")
    :param pool_num: int, 多進程並行數量
    :param sl: bool, 是否啟用 schema linking
    :param n: int, 只處理前 n 筆資料
    :param gpt_version: str, 使用 GPT 時的版本選擇
    """
    domain = False

    # 讀取 K-shot 示例庫 (若 k_shot != 0)
    if k_shot != 0:
        examples_libary = get_examples_ins(select_type)
        print(f"select type: {select_type}, k shot: {k_shot}")

    # 初始化 LLM API
    if model == "codellamaapi":
        llm_instance = CodeLlama(
            model_name="meta-llama/CodeLlama-7b-hf",
            max_memory={"cpu": "4GiB", 0: "22GiB"}
        )
    elif model == "puyuapi":
        llm_instance = Puyu()
    elif model == "llamaapi":
        # Llama3-OGSQL-8B
        # meta-llama/Llama-3.2-1B
        # OneGate/Llama3-OGSQL-FT-8B
        # bstraehle/Meta-Llama-3.1-8B-text-to-sql
        # ruslanmv/Meta-Llama-3.1-8B-Text-to-SQL
        # cssupport/t5-small-awesome-text-to-sql
        llm_instance = Llama2(
            model_name="ruslanmv/Meta-Llama-3.1-8B-Text-to-SQL",
            max_memory={"cpu": "4GiB", 0: "22GiB"}
        )
    elif model == "sqlcoderapi":
        llm_instance = SQLCoder()
    elif model == "vicunaapi":
        llm_instance = vicuna()
    elif model == "gptapi":
        llm_instance = GPT(model=gpt_version)
    else:
        raise Exception("No LLM selected!")

    # 產生 prompts
    all_prompts = []
    print('Generating prompts...')
    output_file = "generated_prompts.txt"

    with open(output_file, "w", encoding='utf-8') as f_out:
        for i, sample in enumerate(input_data[:n]):
            # 決定要用哪種 prompt 格式 (sl: schema linking or not)
            prompt_target = formatting_prompt_sl(sample) if sl else formatting_prompt(sample)

            if k_shot != 0:
                # 擷取 k_shot 筆相似案例
                examples = examples_libary.get_examples(sample, k_shot, cross_domain=domain)
                prompt_example = [format_example(exm) for exm in examples]
                prompt = get_example_prefix() + "\n\n".join(prompt_example + [prompt_target])
            else:
                prompt = prompt_target

            all_prompts.append(prompt)
            f_out.write(prompt + "\n\n")
    
    print('Prompts generated!')

    # 進行推理
    results = []
    if model == "gptapi":
        # GPTAPI => 順序推理 (可能是OpenAI介面)
        results = [llm_instance(p) for p in all_prompts]

    elif model in ["llamaapi", "codellamaapi"]:
        # 支援 batch generation
        if model == "llamaapi":
            batch_responses = llm_instance.generate_batch(
                all_prompts,
                temperature=0.2,
                top_p=0.2,
                max_new_tokens=150,
                repetition_penalty=1.2,
                do_sample=True,
            )
        else:  # codellamaapi
            batch_responses = llm_instance.generate_batch(
                all_prompts,
                temperature=0.4,
                top_p=0.9,
                max_new_tokens=128,
                repetition_penalty=1.05,
                do_sample=True,
                num_beams=1
            )

        print("Batch Responses:")
        for i, res in enumerate(batch_responses, start=1):
            print(f"{i}. {res}")
            results.append(res)

    else:
        # 其餘的情況 => 預設 pipeline 方式 (平行 or 單線程)
        initialize_model()
        # 若要平行處理可啟用 mp.Pool(...)
        # 這裡示範直接一次性 batch
        # (或者改成: results = [llmapi(p) for p in all_prompts])
        # 視實際需求調整

        # print("Parallel processing with mp.Pool... (optional)")
        # mp.set_start_method("spawn", force=True)
        # with mp.Pool(pool_num) as pool:
        #     results = pool.map(llmapi, all_prompts)

        # 這裡示範簡易做法:
        # results = [llmapi(p) for p in all_prompts]

        # 或若 llm_instance 具備 batch 方法:
        #   batch_responses = llm_instance.generate_batch(all_prompts, max_new_tokens=128)
        #   for res in batch_responses:
        #       results.append(res)
        pass

    # 若前面未真正產生 results, 可以在此處補上:
    if not results:
        # 如果你真的要呼叫 pipeline 來生成，可以這樣：
        results = [llmapi(p) for p in all_prompts]

    # 合併結果成字串
    combined_result = '\n'.join(res.replace("\n", " ") for res in results)

    # 紀錄 prompts (log)
    with open(log_file_path, 'w', encoding='utf-8') as log_file:
        print(f"Logging prompts to: {log_file_path}")
        log_file.write("\n".join(all_prompts) + '\n')

    # 輸出最終結果
    with open(out_file, 'w', encoding='utf-8') as output_file:
        print(f"Writing results to: {out_file}")
        output_file.write(combined_result + '\n')


if __name__ == '__main__':

    # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser()

    # 添加命令行选项
    parser.add_argument("--model", type=str, default="puyuapi") # codellamaapi, puyuapi, llamaapi, sqlcoderapi, vicunaapi, gptapi
    parser.add_argument("--gpt_version", type=str, default="o1-preview",
                    help="Which GPT version to use with gptapi? Options: o1-preview, gpt-4, gpt-4o")
    parser.add_argument("--dataset", type=str, default="ppl_dev.json")
    parser.add_argument("--out_file", type=str, default="raw.txt")
    parser.add_argument("--kshot", type=int, default=3)
    parser.add_argument("--pool", type=int, default=1)
    parser.add_argument("--sl", action="store_true")
    parser.add_argument("--select_type", type=str, default="Euclidean_mask")
    # parser.add_argument("--max_seq_len", type=int, default=2048, help="The maximal length that LLM takes") # Larger lengths may include more context but risk exceeding model limits.
    # parser.add_argument("--max_ans_len", type=int, default=200, help="The maximal length that an answer takes") # Sets the maximum token length for the LLM-generated answer.
    parser.add_argument("--n", type=int, default=5, help="Size of self-consistent set") # 自洽集的大小

    # python src/sources/sql_gen/main.py --model "llamaapi" --kshot 9 --pool 1  --out_file src/sources/raw.txt --select_type Euclidean_mask

    # 解析命令行参数
    args = parser.parse_args()
    # Construct the log directory and file path
    log_dir = os.path.join(os.path.dirname(__file__), "logs")
    log_file_path = os.path.join(log_dir, f"{args.model}_{args.select_type}.log")

    # Create the logs directory if it does not exist
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    logging.basicConfig(filename=log_file_path,
                        level=logging.INFO,
                        filemode='w')
    logger = logging.getLogger()
    if args.sl == False:
        input_data = gen_ppl_from_json(args.dataset, args.model[:-3])
    else:
        input_data = json.load(open(args.dataset, 'r'))
    # Specify the output file path
    # output_file_path = "output_data.json"  # Change the file name/path as needed
    # Save `input_data` to a JSON file
    # with open(output_file_path, 'w') as outfile:
    #     json.dump(input_data, outfile, indent=4)  # indent=4 for readable formatting
    
    path_generate = f"data/process/{args.dataset.upper()}-{args.kshot}_SHOT_{args.select_type}_{args.n}"
    os.makedirs(path_generate, exist_ok=True)
    json.dump(input_data, open(os.path.join(path_generate, "questions.json"), "w"), indent=4)
    print(f"Input data has been saved to {path_generate}")
    print("schema linking: ", args.sl)
    print(args.dataset)
    run_sql_generation(args.model, input_data, args.out_file, args.kshot,
                       args.select_type, args.pool, args.sl, args.n,args.gpt_version)
