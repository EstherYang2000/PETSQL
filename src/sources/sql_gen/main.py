import json, re
import os, sys
import torch.multiprocessing as mp
from transformers import pipeline
proj_dir = os.path.dirname(os.path.dirname(__file__))
# print(proj_dir)
sys.path.append(proj_dir)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from llms import CodeLlama, Puyu, Llama2, SQLCoder, vicuna, GPT
from tqdm import tqdm
import argparse
from get_example_modules import get_examples_ins
from data_preprocess import gen_ppl_from_json
import torch

import logging, sqlite3
# Global variable for the model pipeline
model_pipeline = None


def get_example_prefix():
    return "### Some example pairs of question and corresponding SQL query are provided based on similar problems:\n\n"

def prompt_prefix():
    return "### Let's think step by step. Complete sqlite SQL query only and with no explanation\n"
def task_prefix():
    return "### Your task: \n Answer the final question below by providing **only** the final SQLite SQL query syntax without commentary and explanation.  You must minimize SQL execution time while ensuring correctness.\n"
def format_example(example: dict):
    template_qa = "### {}\n{}"
    return template_qa.format(example['question'], example['gold_sql'])


def formatting_prompt(sample):
    question = sample['question']
    ddls = sample['simplified_ddl']
    db = sample['db']
    # 动态加载前三行数据
    simplified_ddl_data = []
    # 读取数据库
    mydb = sqlite3.connect(
        fr"data/spider/test_database/{db}/{db}.sqlite")  # 链接数据库
    cur = mydb.cursor()
    # 表
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    Tables = cur.fetchall()  # Tables 为元组列表
    for table in Tables:
        # 列
        cur.execute(f"select * from `{table[0]}`")
        col_name_list = [tuple[0] for tuple in cur.description]
        db_data_all = []
        # 获取前三行数据
        for i in range(3):
            db_data_all.append(cur.fetchone())
        # ddls_data
        test = ""
        for idx, column_data in enumerate(col_name_list):
            # print(list(db_data_all[2])[idx])
            try:
                test += f"{column_data}[{list(db_data_all[0])[idx]},{list(db_data_all[1])[idx]},{list(db_data_all[2])[idx]}],"
            except:
                test = test
        simplified_ddl_data.append(f"{table[0]}({test[:-1]})")
    ddls_data = "# " + ";\n# ".join(simplified_ddl_data) + ";\n"
    foreign_key = ""
    for foreign_key_data in sample["foreign_key"][0].split("\n"):
        foreign_key += f'# {foreign_key_data};\n'
    foreign_key = foreign_key[:-2]
    # evidence = "".join(sample['gt_evidence'])
    prompt = f'''{task_prefix()}
    ### Sqlite SQL tables, with their properties:\n#\n{ddls}#\n
    # ### Here are some data information about database references.\n
    # #\n{ddls_data}#\n### Foreign key information of Sqlite SQL tables, used for table joins: \n
    # #\n{foreign_key}\n#\n
    # ### Final Question: {question}\n
    # ### SQL: '''


    return prompt


def formatting_prompt_sl(sample):
    linked_tables = [i.lower() for i in sample['linked_tables_gpt']]
    tbs = []
    for tb in sample['simplified_ddl'].split("\n"):
        t = tb.split("(")[0].strip("#").strip()
        if t.lower() in linked_tables:
            tbs.append(t)
    sc_tables = tbs
    ddl = sample['simplified_ddl'][:-2]
    split_ddl = ddl.split(";\n")
    fk_all = sample["foreign_key"][0].split("\n")
    ddl_sc = []
    fk_sc = []
    db = sample['db']
    # 动态加载前三行数据
    simplified_ddl_data = []
    # 读取数据库
    mydb = sqlite3.connect(
        fr"data/spider/test_database/{db}/{db}.sqlite")  # 链接数据库
    cur = mydb.cursor()
    # 外键
    for fk_test in fk_all:
        num = 0
        for tab in sc_tables:
            if str(" " + tab + "(").lower() in " " + str(fk_test).lower():
                num += 1
        if num == 2:
            fk_sc.append(fk_test)
    fk_sc = list(set(fk_sc))
    for table in sc_tables:
        # ddl
        for ddl_test in split_ddl:
            if str(" " + table + "(").lower() in str(ddl_test).lower():
                ddl_sc.append(ddl_test)
        # 前三行数据
        try:
            cur.execute(f"select * from `{table}`")
            col_name_list = [tuple[0] for tuple in cur.description]
            db_data_all = []
            # 获取前三行数据
            for i in range(3):
                db_data_all.append(cur.fetchone())
            # ddls_data
            test = ""
            for idx, column_data in enumerate(col_name_list):
                try:
                    test += f"{column_data}[{list(db_data_all[0])[idx]},{list(db_data_all[1])[idx]},{list(db_data_all[2])[idx]}],"
                except:
                    test = test
            simplified_ddl_data.append(f"{table}({test[:-1]})")
        except:
            print()
    # res_ddl = []
    # tables = []
    # for test in ddl_sc:
    #     tables.append(test.split("(")[0].replace("# ",""))
    # for one_ddl in split_ddl:
    #     hit = 0
    #     for one_table in linked_tables:
    #         if f" {one_table.lower()}(" in one_ddl.lower():
    #             hit = 1
    #     if hit:
    #         res_ddl.append(one_ddl)
    ddl = ";\n".join(ddl_sc) + '.'
    ddls_data = "# " + ";\n# ".join(simplified_ddl_data) + ";\n"
    foreign_key = ""
    if len(fk_sc) > 0:
        for foreign_key_data in fk_sc:
            foreign_key += f'# {foreign_key_data};\n'
        foreign_key = "\n### Foreign key information of Sqlite SQL tables, used for table joins: \n#\n" + foreign_key[:-2]
    else:
        foreign_key = ""

    # prompt=f'''### Answer the question by sqlite SQL query only and with no explanation\n### Sqlite SQL tables, with their properties:\n{ddl}\n### Question: {sample['question']}\n### SQL: '''
    if foreign_key:
        prompt = f'''{task_prefix()}
### Sqlite SQL tables, with their properties:\n#\n{ddl}\n#\n
# ### Here are some data information about database references.\n
# #\n{ddls_data}#{foreign_key}\n
# #\n
# ### Final Question: {sample['question']}\n
# ### SQL: '''
    else:
        prompt = f'''{task_prefix()}### Sqlite SQL tables, with their properties:\n#\n{ddl}\n#\n
### Here are some data information about database references.\n#\n{ddls_data}#\n### Final Question: {sample['question']}\n### SQL: '''
    return prompt

def initialize_model():
    """
    Initializes the model pipeline in the main process.
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
    Worker function to generate a response for a given prompt.
    """
    global model_pipeline
    if model_pipeline is None:
        raise RuntimeError("Model pipeline is not initialized.")
    response = model_pipeline(prompt, max_length=50, num_return_sequences=1)
    return response[0]["generated_text"]
def run_sql_generation(model,
                       input_data,
                       out_file,
                       k_shot=0,
                       select_type="Euclidean_mask",
                       pool_num=1,
                       sl=False,
                       n=5,
                       gpt_version="o1-preview"):

    domain = False

    # load_libray
    if k_shot != 0:
        examples_libary = get_examples_ins(select_type)
        print(f"select type: {select_type}, k shot: {k_shot}")
    # read file

    if model == "codellamaapi":
        llmapi = CodeLlama(model_name="meta-llama/CodeLlama-7b-hf",max_memory={"cpu": "4GiB", 0: "22GiB"})
    elif model == "puyuapi":
        llmapi = Puyu()
    elif model == "llamaapi":
        # Llama3-OGSQL-8B
        # meta-llama/Llama-3.2-1B
        # OneGate/Llama3-OGSQL-FT-8B
        # bstraehle/Meta-Llama-3.1-8B-text-to-sql
        # ruslanmv/Meta-Llama-3.1-8B-Text-to-SQL
        # cssupport/t5-small-awesome-text-to-sql
        llmapi = Llama2(model_name="ruslanmv/Meta-Llama-3.1-8B-Text-to-SQL",max_memory={"cpu": "4GiB", 0: "22GiB"})  # Example memory allocation
    elif model == "sqlcoderapi":
        llmapi = SQLCoder()
    elif model == "vicunaapi":
        llmapi = vicuna()
    elif model == "gptapi":
        llmapi = GPT(model=args.gpt_version)
    else:
        raise Exception("no llm selected!")

    all_prompts = []
    # get all prompts for parallel
    print('Generating ...')
    output_file = "generated_prompts.txt"
    # Open the file in write mode
    with open(output_file, "w") as file:
        for i, sample in enumerate(input_data[:n]):
            prompt_target = formatting_prompt(sample) if not sl else formatting_prompt_sl(sample)

            if k_shot != 0:
                examples = examples_libary.get_examples(sample,
                                                        k_shot,
                                                        cross_domain=domain)
                prompt_example = [format_example(exm) for exm in examples]
                prompt = get_example_prefix() + "\n\n".join(prompt_example +
                                                            [prompt_target])
            else:
                prompt = prompt_target
            # logger.info(prompt)
            all_prompts.append(prompt)
            # Write the prompt to the file
            file.write(prompt + "\n\n")  # Add a newline for separation
    
    print('Generated!')
    result = []
    global model_pipeline
    if model == "gptapi":
        # Sequential processing
        result = [llmapi(prompt) for prompt in all_prompts]
    elif model == "llamaapi":
        batch_responses = llmapi.generate_batch(
            all_prompts, temperature=0.2,
            top_p=0.2,
            max_new_tokens=150,
            repetition_penalty=1.2,
            do_sample=True,
            )  # 一旦產生到此片段就停止
        print("Batch Responses:")
        for i, res in enumerate(batch_responses):
            print(f"{i + 1}. {res}")
            result.append(res)
    elif model == "codellamaapi":
        batch_responses = llmapi.generate_batch(
            all_prompts,
            temperature=0.4,
            top_p=0.9,
            max_new_tokens=128,
            repetition_penalty=1.05,
            do_sample=True,
            num_beams=1
        )
        print("Batch Responses:")
        for i, res in enumerate(batch_responses):
            print(f"{i + 1}. {res}")
            result.append(res)
    else:
        # # Parallel processing
        # mp.set_start_method("spawn", force=True)
        initialize_model()  # Initialize model in the main process

        # with mp.Pool(pool_num) as pool:
        #     result = pool.map(llmapi, all_prompts)
        # batch_responses = llmapi.generate_batch(all_prompts, max_new_tokens=128)
        # print("Batch Responses:")
        # for i, res in enumerate(batch_responses):
        #     print(f"{i + 1}. {res}")
        #     result.append(res)
        

    # Combine results
    combined_result = '\n'.join(result[i].replace("\n", " ") for i in range(len(result)))
    # Write logs
    with open(log_file_path, 'w', encoding='utf-8') as log_file:
        print(f"Logging prompts to: {log_file_path}")
        log_file.write("\n".join(all_prompts) + '\n')

    # Write output
    with open(args.out_file, 'w', encoding='utf-8') as output_file:
        print(f"Writing results to: {args.out_file}")
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
