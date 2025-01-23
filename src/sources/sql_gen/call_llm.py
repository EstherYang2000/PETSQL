"""
call_llm.py

根據已生成的 prompts（如 generated_prompts.txt），呼叫對應 LLM ，
產生最終 SQL 結果，並寫到 out_file。
"""

import os
import argparse
from time import sleep
import torch
from transformers import pipeline
from tqdm import tqdm

# 你的專案內部匯入
from llms import SQLCoder, vicuna, GPT,OllamaChat

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 全域 pipeline (若需 huggingface pipeline)
model_pipeline = None


def initialize_model():
    """
    用 huggingface pipeline 的方式初始化 (若需要)。
    """
    global model_pipeline
    if model_pipeline is None:
        print("Initializing HF pipeline model ...")
        model_pipeline = pipeline(
            "text-generation",
            model="meta-llama/Llama-3.2-1B",
            device_map="auto",
            torch_dtype=torch.float16,
            max_length=4096,
        )
        print("Model pipeline initialized!")


def llmapi(prompt):
    """
    如果你要呼叫 huggingface pipeline 做推理
    """
    global model_pipeline
    if model_pipeline is None:
        raise RuntimeError("Model pipeline not initialized.")
    response = model_pipeline(prompt, max_length=50, num_return_sequences=1)
    return response[0]["generated_text"]

def load_prompts(path_generate,num_prompts=None):
    # 1) 讀取 prompt_file
    prompt_path = os.path.join(path_generate, "prompts.txt")

    with open(prompt_path, "r", encoding='utf-8') as f_in:
        content = f_in.read()

    # 以 "\n\n" 為分隔符，分割出所有 prompt
    all_prompts = content.split("\n\n\n\n")

    # 如果有指定要處理的 prompt 數量，就截斷
    if num_prompts is not None and num_prompts > 0:
        all_prompts = all_prompts[:num_prompts]
    print(f"Loaded {len(all_prompts)} prompts.")
    return all_prompts
def run_sql_generation(model,
                       path_generate,
                       prompts,
                       out_file,
                       pool_num=1,
                       model_version="gpt-4",
                       num_prompts=None,
                       call_mode="write",
                       batch_size=4):
    """
    主函式：根據使用者指定的 model 產生 SQL 查詢，並將輸出寫入 out_file。

    :param model: str, 指定要使用哪個模型 (codellamaapi, puyuapi, llamaapi, sqlcoderapi, vicunaapi, gptapi)
    :param path_generate: str, 已產生之 prompts 檔案路徑 (例如 "generated_prompts.txt")
    :param out_file: str, 輸出檔案路徑
    :param k_shot: int, few-shot 數量 (可視需要保留)
    :param select_type: str, 用什麼方式選取範例 (ex. "Euclidean_mask")
    :param pool_num: int, 多進程並行數量
    :param sl: bool, 是否啟用 schema linking (可視需要保留)
    :param n: int, (可視需要保留，用於其他流程)
    :param gpt_version: str, 使用 GPT 時的版本選擇
    :param num_prompts: int or None, 若指定，表示只處理前 num_prompts 行 prompt
    """
    # # 1) 讀取 prompt_file
    # prompt_path = os.path.join(path_generate, "prompts.txt")

    # with open(prompt_path, "r", encoding='utf-8') as f_in:
    #     content = f_in.read()

    # # 以 "\n\n" 為分隔符，分割出所有 prompt
    # all_prompts = content.split("\n\n\n\n")

    # # 如果有指定要處理的 prompt 數量，就截斷
    # if num_prompts is not None and num_prompts > 0:
    #     all_prompts = all_prompts[:num_prompts]
    # print(f"Loaded {len(all_prompts)} prompts.")
    # print(f"First prompt: {all_prompts[0]}")

    # 2) 初始化對應 model (根據 model 參數)
    if model == "codellamaapi":
        # llm_instance = CodeLlama(model_name="codellama/CodeLlama-34b-Instruct-hf", max_memory={"cpu": "4GiB", 0: "22GiB"})
        # llm_instance = CodeLlama2(
        #     model_name="beneyal/code-llama-7b-spider-qpl-lora",
        #     max_memory={"cpu": "4GiB", 0: "22GiB"},
        # )
        if model_version == "34b-instruct":
            llm_instance = OllamaChat(model="codellama:34b-instruct")
    elif model == "phind-codellamaapi":
        llm_instance = OllamaChat(model="phind-codellama")
    elif model == "qwen2.5-coderaapi":
        llm_instance = OllamaChat(model="qwen2.5-coder:32b-instruct-fp16")
    elif model == "llamaapi":
        llm_instance = OllamaChat(model="llama3.3:latest")
        # llm_instance = Llama2(model_name="ruslanmv/Meta-Llama-3.1-8B-Text-to-SQL", max_memory={"cpu": "4GiB", 0: "22GiB"})
    # elif model == "sqlcoderapi":
    #     llm_instance = SQLCoder()
    # elif model == "vicunaapi":
    #     llm_instance = vicuna()
    elif model == "gptapi":
        llm_instance = GPT(model=model_version)
    elif model == "deepseekapi":
        if model_version == "v2-16b":
            llm_instance = OllamaChat(model="deepseek-coder-v2:16b")
        elif model_version == "r1_70b":
            llm_instance = OllamaChat(model="deepseek-r1:70b")
        # llm_instance = DeepSeek(
        #     model_name="deepseek-ai/deepseek-coder-33b-instruct",
        #     max_memory={
        #         "cpu": "24GiB",
        #     },
        #     torch_dtype=torch.float32,
        # )
    else:
        # 如果要用 huggingface pipeline 作fallback
        initialize_model()
        llm_instance = None
        
    if num_prompts is not None and num_prompts > 0:
        prompts = prompts[:num_prompts]
    print(f"Loaded {len(prompts)} prompts.")
    print(f"Loaded model = {model}")
    print(f"Processing {len(prompts)} prompts...")
    print(f"Batch size: {batch_size}")
    # 3) 產生結果
    results = []
    if llm_instance:
        # 有自訂 generate_batch
        if hasattr(llm_instance, "generate_batch"):
            # batch 推理 (如 llamaapi, codellamaapi ...)
            # batch_responses = llm_instance.generate_batch(
            #     prompts,
            #     temperature=0.2,
            #     top_p=0.9,
            #     max_new_tokens=128,
            #     repetition_penalty=1.05,
            #     do_sample=True
            # )
            # results = batch_responses
            for i in tqdm(range(0, len(prompts), batch_size), desc="Processing Batches"):
                batch = prompts[i:i + batch_size]
                batch_responses = []
                if model == "deepseekapi" or model == "codellamaapi" or model == "llamaapi" or model == "phind-codellamaapi" or model == "qwen2.5-coderaapi":
                    batch_responses = llm_instance.generate_batch(batch)
                elif model == "gptapi":
                    batch_responses = llm_instance.generate_batch(
                        batch,
                        temperature=0.2,
                        top_p=0.9,
                        max_new_tokens=2048,
                        repetition_penalty=1.05,
                        do_sample=True
                    )
                results.extend(batch_responses)
                sleep(1)
        else:
            # 順序生成
            for p in prompts:
                res = llm_instance(p)
                results.append(res)
    else:
        # 代表 fallback 到 huggingface pipeline
        # 直接逐條 llmapi
        for p in prompts:
            r = llmapi(p)
            results.append(r)

    # 4) 後處理並合併
    combined_result = []
    for res in results:
        line = res.replace("\n", " ")
        combined_result.append(line)
    final_text = "\n".join(combined_result)
    raw_path = os.path.join(path_generate, out_file)

    # 5) 寫到 out_file
    if call_mode == "append":
        with open(raw_path,"a" if os.path.exists(raw_path) else "w",encoding="utf-8") as f_out:
            f_out.write(final_text + "\n")
    else:
        with open(raw_path, "w", encoding="utf-8") as f_out:
            f_out.write(final_text + "\n")

    print(f"Done. Generated {len(results)} lines. Written to {out_file}")
    return final_text

if __name__ == '__main__':
    # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser(description="Call LLM on prompts and output results.")
    parser.add_argument("--model", type=str, default="puyuapi",
                        help="Which model to use? codellamaapi, puyuapi, llamaapi, sqlcoderapi, vicunaapi, gptapi")
    parser.add_argument("--model_version", type=str, default="none",
                        help="Which GPT version to use with gptapi? Options: o1-preview, gpt-4, gpt-4o")
    parser.add_argument("--path_generate", type=str,
                        help="Path to the generated raw file.")
    parser.add_argument("--out_file", type=str, default="raw.txt")
    parser.add_argument("--pool", type=int, default=1)
    parser.add_argument("--num_prompts", type=int, default=None,
                        help="Number of prompts to process from the prompt file (if not specified, take all).")
    pqrs = parser.add_argument("--call_mode", type=str, default="write",
                        help="mode to write or append")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="")


    args = parser.parse_args()

    # 根據 dataset 及其他參數組成 path_generate (這裡只示範)
    all_prompts = load_prompts(args.path_generate,num_prompts=args.num_prompts)
    print(len(all_prompts))
    # 執行主程式
    run_sql_generation(
        model=args.model,
        path_generate = args.path_generate,
        prompts = all_prompts,
        out_file=args.out_file,
        pool_num=args.pool,
        model_version=args.model_version,
        num_prompts=args.num_prompts,
        call_mode = args.call_mode,
        batch_size=args.batch_size
    )
    
    # def run_sql_generation(model,
    #                    path_generate,
    #                    prompts,
    #                    out_file,
    #                    pool_num=1,
    #                    model_version="gpt-4",
    #                    num_prompts=None,
    #                    call_mode="write"):
    
    # python src/sources/sql_gen/call_llm.py \
    # --path_generate /home/yyj/Desktop/yyj/thesis/code/PETSQL/data/process/PPL_DEV.JSON-3_SHOT_Euclidean_mask_llamaapi_none_5 \
    # --model llamaapi \
    # --model_version none \
    # --out_file raw.txt \
    # --num_prompts 2
    
    # python src/sources/sql_gen/call_llm.py \
    # --path_generate /home/yyj/Desktop/yyj/thesis/code/PETSQL/data/process/PPL_DEV.JSON-9_SHOT_Euclidean_mask_1034 \
    # --model codellamaapi \
    # --model_version 34b-instruct \
    # --out_file codellama_34b-instruct_api.txt \
    # --num_prompts 1034

