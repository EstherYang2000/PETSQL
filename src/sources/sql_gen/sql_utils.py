import sqlparse
from post_process import extract_sql
from llms import OllamaChat, GPT, GroqChat,TogetherChat,ClaudeChat,GoogleGeminiChat
from time import sleep
from tqdm import tqdm
import json
import os
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

def run_sql_generation(model,
                       path_generate,
                       prompts,
                       out_file,
                       model_version="gpt-4",
                       start_num_prompts = 0,
                       end_num_prompts=1034,
                       call_mode="write",
                       batch_size=1,
                       n_samples=5,
                       question_index=None
                       ):
    
    # 2) 初始化對應 model (根據 model 參數)
    llm_instance = None
    if model == "codellamaapi":
        if model_version == "34b-instruct":

            llm_instance = OllamaChat(model="codellama:34b-instruct")
        elif model_version == "70b":
            llm_instance = OllamaChat(model="codellama:70b")
    elif model == "phind-codellamaapi":
        llm_instance = OllamaChat(model="phind-codellama")
    elif model == "mistralapi":
        if model_version == "small_24b":
            llm_instance = OllamaChat(model="mistral-small:24b")
    elif model == "qwen_api":
        if model_version == "32b-instruct-fp16":
            llm_instance = TogetherChat(model="Qwen/Qwen2.5-Coder-32B-Instruct")
            # llm_instance = OllamaChat(model="qwen2.5-coder:32b-instruct-fp16")
        elif model_version == "72b-instruct-q6_K":
            llm_instance = OllamaChat(model="qwen2.5:72b-instruct-q6_K")
        elif model_version == "2_5_72b":
            llm_instance = TogetherChat(model="Qwen/Qwen2.5-72B-Instruct-Turbo")
            # llm_instance = OllamaChat(model="qwen2.5:72b")
            # api_key = os.environ["GROQ_API_KEY"]
            # llm_instance = GroqChat(api_key=api_key,model="qwen-2.5-coder-32b")
    elif model == "llamaapi":
        if model_version == "3.3":
            llm_instance = TogetherChat(model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free")
            # llm_instance = OllamaChat(model="llama3.3:latest")
            
            # api_key = os.environ["GROQ_API_KEY"]
            # llm_instance = GroqChat(api_key=api_key,model="llama3-70b-8192")
        elif model_version == "3.3_70b_specdec":
            api_key = os.environ["GROQ_API_KEY"]
            llm_instance = GroqChat(api_key=api_key,model="llama3.3:latest")
        # llm_instance = Llama2(model_name="ruslanmv/Meta-Llama-3.1-8B-Text-to-SQL", max_memory={"cpu": "4GiB", 0: "22GiB"})
    elif model == "gptapi":
        llm_instance = GPT(model=model_version)
    elif model == "claudeaapi":
        api_key = os.environ["SEGMIND_API_KEY"]
        llm_instance = ClaudeChat(api_key=api_key)
    elif model == "googlegeminiapi":
        api_key = os.environ["GOOGLE_API_KEY"]
        llm_instance = GoogleGeminiChat(api_key=api_key)
    elif model == "deepseekapi":
        if model_version == "v2-16b":
            llm_instance = OllamaChat(model="deepseek-coder-v2:16b")
        elif model_version == "r1-32b":
            llm_instance = OllamaChat(model="deepseek-r1:32b")
        elif model_version == "r1_70b":
            llm_instance = OllamaChat(model="deepseek-r1:70b")
        elif model_version == "r1_distill_llama_70b":
            api_key = os.environ["GROQ_API_KEY"]
            llm_instance = GroqChat(api_key=api_key,model="deepseek-r1-distill-llama-70b")
        elif model_version == "33b":
            llm_instance = OllamaChat(model="deepseek-coder:33b")
        elif model_version == "coder-v2:16b":
            llm_instance = OllamaChat(model="deepseek-coder-v2:16b")
        elif model_version == "llm_67b":
            llm_instance = OllamaChat(model="deepseek-llm:67b")
    else:
        # 如果要用 huggingface pipeline 作fallback
        initialize_model()
        llm_instance = None
        
    # if end_num_prompts is not None and start_num_prompts is not None and start_num_prompts >=0 and end_num_prompts > 0:
    #     prompts = prompts[start_num_prompts:end_num_prompts]
    print(f"Loaded {len(prompts)} prompts.")
    print(f"Loaded model = {model}")
    print(f"Processing {len(prompts)} prompts...")
    print(f"Batch size: {batch_size}")
    # 3) 產生結果
    results = []
    
    if llm_instance:
        # 有自訂 generate_batch
        if hasattr(llm_instance, "generate_batch"):

            for i in tqdm(range(0, len(prompts), batch_size), desc="Processing Batches"):
                batch = prompts[i:i + batch_size]
                if model in["deepseekapi" ,"codellamaapi" , "llamaapi" , "phind-codellamaapi" , "qwen_api" ,"mistralapi","claudeaapi","googlegeminiapi"]:
                    if n_samples == 1:
                        batch_responses = llm_instance.generate_batch(batch)
                    else:
                        batch_responses = []
                        # for _ in range(n_samples):
                        batch_responses.extend(llm_instance.generate_batch(batch*n_samples))
                elif model == "gptapi":
                    if n_samples == 1:
                        batch_responses = llm_instance.generate_batch(
                            batch,
                            temperature=0.7,
                            top_p=0.9,
                            max_new_tokens=2048,
                            repetition_penalty=1.05,
                            do_sample=True
                        )
                        print(f"Batch responses: {batch_responses}")
                        print(len(batch_responses))
                    else:
                        batch_responses = []
                        # for _ in range(n_samples):
                        batch_responses.extend(llm_instance.generate_batch(
                            batch*n_samples,
                            temperature=0.2,
                            top_p=0.9,
                            max_new_tokens=2048,
                            repetition_penalty=1.05,
                            do_sample=True
                            ))                        
                results.append(batch_responses)
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
    
    for idx, responses in enumerate(results):
        combined_result.append({
            "prompt_index":  question_index if question_index else start_num_prompts + idx,
            "sql_candidates": [res.replace("\n", " ") for res in responses]
        })
    print(f"Processed {len(combined_result)} prompts.")
    # print(f"First result: {combined_result[0]}")
    
    # print(combined_result)
    # 5) 存成JSON檔
    raw_path = os.path.join(path_generate, out_file)
    print(raw_path)
    if call_mode == "append" and os.path.exists(raw_path):
        with open(raw_path, "r", encoding="utf-8") as f:
            existing_data = json.load(f)
    else:
        existing_data = []

    existing_data.extend(combined_result)

    with open(raw_path, "w", encoding="utf-8") as f_out:
        json.dump(existing_data, f_out, indent=4, ensure_ascii=False)

    print(f"Done. Generated SQLs for {len(results)} prompts. Written to {out_file}")
    return combined_result  # 這裡也可以直接return JSON結構，方便後續其他function接著用


def run_refinement_pipeline(db_path:str,prompt:str,sql_candidates:list,path_generate:str,start_num_prompts:int,model:str,model_version):
    from refine.refinement import refine_sql_candidates  # 避免循環import

    for raw_data in sql_candidates:
        raw_clean = [sqlparse.format(extract_sql(sql, "sensechat").strip(), reindent=False) for sql in raw_data['sql_candidates']]
        raw_data['sql_candidates'] = refine_sql_candidates(prompt, raw_clean, model, path_generate, start_num_prompts, model_version, db_path)
    return sql_candidates

