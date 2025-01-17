import re, os
import sqlparse
import argparse

# Set up argument parsing for command-line inputs
parser = argparse.ArgumentParser()
parser.add_argument("--llm", default='codellama')  # Specify the LLM type (default: codellama)
parser.add_argument("--file", default='')  # Path to the input file containing SQL queries
parser.add_argument("--output", default="")  # Path to the output file for processed queries


args = parser.parse_args()

llm = args.llm

# Function to process duplication or remove comments in a SQL query
def process_duplication(sql):
    sql = sql.strip().split("#")[0]
    return sql
# Function to extract SQL queries based on the LLM type
def extract_sql(query, llm='sensechat'):
    def clean_query(sql):
        """Helper function to clean up SQL syntax."""
        return sql.strip().replace('\n', ' ').replace('`', '').strip()

    if llm == "sensechat":
        # Remove specific markers for sensechat and clean query
        query = query.replace("### SQL:", "").replace("###", "").replace("#", "")
        if '```SQL' in query or '```sql' in query:
            # Extract the SQL query from the code block using regex
            try:
                sql = re.findall(r"```(?:SQL|sql)(.*?)```", query.replace('\n', ' '))[0]
                return clean_query(sql)
            except IndexError:
                # If no closing backticks, extract until the end
                sql = re.findall(r"```(?:SQL|sql)(.*?)", query.replace('\n', ' '))[0]
                return clean_query(sql)
        elif ';' in query:
            # Find the query ending with a semicolon
            try:
                sql = re.findall(r'(.*?);', query.replace('\n', ' '))[0]
                return clean_query(sql)
            except IndexError:
                pass
        else:
            # Default case: clean and return query
            return clean_query(query)
    elif llm in {"codellama", "sqlcoder"}:
        # For codellama/sqlcoder, find the query ending with a semicolon
        if ';' in query:
            sql = re.findall(r'(.*?);', query.replace('\n', ' '))[0]
            return clean_query(sql)
        else:
            return "SELECT " + clean_query(query)
    elif llm == "puyu":
        # For puyu, clean up '#' and ensure SELECT at the beginning
        query = query.replace("#", "")
        return clean_query(query) if query.lower().strip().startswith("select") else "SELECT " + clean_query(query)
    elif llm == 'gpt':
        # For gpt, process duplication and ensure SELECT at the beginning
        sql = " ".join(query.replace("\n", " ").split())
        sql = process_duplication(sql)
        return clean_query(sql) if sql.lower().strip().startswith("select") else "SELECT " + clean_query(sql)
    else:
        # Default case: return the query as-is
        return clean_query(query)


def extract_sql_2(query, llm='codellama'):
    """
    Extract ONLY the final SQL from the raw output text, removing prompts/examples/explanations.
    """

    # 全域先把換行統一為空白，免得後續正則跨行不便
    query = query.replace("\n", " ")

    # 針對不同 LLM 做不同處理
    if llm == "llama":
        # 1) 先用正則去除「範例區段」到 '### SQL:' 之間的內容
        #    假設示例都是以 "### Some example pairs" 開頭
        query = re.sub(
            r'(### Some example pairs.*?)(?=### SQL:)',
            '',
            query,
            flags=re.DOTALL
        )
        # 2) 再切割：只留 '### SQL:' 後面的內容
        parts = re.split(r'### SQL:\s*', query)
        if len(parts) > 1:
            query = parts[-1].strip()
        else:
            query = query.strip()

        # 3) 去掉多餘的 # 及 ###（若有）
        query = re.sub(r'#+', '', query).strip()

        # 4) 找是否有分號，若有只取第一個分號前
        if ';' in query:
            query = re.findall(r'(.*?);', query)[0].strip()

        # 5) 確保以 SELECT 開頭
        if not query.lower().startswith("select"):
            query = "SELECT " + query

        # 6) 用 sqlparse 格式化（可視情況 reindent）
        return sqlparse.format(query, reindent=False)

    elif llm == "codellama":
        # 先同樣去除「範例區段」
        query = re.sub(
            r'(### Some example pairs.*?)(?=### SQL:)',
            '',
            query,
            flags=re.DOTALL
        )
        # 切割：只留 '### SQL:' 後面的內容
        parts = re.split(r'### SQL:\s*', query)
        if len(parts) > 1:
            query = parts[-1].strip()
        else:
            query = query.strip()

        # 移除多餘的 #、註解
        query = re.sub(r'#+', '', query).strip()

        # 接著沿用原本對 codellama 的邏輯
        # 先看看是否有分號
        if ';' in query:
            res = re.findall(r'(.*?);', query)[0].strip()
            res = res.replace('# ', '').replace('##', '')
            if res.lower().startswith("select"):
                query = res
            else:
                query = "SELECT " + res
        else:
            # 處理沒有分號的情況
            # 如果 output 中可能還有 '###'，先 split 掉
            res = query.split("###")[0].replace('# ', '').replace('##', '').strip()
            if res.lower().startswith("select"):
                query = res
            else:
                query = "SELECT " + res

        # 最後做一次格式化
        return sqlparse.format(query, reindent=False)

    elif llm == "sqlcoder":
        # 與 codellama 類似，可直接套用 codellama 的分支或自行調整
        if ';' in query:
            ...
        # 這裡省略，保留原邏輯或自行改寫
        return ...

    elif llm == "puyu":
        # 原本的 puyu 處理
        res = query.replace("#","")
        if res.lower().strip().startswith("select"):
            return res
        else:
            return "SELECT " + res

    elif llm == 'gpt':
        # 原本的 gpt 處理
        sql = " ".join(query.split())
        sql = process_duplication(sql)
        if sql.lower().strip().startswith("select"):
            print(sql)
            return sql 
        elif sql.startswith(" "):
            return "SELECT" + sql
        else:
            return "SELECT " + sql
        
    else:
        # Default case: return as is
        
        return query
def extract_sql_from_text(text):
    sql_pattern = text.replace("\n", " ").replace("ി", " ").split('~~')
    return sql_pattern
def extract_sql_3(text: str, from_4o: bool = False) -> str:
    """
    從文字中擷取出完整的 SQL 語句。
    1) 如果 from_4o=True，則優先嘗試在 ```sql ... ``` 區塊中擷取
    2) 若找不到三引號區塊，或 from_4o=False，則再嘗試從 'SELECT' 到第一個 ';'
    3) 如果都找不到，就回傳原文字或空字串
    """
    # 如果是 4o，優先嘗試找 ```sql ... ```
    print(text)
    if from_4o:
        pattern_4o_block = r'```sql\s*(.*?)```'
        blocks_4o = re.findall(pattern_4o_block, text, flags=re.DOTALL | re.IGNORECASE)
        if blocks_4o:
            # 若找到多塊，只取第一塊，也可自己改成合併
            return blocks_4o[0].strip()
        # 若沒找到，繼續 fallback 從 SELECT...; 抽取

    # 通用 fallback：從 SELECT 開始到第一個 ';'
    pattern_select = r'(SELECT.*?;)'
    match_select = re.search(pattern_select, text, flags=re.IGNORECASE | re.DOTALL)
    if match_select:
        return match_select.group(1).strip()

    # 若真的沒找到任何符合，就回傳原始文本或空字串都行
    return text.strip()


if __name__ == "__main__":
    with open(args.file, 'r', encoding='utf-8') as file:
        content = file.readlines()

    print(len(content))
    # mid = extract_sql_from_text("\n".join(content))
    extracted_query = [sqlparse.format(extract_sql(q, "sensechat").strip(), reindent=False) for q in content]

    # extracted_query = [sqlparse.format(extract_sql(q, "sensechat").strip(), reindent=False) for q in extract_sql_from_text("\n".join(content))]
    # extracted_query = [
    #     sqlparse.format(extract_sql(q, "sensechat").strip(), reindent=False) + ";" 
    #     if not sqlparse.format(extract_sql(q, "sensechat").strip(), reindent=False).endswith(";")
    #     else sqlparse.format(extract_sql(q, "sensechat").strip(), reindent=False)
    #     for q in extract_sql_from_text("\n".join(content))
    # ]
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as file:
            file.write('\n'.join(extracted_query))
    else:
        with open(args.file.replace(".txt", "_out.txt") , 'w', encoding='utf-8') as file:
            file.write('\n'.join(extracted_query))


#python src/sources/post_process.py --file src/sources/raw_codellama.txt --output src/sources/output_codellama.txt --llm codellama
#python src/sources/post_process.py --file src/sources/raw_gpt.txt --output src/sources/output_gpt.txt --llm gpt
# python src/sources/post_process.py --file data/process/PPL_DEV.JSON-9_SHOT_Euclidean_mask_1034/phind-codellam_api.txt --output data/process/PPL_DEV.JSON-9_SHOT_Euclidean_mask_1034/phind-codellam_api_output.txt --llm llama