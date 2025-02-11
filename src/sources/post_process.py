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
# def extract_sql(query, llm='sensechat'):
#     def clean_query(sql):
#         """Helper function to clean up SQL syntax."""
#         return sql.strip().replace('\n', ' ').replace('`', '').strip()

import re

def extract_sql(query, llm='sensechat'):
    def clean_query(sql):
            """Helper function to clean up SQL syntax."""
            return sql.strip().replace('\n', ' ').rstrip('.').strip()


    if llm == "sensechat":
        # Handle SenseChat-specific markers and clean the query
        query = query.replace("### SQL:", "").replace("###", "").replace("#", "")
        # Expanded regex patterns for more flexible SQL extraction
        # extraction_patterns = [
        #     # Match SQL within code blocks with various markers
        #     r"```(?:SQL|sql|sqlite)?\s*(.*?)```",
            
        #     # Match SQL after specific markers or sections
        #     r"(?:###\s*SQL:|</think>|<think>)\s*(SELECT\s+.*?(?:;|$))",
        #     r"</think>\s*(?:.*?)(SELECT\s+.*?(?:;|$))",
        #     # r"\*\*(?:SQL Query|Final SQL Query|SQL|Answer):\*\*\s*(\s*SELECT\s+.*?(?:;|$))",
        #     r"\*\*(?:Answer|SQL Query|Final SQL Query|SQL):\*\*\s*(SELECT\s+.*?)(?:\n|$)",

        #     # Match SQL after colon, semicolon, or at the end of text
        #     r":\s*(SELECT\s+.*?(?:;|$))",
        #     # r"\*\*(?:SQL Query|Final SQL Query|SQL):\*\*\s*(\s*SELECT\s+.*?(?:;|$))",


        #     # Match standalone SQL queries
        #     r"(SELECT\s+DISTINCT\s+.*?(?:;|$))",
        #     r"(SELECT\s+.*?(?:;|$))"
        # ]
        extraction_patterns = [
            # More flexible pattern for markdown headers
            r"\*\*(?:Answer|SQL Query|Final SQL Query|SQL):\*\*\s*(SELECT\s+[\s\S]*?)(?:\n|;|$)",

            # Fix for triple-single and triple-double quoted SQL blocks
            r"'''(?:SQL|sql|sqlite)?\s*(SELECT\s+[\s\S]*?)'''",
            r'"""(?:SQL|sql|sqlite)?\s*(SELECT\s+[\s\S]*?)"""',

            # Fix for triple-backtick SQL blocks
            r"```(?:SQL|sql|sqlite)?\s*(SELECT\s+[\s\S]*?)```",
            r"\*\*(?:SQL|sql|sqlite)?\s*(SELECT\s+[\s\S]*?)\*\*",
            # Ensure full SELECT statement is captured, including nested queries
            # r"(?:###\s*SQL:|</think>|<think>)\s*(SELECT\s+[\s\S]*?(?:;|$))",
            r"</think>\s*(?:.*?)(SELECT\s+[\s\S]*?(?:;|$))",
            # r":\s*(SELECT\s+[\s\S]*?(?:;|$))",
            # r"(SELECT\s+DISTINCT\s+[\s\S]*?(?:;|$))",
            r"(SELECT\s+[\s\S]*?(?:;|$))"
        ]

        # Enhanced regex patterns with more flexible matching

        # Try each pattern in order
        for pattern in extraction_patterns:
            if not pattern:  # Skip empty patterns
                continue
            try:
                matches = re.findall(pattern, query, re.DOTALL)
                if matches:
                    return clean_query(matches[-1])  # Return last match if found
            except re.error as e:
                print(f"Regex error in pattern: {pattern} - {e}")  # Debugging log
        return ""
        # # Extract SQL inside code blocks
        # if '```SQL' in query or '```sql' in query:
        #     try:
        #         sql = re.findall(r"```(?:SQL|sql|sqlite)(.*?)```", query, re.DOTALL | re.IGNORECASE)[0]
        #         return clean_query(sql)
        #     except IndexError:
        #         sql = re.findall(r"```(?:SQL|sql)(.*?)", query, re.DOTALL | re.IGNORECASE)[0]
        #         return clean_query(sql)
        
        # # Extract SQL after "</think>"
        # if '</think>' in query:
        #     sql_queries = re.findall(r"</think>\s*(SELECT .*?;)", query, re.DOTALL | re.IGNORECASE)
        #     print(sql_queries)
        #     sql_queries = re.findall(r"</think>\s*(SELECT .*?)(?:;|$)", query, re.DOTALL | re.IGNORECASE)
        #     print(sql_queries)
        #     if sql_queries:
        #         return clean_query(sql_queries[-1])

        # # Extract SQL within THINK blocks
        # if '<think>' in query:
        #     sql_queries = re.findall(r"<think>.*?(SELECT .*?)(?:;|$)", query, re.DOTALL | re.IGNORECASE)
        #     if sql_queries:
        #         return clean_query(sql_queries[-1])

        # # Extract SQL after colons
        # if ':' in query:
        #     sql_queries = re.findall(r":\s*(SELECT .*?;)", query, re.DOTALL | re.IGNORECASE)
        #     if sql_queries:
        #         return clean_query(sql_queries[-1])

        # # Extract SQL ending with a semicolon
        # if ';' in query:
        #     sql_queries = re.findall(r"(SELECT .*?;)", query, re.DOTALL | re.IGNORECASE)
        #     if sql_queries:
        #         return clean_query(sql_queries[-1])


        # Default case
        

    # elif llm in {"codellama", "sqlcoder"}:
    #     # Handle codellama/sqlcoder cases
    #     if ';' in query:
    #         sql = re.findall(r'(.*?);', query.replace('\n', ' '))[0]
    #         return clean_query(sql)
    #     else:
    #         return "SELECT " + clean_query(query)

    # elif llm == "puyu":
    #     # Handle puyu cases
    #     query = query.replace("#", "")
    #     return clean_query(query) if query.lower().strip().startswith("select") else "SELECT " + clean_query(query)

    # elif llm == 'gpt':
    #     # Handle GPT-specific processing
    #     sql = " ".join(query.replace("\n", " ").split())
    #     return clean_query(sql) if sql.lower().strip().startswith("select") else "SELECT " + clean_query(sql)

    else:
        # Default case
        return clean_query(query)

# input = """
# <think> Okay, so I need to figure out which car model has the minimum horsepower using the provided SQLite database schema. Let me start by understanding the tables and their relationships.   First, I see that the relevant tables here are probably 'cars_data' and 'model_list' or 'car_names'. The 'cars_data' table holds information like MPG, Cylinders, Edispl, Horsepower, etc., and each entry is linked via the 'Id' to 'car_names'. The 'car_names' table has MakeId, Model, and Make. Wait, actually, looking closer, 'car_names' has Model as a field, which might correspond to the model name, but I'm not entirely sure.   Wait, there's also a 'model_list' table with ModelId, Maker, and Model. So perhaps 'model_list' is where the actual model names are stored, and 'car_names' might be a different structure. Let me check the foreign keys again.   The foreign key information says that 'cars_data(Id)' references 'car_names(MakeId)'. So each car in 'cars_data' is linked to a MakeId in 'car_names'. Then, 'car_names(Model)' references 'model_list(Model)'. Wait, that seems a bit confusing. Let me parse this correctly.  Looking at the foreign keys:  - cars_data(Id) REFERENCES car_names(MakeId) - car_names(Model) REFERENCES model_list(Model)  So, 'car_names' has a field 'Model' that is a foreign key to 'model_list(Model)'. Therefore, to get the model name, I need to join 'car_names' with 'model_list' on the 'Model' field.  But wait, in the 'model_list' table, the 'Model' is like 'amc', 'audi', 'bmw', and in 'car_names', the 'Model' is 'chevrolet', 'buick', 'plymouth'. That doesn't seem to match. Maybe I'm misunderstanding the structure.  Wait, looking at the data information:  - model_list(ModelId[1,2,3], Maker[1,2,3], Model[amc,audi,bmw]) - car_names(MakeId[1,2,3], Model[chevrolet, buick, plymouth], Make[chevrolet chevelle malibu, etc.])  So, 'model_list' has models like 'amc', 'audi', 'bmw', while 'car_names' has models like 'chevrolet', 'buick', etc. So perhaps the 'Model' in 'car_names' refers to the model name, and 'model_list' is a separate list. But the foreign key is 'car_names(Model) REFERENCES model_list(Model)', which would mean that 'chevrolet' should be a value in 'model_list.Model', but in the data, 'model_list.Model' has 'amc', 'audi', 'bmw'. That doesn't match. Hmm, maybe I'm misunderstanding the schema.  Alternatively, perhaps 'model_list' is a list of models made by each car maker, and 'car_names' links a MakeId to a specific model. So, to get the model name, I need to join 'car_names' with 'model_list' on the 'Model' field.  But given the data, it's a bit confusing. Maybe I should approach it differently. The 'cars_data' table has an 'Id' that links to 'car_names.MakeId'. So each car in 'cars_data' is associated with a 'MakeId' in 'car_names', which in turn has a 'Model' that links to 'model_list.Model'. So, to get the model name, I need to go through 'car_names' and 'model_list'.  Wait, but in the data provided, 'model_list.Model' has 'amc', 'audi', 'bmw', while 'car_names.Model' has 'chevrolet', 'buick', 'plymouth'. So perhaps the foreign key is incorrect, or maybe I'm misinterpreting it. Alternatively, perhaps 'car_names.Model' is actually the 'ModelId' from 'model_list', but that doesn't make sense because 'Model' is a string.  This is getting a bit tangled. Maybe I should focus on the 'cars_data' table and see how to get the model name. Since 'cars_data.Id' links to 'car_names.MakeId', and 'car_names.Model' links to 'model_list.Model', perhaps the correct way is to join 'cars_data' with 'car_names' on MakeId, and then join 'car_names' with 'model_list' on Model.  But given the data, perhaps the model name is directly in 'car_names.Model'. So, maybe I can just join 'cars_data' with 'car_names' on MakeId, and then select the Model from 'car_names' along with Horsepower from 'cars_data'.  Wait, the question is asking for the model with the minimum horsepower. So, I need to find the model name and the horsepower value, and then find which model has the smallest horsepower.  So, the steps would be:  1. Select the Model name and Horsepower from the relevant tables. 2. Group by Model name. 3. Order by Horsepower in ascending order to find the minimum.  But wait, each car in 'cars_data' is a specific instance, so if multiple cars have the same model, their horsepower might vary. But in the data provided, all cars have the same Year (1970), and their horsepower varies: 130, 165, 150. So, the minimum would be 130.  But I need to get the model name associated with that horsepower. So, perhaps I need to join 'cars_data' with 'car_names' on MakeId to get the Model name.  Wait, in the data:  cars_data has Id 1,2,3.  car_names has MakeId 1,2,3 with Model 'chevrolet', 'buick', 'plymouth'.  So, for each car in cars_data, the MakeId links to car_names.Model, which gives the model name.  So, the SQL would be:  SELECT T1.Model, T2.Horsepower FROM car_names AS T1 INNER JOIN cars_data AS T2 ON T1.MakeId = T2.Id  But the question is to find the model with the minimum horsepower. So, I need to group by model and find the one with the smallest horsepower. Wait, but each model might have multiple entries in cars_data. Do I need to average or take the minimum per model?  Wait, the question is "Which model of the car has the minimum horsepower?" So, it's possible that each model has only one horsepower value, or multiple. In the data, each model is linked to one car, so each model has one horsepower. So, I can just select the model with the smallest horsepower.  So, the SQL would be:  SELECT T1.Model, T2.Horsepower FROM car_names AS T1 INNER JOIN cars_data AS T2 ON T1.MakeId = T2.Id ORDER BY T2.Horsepower ASC LIMIT 1;  But wait, looking at the data, 'cars_data' has three entries with Horsepower 130, 165, 150. So, the minimum is 130, which corresponds to MakeId 1, which is 'chevrolet' in car_names. So, the result would be 'chevrolet' with 130 horsepower.  But the user's example outputs don't include the value, just the model. Wait, looking at the examples, sometimes they select only the model, sometimes they select both. In the question, it's "Which model of the car has the minimum horsepower?" So, perhaps just selecting the model name is sufficient, but the example given in the problem statement shows selecting both the model name and the horsepower.  Wait, in the examples provided, like "Which model has the least amount of RAM? List the model name and the amount of RAM." The SQL selects both. So, in this case, the question isn't explicitly asking for both, but perhaps it's better to include both to be thorough.  But the user's instruction says: "Answer the final question below by providing only the final SQLite SQL query syntax without commentary and explanation." So, I just need to write the SQL.  Wait, in the examples, sometimes they select only the relevant column, like the school_code or customer_status_code, and sometimes they select two columns. So, in this case, perhaps selecting both Model and Horsepower is appropriate.  Alternatively, if the question is just asking for the model name, maybe only select the Model. But the examples vary, so perhaps it's safer to select both.  Wait, let me look at the example given in the problem statement:  "Which model has the least amount of RAM? List the model name and the amount of RAM."  The SQL is:  SELECT Model_name , RAM_MiB FROM chip_model ORDER BY RAM_MiB ASC LIMIT 1;  So, in that case, both columns are selected.  In our case, the question is "Which model of the car has the minimum horsepower?" It doesn't explicitly say to list both, but to be consistent with the examples, perhaps selecting both the model name and the horsepower is better.  Therefore, the SQL would be:  SELECT T1.Model, T2.Horsepower FROM car_names AS T1 INNER JOIN cars_data AS T2 ON T1.MakeId = T2.Id ORDER BY T2.Horsepower ASC LIMIT 1;  But wait, in the data, each MakeId in cars_data corresponds to a unique Model in car_names, so this should work.  Alternatively, perhaps the model_list table is involved. Let me check again.  The model_list table has ModelId, Maker, Model. So, perhaps the model name is in model_list, and car_names has a different structure

# """
# clean_sql_deepseek_r1_70b = sqlparse.format(extract_sql(input, "sensechat").strip(), reindent=False)
# print(clean_sql_deepseek_r1_70b)
if __name__ == "__main__":
    
    
    with open(args.file, 'r', encoding='utf-8') as file:
        content = file.readlines()

    # print(len(content))
    # input = """
    # <think> Alright, let's tackle this problem step by step. So, the question is asking for all distinct countries where singers above the age of 20 are from. I need to write an SQLite query that fetches this information efficiently.  First, I'll look at the tables provided. The relevant table here seems to be the 'singer' table because it contains details about each singer, including their country and age. The columns in the singer table are Singer_ID, Name, Country, Song_Name, Song_release_year, Age, and Is_male.  The goal is to find countries of singers who are older than 20 years old. Since we're only interested in distinct countries, I should use the DISTINCT keyword to avoid duplicates.  I need to select the 'Country' from the singer table where the singer's age is greater than 20. So, my query will be selecting Country from singer with a WHERE clause that filters Age > 20.  Also, considering efficiency, since this is a straightforward SELECT with a simple condition and using DISTINCT, it should execute quickly without needing complex joins or subqueries. There's no need to involve other tables like concert or stadium because the information required is all within the singer table.  Putting it all together, the SQL query would be selecting the distinct Country from the singer table where Age is greater than 20. </think>  To find all distinct countries of singers above age 20:  SELECT DISTINCT Country FROM singer WHERE Age > 20

    # """
    # print(extract_sql(input, "sensechat"))
    # mid = extract_sql_from_text("\n".join(content))
    # extracted_query = [sqlparse.format(extract_sql(q, "sensechat").strip(), reindent=False) for q in content]
    extracted_query = []
    for index,q in enumerate(content):
        result = extract_sql(q, "sensechat")
        if not result:  
        #     print(f"Extracted SQL {index}: {result}")  # Debugging log
        # else:
            print(f"Failed to extract SQL from line {index}:")  # Log missing SQL
        if result:  # Check if result is not None or empty
            if isinstance(result, list):  # Handle multiple queries
                formatted_queries = [sqlparse.format(query.strip(), reindent=False) if query else None for query in result]
                extracted_query.extend(formatted_queries)
            else:  # Single query
                formatted_query = sqlparse.format(result.strip(), reindent=False)
                extracted_query.append(formatted_query.replace(" '''",""))
        else:
            extracted_query.append("")
    # else:
        
    #     print(f"No valid SQL extracted for: {q}")  # Debugging log (optional)
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