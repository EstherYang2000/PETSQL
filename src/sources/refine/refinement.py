import sqlparse
from post_process import extract_sql
from sql_gen.sql_utils import run_sql_generation
import sqlite3
# def execute_sql(sql: str, db_path: str) -> (bool, str):
#     """
#     Execute SQL against the target SQLite database.
#     :return: (success, error_message)
#     """
#     try:
#         conn = sqlite3.connect(db_path)
#         cursor = conn.cursor()
#         cursor.execute(sql)
#         conn.close()
#         return True, "ç
#     except sqlite3.Error as e:
#         return False, str(e)
def connect_db(sql_dialect, db_path):
    if sql_dialect == "SQLite":
        conn = sqlite3.connect(db_path)
    else:
        raise ValueError("Unsupported SQL dialect")
    return conn
def execute_sql(predicted_sql, db_path, sql_dialect):
    try:

        conn = connect_db(sql_dialect, db_path)
        # Connect to the database
        cursor = conn.cursor()
        cursor.execute(predicted_sql)
        conn.close()
        return True, ""
    except sqlite3.Error as e:
        print(f"SQLite error: {e}")
        return False,str(e)
def format_sql(raw_sql: str, llm: str = "sensechat") -> str:
    """
    Clean and format SQL using extract_sql and sqlparse.
    """
    extracted_sql = extract_sql(raw_sql, llm=llm)
    return sqlparse.format(extracted_sql.strip(), reindent=False)

def refine_sql_with_feedback(prompt: str, raw_sql: str, error_message: str, 
                             expert_name: str, path_generate: str, 
                             start_num_prompts: int, model_version: str = None,database_type:str = "SQLite",n_samples:int = 1) -> str:
    """
    When SQL execution fails, this triggers the LLM to rewrite the SQL
    using the original prompt + failed SQL + error message as input.
    """
    refinement_prompt = f"""
    ### Task
    You are a SQL expert responsible for fixing incorrect {database_type} SQL queries.
    **only** the final {database_type} SQL query syntax without commentary and explanation.

    ### User Question
    {prompt}

    ### Initial Generated SQL (which failed)
    {raw_sql}

    ### Error Message from Database Execution
    {error_message}

    ### Instruction
    Please rewrite the {database_type} SQL query to correct the above errors. 
    - Ensure the syntax strictly follows the target SQL dialect.
    - Refer to the provided schema if necessary.
    - Ensure table names and column names are correctly referenced.
    - Double-check joins, conditions, and aggregations.

    ### Final Corrected SQL 
    ### SQL: 

    """
    refined_sql = run_sql_generation(
        model=expert_name,
        prompts=[refinement_prompt],
        path_generate=path_generate,
        out_file=f"{expert_name}_{model_version}_refine.json",
        start_num_prompts=start_num_prompts,
        call_mode="append",
        model_version=model_version,
        n_samples=n_samples
    )
    print(refined_sql)
    # print(type(refined_sql['sql_candidates']))
    refined_sql = refined_sql[0]['sql_candidates'][0]
    return format_sql(refined_sql, llm="sensechat")

def refine_sql_candidates(prompt: str, raw_sql_candidates: list, expert_name: str, 
                          path_generate: str, start_num_prompts: int, model_version: str = None,
                          db_path: str = None, max_attempts: int = 3) -> list:
    """
    Check & refine SQL candidates by actually running them on SQLite.
    - Successful execution: Keep the SQL.
    - Execution failure: Trigger refinement with feedback.
    """
    refined_candidates = []
    
    for raw_sql in raw_sql_candidates:
        clean_sql = format_sql(raw_sql)
        # Validate SQL syntax using sqlparse before executing
        try:
            parsed = sqlparse.parse(clean_sql)
            if not parsed:
                raise ValueError(f"SQL parsing failed: {clean_sql}")
        except Exception as e:
            print(f"[ERROR] SQL parsing failed before execution: {e}")
            continue
        success, error_message = execute_sql(clean_sql, db_path,"SQLite")
        
        print(f"clean_sql : {clean_sql}")
        print(f"status : {success}")
        print(f"message : {error_message}")
        
        
        attempts = 0

        while not success and attempts < max_attempts:
            print(f"[Refinement Triggered] Refining failed SQL (attempt {attempts+1}/{max_attempts})")
            clean_sql = refine_sql_with_feedback(prompt, clean_sql, error_message,
                                                 expert_name, path_generate, start_num_prompts, model_version)
            clean_sql = sqlparse.format(extract_sql(clean_sql, "sensechat").strip(), reindent=False)
            success, error_message = execute_sql(clean_sql, db_path,"SQLite")
            print(f"attemps times : {attempts}" )
            print(f"clean_sql : {clean_sql}")
            print(f"status : {success}")
            print(f"message : {error_message}")
            attempts += 1

        if not success:
            print(f"[WARNING] SQL still failed after {max_attempts} refinements: {error_message}")
        refined_candidates.append(clean_sql)

    return refined_candidates

def run_refinement_pipeline(db_path:str,prompt:str,sql_candidates:list,path_generate:str,end_num_prompts:int,model:str):
    for raw_data in sql_candidates:
        raw_clean = []
        for raw_sql in raw_data['sql_candidates']:
            
            raw_clean.append(sqlparse.format(extract_sql(raw_sql, "sensechat").strip(), reindent=False))
            # 執行Refinement Pipeline（真正連DB執行，失敗就修）
        raw_data['sql_candidates'] = raw_clean
        raw_data['sql_candidates'] = refine_sql_candidates(
            prompt,raw_clean , 
            model, path_generate, end_num_prompts, "3.3", 
            db_path=db_path
        )
    return sql_candidates
    
    
if __name__ == '__main__':
    # Example DB Path
    db_id = "soccer_3"
    db_path = f"./data/spider/test_database/{db_id}/{db_id}.sqlite"
    prompt = """
        ### Some example pairs of question and corresponding SQL query are provided based on similar problems:

        ### How many services are there?
        SELECT count(*) FROM services

        ### How many performances are there?
        SELECT count(*) FROM performance

        ### How many artworks are there?
        SELECT count(*) FROM artwork

        ### How many premises are there?
        SELECT count(*) FROM premises

        ### How many artists are there?
        SELECT count(*) FROM artist

        ### How many actors are there?
        SELECT count(*) FROM actor

        ### How many debates are there?
        SELECT count(*) FROM debate

        ### How many users are there?
        SELECT count(*) FROM useracct

        ### How many players are there?
        SELECT count(*) FROM player

        ### Your task: 
        Answer the final question below by providing **only** the final SQLite SQL query syntax without commentary and explanation.  You must minimize SQL execution time while ensuring correctness.

            ### Sqlite SQL tables, with their properties:
        #
        # club(Club_ID, Name, Manager, Captain, Manufacturer, Sponsor);
        # player(Player_ID, Name, Country, Earnings, Events_number, Wins_count, Club_ID).
        #
            # ### Here are some data information about database references.
            # #
        # club(Club_ID[1,2,3],Name[Arsenal,Aston Villa,Blackburn Rovers],Manager[Arsène Wenger,Martin O'Neill,Sam Allardyce],Captain[Cesc Fàbregas,Martin Laursen,Ryan Nelsen],Manufacturer[Nike,Nike,Umbro],Sponsor[Fly Emirates,Acorns,Crown Paints]);
        # player(Player_ID[1.0,2.0,3.0],Name[Nick Price,Paul Azinger,Greg Norman],Country[Zimbabwe,United States,Australia],Earnings[1478557.0,1458456.0,1359653.0],Events_number[18,24,15],Wins_count[4,3,2],Club_ID[1,3,5]);
        #
        ### Foreign key information of Sqlite SQL tables, used for table joins: 
        #
        # player(Club_ID) REFERENCES club(Club_ID)
        #
        ### Final Question: How many clubs are there?
        ### SQL: 
    
    
    """
    # 生成raw_sql_candidates
    raw_sql_llama3_3 = [
    {
        "prompt_index": 0,
        "sql_candidates": [
            "'''sql SELECT count(*) FROM club '''",
            "'''sql SELECT count(*) FROM club '''",
            "'''sql SELECT count(*) FROM clubs '''",
            "'''sql SELECT count(*) FROM club '''",
            "'''sql SELECT count(*) FROM club '''"
        ]
    }
    ]
    path_generate = "data/process/PPL_TEST.JSON-9_SHOT_Euclidean_mask_1034"
    end_num_prompts = 1
    for raw_data in raw_sql_llama3_3:
        raw_clean = []
        for raw_sql in raw_data['sql_candidates']:
            
            raw_clean.append(sqlparse.format(extract_sql(raw_sql, "sensechat").strip(), reindent=False))
            # 執行Refinement Pipeline（真正連DB執行，失敗就修）
        raw_data['sql_candidates'] = raw_clean
        raw_data['sql_candidates'] = refine_sql_candidates(
            prompt,raw_clean , 
            "llamaapi", path_generate, end_num_prompts, "3.3", 
            db_path=db_path
        )
    print(raw_sql_llama3_3)