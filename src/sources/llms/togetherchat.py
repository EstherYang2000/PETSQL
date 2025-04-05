from together import Together
import os

class TogetherChat:
    def __init__(self, model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"):
        api_key = os.getenv("TOGETHER_API_KEY")  # 读取环境变量中的 API Key
        if not api_key:
            raise ValueError("API key not found. Please set TOGETHER_API_KEY as an environment variable.")
        self.client = Together()
        self.model = model

    def chat(self, messages, stream=True):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=None,
            temperature=0.0,
            top_p=0.7,
            top_k=50,
            repetition_penalty=1,
            stop=["<|eot_id|>", "<|eom_id|>"],
            stream=stream
        )
        if stream:
            for token in response:
                if hasattr(token, 'choices'):
                    print(token.choices[0].delta.content, end='', flush=True)
        else:
            return response

    def generate_batch(self, prompts):
        responses = []
        for prompt in prompts:
            try:
                messages = [{"role": "user", "content": prompt}]

                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=512,
                    temperature=0.5,
                    top_p=1,
                    top_k=50,
                    repetition_penalty=1,
                    stop=["<|eot_id|>", "<|eom_id|>"],
                )
                responses.append(response.choices[0].message.content)
            except Exception as e:
                print(f"Error for prompt '{prompt}': {e}")
                responses.append("")
        return responses

import sqlite3
def execute_sql(sql: str, db_path: str) -> (bool, str):
        """
        Execute SQL against the target SQLite database.
        :return: (success, error_message)
        """
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute(sql)
            conn.close()
            return True, ""
        except sqlite3.Error as e:
            return False, str(e)

# 使用示例
if __name__ == "__main__":
    chat_bot = TogetherChat()
    prompts = """
### Some example pairs of question and corresponding SQL query are provided based on similar problems:

### Show the phone, room, and building for the faculty named Jerry Prince.
SELECT phone ,  room ,  building FROM Faculty WHERE Fname  =  "Jerry" AND Lname  =  "Prince"

### Show the document name and the document date for all documents on project with details 'Graph Database project'.
SELECT document_name ,  document_date FROM Documents AS T1 JOIN projects AS T2 ON T1.project_id  =  T2.project_id WHERE T2.project_details  =  'Graph Database project'

### Show the positions of the players from the team with name "Ryley Goldner".
SELECT T1.Position FROM match_season AS T1 JOIN team AS T2 ON T1.Team  =  T2.Team_id WHERE T2.Name  =  "Ryley Goldner"

### Show the players and years played for players from team "Columbus Crew".
SELECT T1.Player , T1.Years_Played FROM player AS T1 JOIN team AS T2 ON T1.Team  =  T2.Team_id WHERE T2.Name  =  "Columbus Crew"

### Find the total budgets of the Marketing or Finance department.
SELECT sum(budget) FROM department WHERE dept_name  =  'Marketing' OR dept_name  =  'Finance'

### Show the description for role name "Proof Reader".
SELECT role_description FROM ROLES WHERE role_name  =  "Proof Reader"

### Find the names and descriptions of the photos taken at the tourist attraction called "film festival".
SELECT T1.Name ,  T1.Description FROM PHOTOS AS T1 JOIN TOURIST_ATTRACTIONS AS T2 ON T1.Tourist_Attraction_ID  =  T2.Tourist_Attraction_ID WHERE T2.Name  =  "film festival"

### Find the first name and last name for the "CTO" of the club "Hopkins Student Enterprises"?
SELECT t3.fname ,  t3.lname FROM club AS t1 JOIN member_of_club AS t2 ON t1.clubid  =  t2.clubid JOIN student AS t3 ON t2.stuid  =  t3.stuid WHERE t1.clubname  =  "Hopkins Student Enterprises" AND t2.position  =  "CTO"

### Find the name and capacity of the stadium where the event named "World Junior" happened.
SELECT t1.name ,  t1.capacity FROM stadium AS t1 JOIN event AS t2 ON t1.id  =  t2.stadium_id WHERE t2.name  =  'World Junior'

### Your task: 
Answer the final question below by providing **only** the final SQLite SQL query syntax without commentary and explanation.  You must minimize SQL execution time while ensuring correctness.

    ### Sqlite SQL tables, with their properties:
#
# battle(id, name, date, bulgarian_commander, latin_commander, result);
# ship(lost_in_battle, id, name, tonnage, ship_type, location, disposition_of_ship);
# death(caused_by_ship_id, id, note, killed, injured).
#
    # ### Here are some data information about database references.
    # #
# battle(id[1,2,3],name[Battle of Adrianople,Battle of Serres,Battle of Rusion],date[14 April 1205,June 1205,31 January 1206],bulgarian_commander[Kaloyan,Kaloyan,Kaloyan],latin_commander[Baldwin I,Unknown,Thierry de Termond],result[Bulgarian victory,Bulgarian victory,Bulgarian victory]);
# ship(lost_in_battle[8,7,6],id[1,2,3],name[Lettice,Bon Accord,Mary],tonnage[t,t,t],ship_type[Brig,Brig,Brig],location[English Channel,English Channel,English Channel],disposition_of_ship[Captured,Captured,Captured]);
# death(caused_by_ship_id[1,2,3],id[1,2,13],note[Dantewada, Chhattisgarh,Dantewada, Chhattisgarh,Erraboru, Chhattisgarh],killed[8,3,25],injured[0,0,0]);
#
### Foreign key information of Sqlite SQL tables, used for table joins: 
#
# ship(lost_in_battle) REFERENCES battle(id);
# death(caused_by_ship_id) REFERENCES ship(id)
#
### Final Question: Show names, results and bulgarian commanders of the battles with no ships lost in the 'English Channel'.
### SQL:
    """
    
    # response = chat_bot.generate_batch([prompts])
    # print(response)
    
    
    
    
    
    db_id = "dog_kennels"
    db_path = f"./data/spider/test_database/{db_id}/{db_id}.sqlite"
    raw_sql = """SELECT T2.breed_name FROM Dogs AS T1 JOIN Breeds AS T2 ON T1.breed_code  =  T2.breed_code GROUP BY T1.breed_code ORDER BY count(*) DESC LIMIT 1"""
    result, error_message = execute_sql(raw_sql,db_path)
    print(f"Execution result: {result}, Error message: {error_message}")
    
    # database_type = "SQLite"
    # refinement_prompt = f"""
    # ### Task
    # You are a SQL expert responsible for fixing incorrect {database_type} SQL queries.
    # **only** the final {database_type} SQL query syntax without commentary and explanation.

    # ### User Question
    # {prompts}

    # ### Initial Generated SQL (which failed)
    # {raw_sql}

    # ### Error Message from Database Execution
    # {error_message}

    # ### Instruction
    # Please rewrite the {database_type} SQL query to correct the above errors. 
    # - Ensure the syntax strictly follows the target SQL dialect.
    # - Refer to the provided schema if necessary.
    # - Ensure table names and column names are correctly referenced.
    # - Double-check joins, conditions, and aggregations.

    # ### Final Corrected SQL 
    # ### SQL: 

    # """
    # # response = chat_bot.generate_batch([refinement_prompt])
    # print(response)
    