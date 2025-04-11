# import openai
import os
from openai import OpenAI
import time 
# Configure OpenAI API
# openai.api_key = os.getenv("OPEN_API_KEY")  # Replace with your actual OpenAI API key
# openai.organization = os.getenv("OPEN_GROUP_ID")  # Set the OpenAI organization ID if needed


class GPT:
    def __init__(self, model="o1-preview"):
        """
        Initialize the GPT class.

        Args:
            model (str): Default model name for the OpenAI API. For example: "gpt-3.5-turbo".
        """
        self.model = model
        # 初始化 OpenAI 客戶端
        self.client = OpenAI(
            api_key=os.environ.get("OPEN_API_KEY"),  # or your actual API key
        )

    def __call__(
        self,
        prompt: str,
        model: str = None,  # <-- 新增 model 參數，預設 None
        temperature: float = 1,
        max_tokens: int = 2048,
        retries: int = 3,  # Number of retries if result is empty
        **kwargs
    ) -> str:
        """
        Generate a response from the OpenAI API using the ChatCompletion endpoint.

        Args:
            prompt (str): The input prompt for the model.
            model (str): Which model to use. If None, defaults to self.model.
            temperature (float): Sampling temperature for randomness.
            max_tokens (int): Maximum number of tokens to generate.
            **kwargs: Additional parameters for the OpenAI API.

        Returns:
            str: The generated response.
        """
        if not prompt or prompt.strip() == "":
            raise ValueError("Prompt cannot be None or empty")
        use_model = model if model else self.model
        client = self.client
        
        attempt = 0
        while attempt < retries:
            try:
                # Call ChatCompletion API
                response = client.chat.completions.create(
                    model=use_model,
                    messages=[
                        {"role": "user", "content": str(prompt)}  # Ensure prompt is string
                    ],
                    # temperature=temperature,
                    # max_tokens=max_tokens,
                    **kwargs
                )
                result = response.choices[0].message.content.strip()  # Strip whitespace
                
                # Check if result is empty
                if not result:
                    attempt += 1
                    print(f"Empty result on attempt {attempt}/{retries}. Retrying...")
                    time.sleep(1)  # Brief delay before retrying
                    continue
                
                print(f"Model: {use_model}")
                print(f"Result: {result}")
                return [result]  # Return as a list per your original code
            
            except Exception as e:
                print(f"Error calling OpenAI API on attempt {attempt + 1}/{retries}: {str(e)}")
                attempt += 1
                if attempt == retries:
                    raise Exception(f"Failed to get a valid response after {retries} attempts: {str(e)}")
                time.sleep(1)  # Delay before retrying
        
        raise Exception(f"Failed to get a non-empty response after {retries} attempts")

    def generate_batch(
        self,
        prompts: list,
        model: str = None,  # <-- 同樣在 batch_generate 加上 model 參數
        temperature: float = 0.7,
        max_tokens: int = 200,
        **kwargs
    ) -> list:
        """
        Generate responses for a batch of prompts.

        Args:
            prompts (list of str): List of input prompts.
            model (str): Which model to use. If None, defaults to self.model.
            temperature (float): Sampling temperature for randomness.
            max_tokens (int): Maximum number of tokens to generate.
            **kwargs: Additional parameters for the OpenAI API.

        Returns:
            list of str: List of generated responses.
        """
        responses = []
        for prompt in prompts:
            try:
                prompt = f"""Please only output the final SQL with this format '''sql <predicted sql here>.''' {prompt}"""

                # 每條 prompt 都可以帶入同樣的 model。若要單獨針對不同 prompt 使用不同 model，請自行修改。
                response = self.__call__(
                    prompt,
                    model=model,
                    # temperature=temperature,
                    # max_completion_tokens=max_tokens,
                    **kwargs
                )
                responses.extend(response)
                time.sleep(2)
            except Exception as e:
                print(f"Error: {e}")
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

if __name__ == '__main__':
    # Example usage
    llm = GPT(model="o3-mini")  # 也可以改成 "gpt-3.5-turbo" 或其他
    
    # Single prompt example
    # prompt = "请用一句话解释万有引力"
    # response = llm(prompt)  # 不傳 model 時，預設使用 self.model
    # print(f"Response: {response}")

    # 若要在呼叫時另外指定模型，可這樣寫
    # response = llm(prompt, model="gpt-3.5-turbo")

    # Batch prompts example
    prompts = [
        """
### Some example pairs of question and corresponding SQL query are provided based on similar problems:

### What are the name and code of the location with the smallest number of documents?
SELECT T2.location_name ,  T1.location_code FROM Document_locations AS T1 JOIN Ref_locations AS T2 ON T1.location_code  =  T2.location_code GROUP BY T1.location_code ORDER BY count(*) ASC LIMIT 1

### What is the code of the city with the most students?
SELECT city_code FROM student GROUP BY city_code ORDER BY count(*) DESC LIMIT 1

### What is the id, name and IATA code of the airport that had most number of flights?
SELECT T1.id ,  T1.name ,  T1.IATA FROM airport AS T1 JOIN flight AS T2 ON T1.id  =  T2.airport_id GROUP BY T2.id ORDER BY count(*) DESC LIMIT 1

### What is the status code with the least number of customers?
SELECT customer_status_code FROM Customers GROUP BY customer_status_code ORDER BY count(*) ASC LIMIT 1;

### What are the codes of the locations with at least three documents?
SELECT location_code FROM Document_locations GROUP BY location_code HAVING count(*)  >=  3

### What are the codes corresponding to document types for which there are less than 3 documents?
SELECT document_type_code FROM Documents GROUP BY document_type_code HAVING count(*)  <  3

### How many different status codes of things are there?
SELECT count(DISTINCT Status_of_Thing_Code) FROM Timed_Status_of_Things

### Find the code of the location with the largest number of documents.
SELECT location_code FROM Document_locations GROUP BY location_code ORDER BY count(*) DESC LIMIT 1

### What destination has the fewest number of flights?
SELECT destination FROM Flight GROUP BY destination ORDER BY count(*) LIMIT 1

### Your task: 
Answer the final question below by providing **only** the final SQLite SQL query syntax without commentary and explanation.  You must minimize SQL execution time while ensuring correctness.

    ### Sqlite SQL tables, with their properties:
#
# airlines(uid, Airline, Abbreviation, Country);
# airports(City, AirportCode, AirportName, Country, CountryAbbrev);
# flights(Airline, FlightNo, SourceAirport, DestAirport).
#
    # ### Here are some data information about database references.
    # #
# airlines(uid[1,2,3],Airline[United Airlines,US Airways,Delta Airlines],Abbreviation[UAL,USAir,Delta],Country[USA,USA,USA]);
# airports(City[Aberdeen ,Aberdeen ,Abilene ],AirportCode[APG,ABR,DYS],AirportName[Phillips AAF ,Municipal ,Dyess AFB ],Country[United States ,United States ,United States ],CountryAbbrev[US ,US,US]);
# flights(Airline[1,1,1],FlightNo[28,29,44],SourceAirport[ APG, ASY, CVO],DestAirport[ ASY, APG, ACV]);
#
### Foreign key information of Sqlite SQL tables, used for table joins: 
#
# flights(DestAirport) REFERENCES airports(AirportCode);
# flights(SourceAirport) REFERENCES airports(AirportCode)
#
### Final Question: What is the code of airport that has fewest number of flights?
### SQL: 
        
        """
    ]
    # batch_responses = llm.batch_generate(prompts)
    # print("Batch Responses:")
    # for i, res in enumerate(batch_responses):
        
    #     print(f"{i + 1}. {res}")
    db_id = "flight_2"
    db_path = f"./data/spider/test_database/{db_id}/{db_id}.sqlite"
    print(execute_sql("""SELECT AirportCode 
FROM (
  SELECT SourceAirport AS AirportCode FROM flights
  UNION ALL
  SELECT DestAirport FROM flights
) AS all_flights
GROUP BY AirportCode
ORDER BY COUNT(*) ASC
LIMIT 1;""",db_path))