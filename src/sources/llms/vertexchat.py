import os
import google.generativeai as genai
from base64 import b64encode
import requests
import time
class GoogleGeminiChat: 
    def __init__(self, api_key=None, model_name="models/gemini-2.5-pro-preview-03-25"): # gemini-2.5-pro-exp-03-25 # gemini-2.5-pro-preview-03-25
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("API key not found. Set GOOGLE_API_KEY as an environment variable or pass it in.")
        # Configure the SDK with the API key
        genai.configure(api_key=self.api_key)
        # Initialize the model
        self.model = genai.GenerativeModel(model_name=model_name)


    def chat(self, messages):
        """Send messages to the Gemini model and return the response."""
        try:
            # For simplicity, assuming messages is a single string or list of content
            if isinstance(messages, str):
                response = self.model.generate_content(messages)
            else:
                # Handle structured input (e.g., list of dicts or content parts)
                response = self.model.generate_content(messages)
            return response.text
        except Exception as e:
            return f"Error generating content: {str(e)}"

    def generate_batch(self, prompts):
        """Batch process text-only prompts and return responses as a list."""
        results = []
        for prompt in prompts:
            max_retries = 3  # Set the maximum number of retries
            retries = 0
            while retries < max_retries:
                try:
                    response = self.chat(prompt)
                    # Directly use response.text from the SDK
                    print(f"Raw response: {response}")
                    results.append(response.strip())
                    time.sleep(5)  # Wait 10 seconds between prompts
                    break  # Exit the retry loop on success
                except Exception as e:
                    retries += 1
                    print(f"Error for prompt '{prompt}': {e}. Retrying in 1 minute... ({retries}/{max_retries})")
                    time.sleep(60)  # Wait 1 minute before retrying
                    if retries == max_retries:
                        print(f"Failed after {max_retries} retries for prompt '{prompt}'.")
                        results.append("")  # Append an empty result after max retries
        return results

if __name__ == "__main__":
    # Initialize the chat bot
    chat_bot = GoogleGeminiChat()  # Ensure GOOGLE_API_KEY is set in your environment

    # # Simple test
    # prompt = "What is the capital of France?"
    # response = chat_bot.chat(prompt)
    # print(f"Single response: {response}")

    # Batch test with your previous SQL example
    batch_prompts = [
        """
### Some example pairs of question and corresponding SQL query are provided based on similar problems:

### Return the positions of players on the team Ryley Goldner.
SELECT T1.Position FROM match_season AS T1 JOIN team AS T2 ON T1.Team  =  T2.Team_id WHERE T2.Name  =  "Ryley Goldner"

### Return the address of customer 10.
SELECT T1.address_details FROM addresses AS T1 JOIN customer_addresses AS T2 ON T1.address_id  =  T2.address_id WHERE T2.customer_id  =  10

### Return the characteristic names of the 'sesame' product.
SELECT t3.characteristic_name FROM products AS t1 JOIN product_characteristics AS t2 ON t1.product_id  =  t2.product_id JOIN CHARACTERISTICS AS t3 ON t2.characteristic_id  =  t3.characteristic_id WHERE t1.product_name  =  "sesame"

### Return the founder of Sony.
SELECT founder FROM manufacturers WHERE name  =  'Sony'

### Return the types of film market estimations in 1995.
SELECT TYPE FROM film_market_estimation WHERE YEAR  =  1995

### Return the names of musicals who have the nominee Bob Fosse.
SELECT Name FROM musical WHERE Nominee  =  "Bob Fosse"

### Return the the "active to date" of the latest contact channel used by the customer named "Tillman Ernser".
SELECT max(t2.active_to_date) FROM customers AS t1 JOIN customer_contact_channels AS t2 ON t1.customer_id  =  t2.customer_id WHERE t1.customer_name  =  "Tillman Ernser"

### Return the phone numbers of employees with salaries between 8000 and 12000.
SELECT phone_number FROM employees WHERE salary BETWEEN 8000 AND 12000

### Return the cities with more than 3 airports in the United States.
SELECT city FROM airports WHERE country  =  'United States' GROUP BY city HAVING count(*)  >  3

### Your task: 
Answer the final question below by providing **only** the final SQLite SQL query syntax without commentary and explanation.  You must minimize SQL execution time while ensuring correctness.

    ### Sqlite SQL tables, with their properties:
#
# vehicle(Vehicle_ID, Model, Build_Year, Top_Speed, Power, Builder, Total_Production);
# driver(Driver_ID, Name, Citizenship, Racing_Series);
# vehicle_driver(Driver_ID, Vehicle_ID).
#
    # ### Here are some data information about database references.
    # #
# vehicle(Vehicle_ID[1,2,3],Model[AC4000,DJ ,DJ1],Build_Year[1996,2000,2000–2001],Top_Speed[120,200,120],Power[4000,4800,6400],Builder[Zhuzhou,Zhuzhou,Zhuzhou Siemens , Germany],Total_Production[1,2,20]);
# driver(Driver_ID[1,2,3],Name[Jeff Gordon,Jimmie Johnson,Tony Stewart],Citizenship[United States,United States,United States],Racing_Series[NASCAR,NASCAR,NASCAR]);
# vehicle_driver(Driver_ID[1,1,1],Vehicle_ID[1,3,5]);
#
### Foreign key information of Sqlite SQL tables, used for table joins: 
#
# vehicle_driver(Vehicle_ID) REFERENCES vehicle(Vehicle_ID);
# vehicle_driver(Driver_ID) REFERENCES driver(Driver_ID)
#
### Final Question: Return the names of drivers with citizenship from the United States.
### SQL: 
        """
    ]
    responses = chat_bot.generate_batch(batch_prompts)
    for i, res in enumerate(responses):
        print(f"Batch Response {i+1}: {res}")