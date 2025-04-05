import os
import google.generativeai as genai
from base64 import b64encode
import requests
import time
class GoogleGeminiChat:
    def __init__(self, api_key=None, model_name="gemini-2.5-pro-exp-03-25"):
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
            try:
                response = self.chat(prompt)
                # Directly use response.text from the SDK
                print(f"Raw response: {response}")
                results.append(response.strip())
                time.sleep(1)
            except Exception as e:
                print(f"Error for prompt '{prompt}': {e}")
                results.append("")
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

### Which customer have the most policies? Give me the customer details.
SELECT t2.customer_details FROM policies AS t1 JOIN customers AS t2 ON t1.customer_id  =  t2.customer_id GROUP BY t2.customer_details ORDER BY count(*) DESC LIMIT 1

### What are the names of cities that are in the county with the most police officers?
SELECT name FROM city WHERE county_ID  =  (SELECT county_ID FROM county_public_safety ORDER BY Police_officers DESC LIMIT 1)

### What is the country that has the most perpetrators?
SELECT Country ,  COUNT(*) FROM perpetrator GROUP BY Country ORDER BY COUNT(*) DESC LIMIT 1

### What is the label that has the most albums?
SELECT label FROM albums GROUP BY label ORDER BY count(*) DESC LIMIT 1

### Which artist has the most albums?
SELECT T2.Name FROM ALBUM AS T1 JOIN ARTIST AS T2 ON T1.ArtistId  =  T2.ArtistId GROUP BY T2.Name ORDER BY COUNT(*) DESC LIMIT 1

### Which nationality has the most hosts?
SELECT Nationality FROM HOST GROUP BY Nationality ORDER BY COUNT(*) DESC LIMIT 1

### Which industry has the most companies?
SELECT Industry FROM Companies GROUP BY Industry ORDER BY COUNT(*) DESC LIMIT 1

### Which song has the most vocals?
SELECT title FROM vocals AS T1 JOIN songs AS T2 ON T1.songid  =  T2.songid GROUP BY T1.songid ORDER BY count(*) DESC LIMIT 1

### Which major has the most students?
SELECT Major FROM STUDENT GROUP BY major ORDER BY count(*) DESC LIMIT 1

### Your task: 
Answer the final question below by providing **only** the final SQLite SQL query syntax without commentary and explanation.  You must minimize SQL execution time while ensuring correctness.
### Sqlite SQL tables, with their properties:
#
.
#
### Here are some data information about database references.
#
# Breeds(breed_code[ESK,HUS,BUL],breed_name[Eskimo,Husky,Bulldog]);
# Dogs(dog_id[1,2,3],owner_id[3,11,1],abandoned_yn[1,0,0],breed_code[ESK,BUL,BUL],size_code[LGE,LGE,MED],name[Kacey,Hipolito,Mavis],age[6,9,8],date_of_birth[2012-01-27 05:11:53,2013-02-13 05:15:21,2008-05-19 15:54:49],gender[1,0,1],weight[7.57,1.72,8.04],date_arrived[2017-09-08 20:10:13,2017-12-22 05:02:02,2017-06-25 10:14:05],date_adopted[2018-03-06 16:32:11,2018-03-25 08:12:51,2018-03-07 21:45:43],date_departed[2018-03-25 06:58:44,2018-03-25 02:11:32,2018-03-25 10:25:46]);
#
### Final Question: Which breed do the most dogs have? Give me the breed name.
### SQL:
        """
    ]
    responses = chat_bot.generate_batch(batch_prompts)
    for i, res in enumerate(responses):
        print(f"Batch Response {i+1}: {res}")