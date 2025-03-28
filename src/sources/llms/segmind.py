import os
import requests
from base64 import b64encode

class ClaudeChat:
    def __init__(self, api_key=None, model_url="https://api.segmind.com/v1/claude-3.7-sonnet"):
        self.api_key = api_key or os.getenv("SEGMIND_API_KEY")
        if not self.api_key:
            raise ValueError("API key not found. Set SEGMIND_API_KEY as an environment variable or pass it in.")
        self.model_url = model_url

    def to_base64(self, img_url):
        return b64encode(requests.get(img_url).content).decode('utf-8')

    def chat(self, messages):
        headers = {"x-api-key": self.api_key}
        data = {"messages": messages}
        response = requests.post(self.model_url, json=data, headers=headers)
        return response.json()

    def chat_with_image(self, prompt_text, img_url):
        image_b64 = self.to_base64(img_url)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": image_b64
                        }
                    }
                ]
            }
        ]
        return self.chat(messages)

    def generate_batch(self, prompts):
        """Batch process text-only prompts and return responses as a list."""
        results = []
        for prompt in prompts:
            try:
                messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
                response = self.chat(messages)
                # Segmind's response format assumption: {'message': '...'}
                print(response)
                content = response.get("content", [])
                sql_query = ""
                for item in content:
                    if item.get("type") == "text":
                        sql_query = item.get("text", "").strip()
                        break
                results.append(sql_query)
            except Exception as e:
                print(f"Error for prompt '{prompt}': {e}")
                results.append("")
        return results
    
if __name__ == "__main__":
    chat_bot = ClaudeChat()
    batch_prompts = [
        """
        ### Some example pairs of question and corresponding SQL query are provided based on similar problems:

### What is the author of the submission with the highest score?
SELECT Author FROM submission ORDER BY Scores DESC LIMIT 1

### What is the country of the airport with the highest elevation?
SELECT country FROM airports ORDER BY elevation DESC LIMIT 1

### What is the name and sex of the candidate with the highest support rate?
SELECT t1.name ,  t1.sex FROM people AS t1 JOIN candidate AS t2 ON t1.people_id  =  t2.people_id ORDER BY t2.support_rate DESC LIMIT 1

### What is the name, city, and country of the airport with the highest elevation?
SELECT name ,  city ,  country FROM airports ORDER BY elevation DESC LIMIT 1

### What is the title of the film that has the highest high market estimation.
SELECT t1.title FROM film AS T1 JOIN film_market_estimation AS T2  ON T1.Film_ID  =  T2.Film_ID ORDER BY high_estimate DESC LIMIT 1

### What are the title and rental rate of the film with the highest rental rate?
SELECT title ,  rental_rate FROM film ORDER BY rental_rate DESC LIMIT 1

### What is the id of the trip that started from the station with the highest dock count?
SELECT T1.id FROM trip AS T1 JOIN station AS T2 ON T1.start_station_id  =  T2.id ORDER BY T2.dock_count DESC LIMIT 1

### which poll source does the highest oppose rate come from?
SELECT poll_source FROM candidate ORDER BY oppose_rate DESC LIMIT 1

### What is the stories of highest building?
SELECT Stories FROM buildings ORDER BY Height DESC LIMIT 1

### Your task: 
Answer the final question below by providing **only** the final SQLite SQL query syntax without commentary and explanation.  You must minimize SQL execution time while ensuring correctness.

    ### Sqlite SQL tables, with their properties:
#
# stadium(Stadium_ID, Location, Name, Capacity, Highest, Lowest, Average);
# singer(Singer_ID, Name, Country, Song_Name, Song_release_year, Age, Is_male);
# concert(concert_ID, concert_Name, Theme, Stadium_ID, Year);
# singer_in_concert(concert_ID, Singer_ID).
#
    # ### Here are some data information about database references.
    # #
# stadium(Stadium_ID[1,2,3],Location[Raith Rovers,Ayr United,East Fife],Name[Stark's Park,Somerset Park,Bayview Stadium],Capacity[10104,11998,2000],Highest[4812,2363,1980],Lowest[1294,1057,533],Average[2106,1477,864]);
# singer(Singer_ID[1,2,3],Name[Joe Sharp,Timbaland,Justin Brown],Country[Netherlands,United States,France],Song_Name[You,Dangerous,Hey Oh],Song_release_year[1992,2008,2013],Age[52,32,29],Is_male[F,T,T]);
# concert(concert_ID[1,2,3],concert_Name[Auditions,Super bootcamp,Home Visits],Theme[Free choice,Free choice 2,Bleeding Love],Stadium_ID[1,2,2],Year[2014,2014,2015]);
# singer_in_concert(concert_ID[1,1,1],Singer_ID[2,3,5]);
#
### Foreign key information of Sqlite SQL tables, used for table joins: 
#
# concert(Stadium_ID) REFERENCES stadium(Stadium_ID);
# singer_in_concert(Singer_ID) REFERENCES singer(Singer_ID);
# singer_in_concert(concert_ID) REFERENCES concert(concert_ID)
#
### Final Question: Show all countries and the number of singers in each country.
### SQL: 
        """
    ]
    responses = chat_bot.generate_batch(batch_prompts)
    for i, res in enumerate(responses):
        print(f"Response: {res}")
