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

# 使用示例
if __name__ == "__main__":
    chat_bot = TogetherChat()
    prompts = """
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
    
    response = chat_bot.chat_batch([prompts])
    print(response)
    
    # from together import Together

    # client = Together()

    # stream = client.chat.completions.create(
    # model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
    # messages=[{"role": "user", "content": "What are the top 3 things to do in New York?"}],
    # stream=True,
    # )

    # for chunk in stream:
    #     print(chunk.choices[0].delta.content or "", end="", flush=True)