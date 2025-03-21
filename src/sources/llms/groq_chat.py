import os
from groq import Groq
import time
class GroqChat:
    """
    A class to encapsulate interaction with the Groq API.
    """

    def __init__(self, api_key: str = None, model: str = "deepseek-r1-distill-llama-70b"):
        """
        Initialize the GroqChat class with an API key and specified model.

        Args:
            api_key (str, optional): The API key for authentication. Defaults to environment variable 'GROQ_API_KEY'.
            model (str): The name of the model to interact with. Default is 'deepseek-r1-distill-llama-70b'.
        """
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("API key is required. Set it in environment variables or pass it explicitly.")
        
        self.model = model
        self.client = Groq(api_key=self.api_key)
    
    def generate_response(self, prompt: str) -> str:
        """
        Sends a single prompt to the specified model and retrieves the response.

        Args:
            prompt (str): The input prompt to send to the model.

        Returns:
            str: The content of the model's response.
        """
        retries = 0
        max_retries = 100
        while retries < max_retries:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_completion_tokens=4096,
                    top_p=1,
                )
                return response.choices[0].message.content
            except Exception as e:
                error_msg = str(e)
                if "rate_limit_exceeded" in error_msg:
                    wait_time = max(60, (retries + 1) * 30)  # Increase wait time dynamically
                    print(f"Rate limit exceeded. Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                    retries += 1
                else:
                    print(f"An error occurred: {e}")
                    return ""
        return "Request failed due to repeated rate limiting."

    def stream_response(self, prompt: str):
        """
        Streams the response to the prompt from the specified model.

        Args:
            prompt (str): The input prompt to send to the model.

        Yields:
            str: Chunks of the model's response content.
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_completion_tokens=4096,
                top_p=1,
                stream=True,
            )
            for chunk in response:
                yield chunk.choices[0].message.content
        except Exception as e:
            print(f"An error occurred while streaming: {e}")
    
    def generate_batch(self, prompts: list) -> list:
        """
        Sends a batch of prompts to the specified model and retrieves their responses.

        Args:
            prompts (list): A list of input prompts to send to the model.

        Returns:
            list: A list of responses corresponding to each prompt.
        """
        responses = []
        for prompt in prompts:
            prompt = f"""Please only output the final sql with this format '''sql <predicted sql here>.''' {prompt}"""
            response = self.generate_response(prompt)
            responses.append(response)
            time.sleep(1)  # Sleep for 1 second between requests
        return responses

# Example usage
if __name__ == "__main__":
    api_key = "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"
    groq_chat = GroqChat(api_key=api_key)
    
    # Define a single prompt
    prompt = """
    ### Some example pairs of question and corresponding SQL query are provided based on similar problems:

### Show the working years of managers in descending order of their level.
SELECT Working_year_starts FROM manager ORDER BY LEVEL DESC

### Find the name of the students and their department names sorted by their total credits in ascending order.
SELECT name ,  dept_name FROM student ORDER BY tot_cred

### Show the names of members in ascending order of their rank in rounds.
SELECT T1.Name FROM member AS T1 JOIN round AS T2 ON T1.Member_ID  =  T2.Member_ID ORDER BY Rank_in_Round ASC

### Find the name, headquarter and revenue of all manufacturers sorted by their revenue in the descending order.
SELECT name ,  headquarter ,  revenue FROM manufacturers ORDER BY revenue DESC

### For each party, find its location and the name of its host. Sort the result in ascending order of the age of the host.
SELECT T3.Location ,  T2.Name FROM party_host AS T1 JOIN HOST AS T2 ON T1.Host_ID  =  T2.Host_ID JOIN party AS T3 ON T1.Party_ID  =  T3.Party_ID ORDER BY T2.Age

### List all the cities in a decreasing order of each city's stations' highest latitude.
SELECT city FROM station GROUP BY city ORDER BY max(lat) DESC

### Show names of actors in descending order of the year their musical is awarded.
SELECT T1.Name FROM actor AS T1 JOIN musical AS T2 ON T1.Musical_ID  =  T2.Musical_ID ORDER BY T2.Year DESC

### Show the name and service for all trains in order by time.
SELECT name ,  service FROM train ORDER BY TIME

### Show all movie titles, years, and directors, ordered by budget.
SELECT title ,  YEAR ,  director FROM movie ORDER BY budget_million

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
### Final Question: Show name, country, age for all singers ordered by age from the oldest to the youngest.
### SQL: 
    
    """
    
    # Generate a single response
    print("Single Response:")
    response = groq_chat.generate_response(prompt)
    print(response)
    
    # Stream a single response
    # print("\nStreaming Response:")
    # for chunk in groq_chat.stream_response(prompt):
    #     print(chunk, end='', flush=True)
    
    # Generate responses for a batch of prompts
    batch_prompts = ["Who wrote Hamlet?", "What is the largest ocean?"]
    print("\n\nBatch Responses:")
    batch_responses = groq_chat.generate_batch(batch_prompts)
    for i, res in enumerate(batch_responses):
        print(f"Response {i + 1}: {res}\n")

# Example usage
# if __name__ == "__main__":
#     api_key = "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"
    # groq_chat = GroqChat(api_key=api_key, model="deepseek-r1-distill-llama-70b	")
    
    # Define a single prompt
    
    
    
    # # Generate a single response
    # print("Single Response:")
    # response = groq_chat.generate_response(prompt)
    # print(response)
    
    # # Stream a single response
    # print("\nStreaming Response:")
    # for chunk in groq_chat.stream_response(prompt):
    #     print(chunk, end='', flush=True)
    
    # # Generate responses for a batch of prompts
    # batch_prompts = ["Who wrote Hamlet?", "What is the largest ocean?"]
    # print("\n\nBatch Responses:")
    # batch_responses = groq_chat.generate_batch(batch_prompts)
    # for i, res in enumerate(batch_responses):
    #     print(f"Response {i + 1}: {res}\n")
    
    # from groq import Groq

    # client = Groq(api_key=api_key)

    # chat_completion = client.chat.completions.create(
    #     #
    #     # Required parameters
    #     #
    #     messages=[
    #         # Set an optional system message. This sets the behavior of the
    #         # assistant and can be used to provide specific instructions for
    #         # how it should behave throughout the conversation.
    #         {
    #             "role": "system",
    #             "content": "you are a helpful assistant."
    #         },
    #         # Set a user message for the assistant to respond to.
    #         {
    #             "role": "user",
    #             "content": "Explain the importance of fast language models",
    #         }
    #     ],

    #     # The language model which will generate the completion.
    #     model="deepseek-r1-distill-llama-70b",

    #     #
    #     # Optional parameters
    #     #

    #     # Controls randomness: lowering results in less random completions.
    #     # As the temperature approaches zero, the model will become deterministic
    #     # and repetitive.
    #     temperature=0.5,

    #     # The maximum number of tokens to generate. Requests can use up to
    #     # 32,768 tokens shared between prompt and completion.
    #     max_completion_tokens=1024,

    #     # Controls diversity via nucleus sampling: 0.5 means half of all
    #     # likelihood-weighted options are considered.
    #     top_p=1,

    #     # A stop sequence is a predefined or user-specified text string that
    #     # signals an AI to stop generating content, ensuring its responses
    #     # remain focused and concise. Examples include punctuation marks and
    #     # markers like "[end]".
    #     stop=None,

    #     # If set, partial message deltas will be sent.
    #     stream=False,
    # )

    # # Print the completion returned by the LLM.
    # print(chat_completion.choices[0].message.content)
