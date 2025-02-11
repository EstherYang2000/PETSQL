import os
from openai import OpenAI

class GPT:
    def __init__(self, model="deepseek-chat"):
        """
        Initialize the GPT class for DeepSeek API.

        Args:
            model (str): Default model name for the DeepSeek API. Default is "deepseek-chat".
        """
        self.model = model
        # Initialize OpenAI client with DeepSeek configuration
        self.client = OpenAI(
            api_key=os.environ.get("DEEPSEEK_API_KEY"),  # Use DeepSeek API key
            base_url="https://api.deepseek.com"  # DeepSeek API endpoint
        )

    def __call__(
        self,
        prompt: str,
        model: str = None,
        temperature: float = 0.7,
        max_tokens: int = 200,
        **kwargs
    ) -> str:
        """
        Generate a response from the DeepSeek API using the ChatCompletion endpoint.

        Args:
            prompt (str): The input prompt for the model.
            model (str): Which model to use. If None, defaults to self.model.
            temperature (float): Sampling temperature for randomness.
            max_tokens (int): Maximum number of tokens to generate.
            **kwargs: Additional parameters for the DeepSeek API.

        Returns:
            str: The generated response.
        """
        if not prompt or prompt.strip() == "":
            raise ValueError("Prompt cannot be None or empty")
        use_model = model if model else self.model
        
        try:
            # Call ChatCompletion API
            response = self.client.chat.completions.create(
                model=use_model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant"},
                    {"role": "user", "content": str(prompt)}
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            result = response.choices[0].message.content
            print(f"Prompt: {prompt}")
            print(f"Model: {use_model}")
            print(f"Result: {result}")
            return result
            
        except Exception as e:
            print(f"Error calling DeepSeek API: {str(e)}")
            raise

    def batch_generate(
        self,
        prompts: list,
        model: str = None,
        temperature: float = 0.7,
        max_tokens: int = 200,
        **kwargs
    ) -> list:
        """
        Generate responses for a batch of prompts using DeepSeek API.

        Args:
            prompts (list of str): List of input prompts.
            model (str): Which model to use. If None, defaults to self.model.
            temperature (float): Sampling temperature for randomness.
            max_tokens (int): Maximum number of tokens to generate.
            **kwargs: Additional parameters for the DeepSeek API.

        Returns:
            list of str: List of generated responses.
        """
        responses = []
        for prompt in prompts:
            try:
                prompt = f"""Please only output the final sql with this format '''sql <predicted sql here>.''' {prompt}"""
                response = self.__call__(
                    prompt,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs
                )
                responses.append(response)
            except Exception as e:
                responses.append(f"Error: {e}")
        return responses


if __name__ == '__main__':
    # Example usage
    llm = GPT(model="deepseek-chat")  # Use DeepSeek's model
    
    # Single prompt example
    # prompt = "请用一句话解释万有引力"
    # response = llm(prompt)
    # print(f"Response: {response}")

    # Example with specific model
    # response = llm(prompt, model="deepseek-chat")

    # Batch prompts example
    prompts = [
        """
        ### Some example pairs of question and corresponding SQL query are provided based on similar problems:

        ### What is the name of the department in the Building Mergenthaler?
        SELECT DName FROM DEPARTMENT WHERE Building  =  "Mergenthaler"

        ### What are the card numbers of members from Kentucky?
        SELECT card_number FROM member WHERE Hometown LIKE "%Kentucky%"

        ### Which committees have delegates from the Democratic party?
        SELECT T1.Committee FROM election AS T1 JOIN party AS T2 ON T1.Party  =  T2.Party_ID WHERE T2.Party  =  "Democratic"

        ### Find the name and active date of the customer that use email as the contact channel.
        SELECT t1.customer_name ,  t2.active_from_date FROM customers AS t1 JOIN customer_contact_channels AS t2 ON t1.customer_id  =  t2.customer_id WHERE t2.channel_code  =  'Email'

        ### What are the names of all songs in English?
        SELECT song_name FROM song WHERE languages  =  "english"

        ### Find the cities which were once a host city after 2010?
        SELECT T1.city FROM city AS T1 JOIN hosting_city AS T2 ON T1.city_id = T2.host_city WHERE T2.year  >  2010

        ### What are the names of the airports in the city of Goroka?
        SELECT name FROM airports WHERE city  =  'Goroka'

        ### What is the school code of the accounting department?
        SELECT school_code FROM department WHERE dept_name  =  "Accounting"

        ### What city and state is the bank with the name morningside in?
        SELECT city ,  state FROM bank WHERE bname  =  'morningside'

        ### Your task: 
        Answer the final question below by providing **only** the final SQLite SQL query syntax without commentary and explanation.  You must minimize SQL execution time while ensuring correctness.

            ### Sqlite SQL tables, with their properties:
        #
        # continents(ContId, Continent);
        # countries(CountryId, CountryName, Continent);
        # car_makers(Id, Maker, FullName, Country);
        # model_list(ModelId, Maker, Model);
        # car_names(MakeId, Model, Make);
        # cars_data(Id, MPG, Cylinders, Edispl, Horsepower, Weight, Accelerate, Year).
        #
            # ### Here are some data information about database references.
            # #
        # continents(ContId[1,2,3],Continent[america,europe,asia]);
        # countries(CountryId[1,2,3],CountryName[usa,germany,france],Continent[1,2,2]);
        # car_makers(Id[1,2,3],Maker[amc,volkswagen,bmw],FullName[American Motor Company,Volkswagen,BMW],Country[1,2,2]);
        # model_list(ModelId[1,2,3],Maker[1,2,3],Model[amc,audi,bmw]);
        # car_names(MakeId[1,2,3],Model[chevrolet,buick,plymouth],Make[chevrolet chevelle malibu,buick skylark 320,plymouth satellite]);
        # cars_data(Id[1,2,3],MPG[18,15,18],Cylinders[8,8,8],Edispl[307.0,350.0,318.0],Horsepower[130,165,150],Weight[3504,3693,3436],Accelerate[12.0,11.5,11.0],Year[1970,1970,1970]);
        #
        ### Foreign key information of Sqlite SQL tables, used for table joins: 
        #
        # countries(Continent) REFERENCES continents(ContId);
        # car_makers(Country) REFERENCES countries(CountryId);
        # model_list(Maker) REFERENCES car_makers(Id);
        # car_names(Model) REFERENCES model_list(Model);
        # cars_data(Id) REFERENCES car_names(MakeId)
        #
        ### Final Question: Find the name of the makers that produced some cars in the year of 1970?
        ### SQL: 
        
        """
    ]
    batch_responses = llm.batch_generate(prompts)
    print("Batch Responses:")
    for i, res in enumerate(batch_responses):
        print(f"{i + 1}. {res}")