from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class HuggingFaceChat:
    """A class to encapsulate interaction with the Hugging Face Llama-3.3-70B-Instruct model."""
    
    def __init__(self, model_name: str = "meta-llama/Llama-3.3-70B-Instruct"):
        """Initialize the HuggingFaceChat class with a specified model.
        
        Args:
            model_name (str): The name of the model to interact with. Default is 'meta-llama/Llama-3.3-70B-Instruct'.
        """
        self.model_name = model_name
        # Load tokenizer and model (this requires significant GPU memory for 70B model)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def generate_response(self, prompt: str) -> str:
        """Sends a single prompt to the specified model and retrieves the response.
        
        Args:
            prompt (str): The input prompt to send to the model.
        
        Returns:
            str: The content of the model's response.
        """
        try:
            # Format the prompt as per your requirement
            formatted_prompt = f"""Please only output the final sql with this format '''sql <predicted sql here>.''' {prompt}"""
            inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.device)
            
            # Generate response
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=200,  # Adjust based on your needs
                do_sample=False,     # Deterministic output
                temperature=0.0      # Ensure predictable SQL output
            )
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return response
        except Exception as e:
            print(f"An error occurred: {e}")
            return ""

    def stream_response(self, prompt: str):
        """Streams the response to the prompt from the specified model.
        
        Args:
            prompt (str): The input prompt to send to the model.
        
        Yields:
            str: Chunks of the model's response content.
        """
        try:
            formatted_prompt = f"""Please only output the final sql with this format '''sql <predicted sql here>.''' {prompt}"""
            inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.device)
            
            # Generate with streaming (not natively supported, so we simulate it)
            for i in range(0, 200, 10):  # Simulate streaming by generating in chunks
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=i + 10,
                    do_sample=False,
                    temperature=0.0
                )
                chunk = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                yield chunk[len(formatted_prompt):]  # Yield only the new content
        except Exception as e:
            print(f"An error occurred while streaming: {e}")

    def generate_batch(self, prompts: list) -> list:
        """Sends a batch of prompts to the specified model and retrieves their responses.
        
        Args:
            prompts (list): A list of input prompts to send to the model.
        
        Returns:
            list: A list of responses corresponding to each prompt.
        """
        responses = []
        for prompt in prompts:
            try:
                formatted_prompt = f"""Please only output the final sql with this format '''sql <predicted sql here>.''' {prompt}"""
                inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.device)
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=200,
                    do_sample=False,
                    temperature=0.0
                )
                response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                responses.append(response)
            except Exception as e:
                print(f"An error occurred for prompt '{prompt}': {e}")
                responses.append("")
        return responses

# Example usage
if __name__ == "__main__":
    # Define a single prompt (same as in your example)
    prompt = """
    ### Some example pairs of question and corresponding SQL query are provided based on similar problems:
    ### What is the name of the department in the Building Mergenthaler? SELECT DName FROM DEPARTMENT WHERE Building  =  "Mergenthaler"
    ### What are the card numbers of members from Kentucky? SELECT card_number FROM member WHERE Hometown LIKE "%Kentucky%"
    ### Which committees have delegates from the Democratic party? SELECT T1.Committee FROM election AS T1 JOIN party AS T2 ON T1.Party  =  T2.Party_ID WHERE T2.Party  =  "Democratic"
    ### Find the name and active date of the customer that use email as the contact channel. SELECT t1.customer_name ,  t2.active_from_date FROM customers AS t1 JOIN customer_contact_channels AS t2 ON t1.customer_id  =  t2.customer_id WHERE t2.channel_code  =  'Email'
    ### What are the names of all songs in English? SELECT song_name FROM song WHERE languages  =  "english"
    ### Find the cities which were once a host city after 2010? SELECT T1.city FROM city AS T1 JOIN hosting_city AS T2 ON T1.city_id = T2.host_city WHERE T2.year  >  2010
    ### What are the names of the airports in the city of Goroka? SELECT name FROM airports WHERE city  =  'Goroka'
    ### What is the school code of the accounting department? SELECT school_code FROM department WHERE dept_name  =  "Accounting"
    ### What city and state is the bank with the name morningside in? SELECT city ,  state FROM bank WHERE bname  =  'morningside'
    ### Your task: Answer the final question below by providing **only** the final SQLite SQL query syntax without commentary and explanation.
    ### Sqlite SQL tables, with their properties:
    # continents(ContId, Continent);
    # countries(CountryId, CountryName, Continent);
    # car_makers(Id, Maker, FullName, Country);
    # model_list(ModelId, Maker, Model);
    # car_names(MakeId, Model, Make);
    # cars_data(Id, MPG, Cylinders, Edispl, Horsepower, Weight, Accelerate, Year).
    ### Here are some data information about database references.
    # continents(ContId[1,2,3],Continent[america,europe,asia]);
    # countries(CountryId[1,2,3],CountryName[usa,germany,france],Continent[1,2,2]);
    # car_makers(Id[1,2,3],Maker[amc,volkswagen,bmw],FullName[American Motor Company,Volkswagen,BMW],Country[1,2,2]);
    # model_list(ModelId[1,2,3],Maker[1,2,3],Model[amc,audi,bmw]);
    # car_names(MakeId[1,2,3],Model[chevrolet,buick,plymouth],Make[chevrolet chevelle malibu,buick skylark 320,plymouth satellite]);
    # cars_data(Id[1,2,3],MPG[18,15,18],Cylinders[8,8,8],Edispl[307.0,350.0,318.0],Horsepower[130,165,150],Weight[3504,3693,3436],Accelerate[12.0,11.5,11.0],Year[1970,1970,1970]);
    ### Foreign key information of Sqlite SQL tables, used for table joins:
    # countries(Continent) REFERENCES continents(ContId);
    # car_makers(Country) REFERENCES countries(CountryId);
    # model_list(Maker) REFERENCES car_makers(Id);
    # car_names(Model) REFERENCES model_list(Model);
    # cars_data(Id) REFERENCES car_names(MakeId)
    ### Final Question: Find the name of the makers that produced some cars in the year of 1970?
    ### SQL:
    """

    # Define a batch of prompts
    batch_prompts = [prompt]

    # Initialize the class
    hf_chat = HuggingFaceChat()

    # Generate a single response
    print("Single Response:")
    response = hf_chat.generate_response(prompt)
    print(response)

    # Stream a single response
    print("\nStreaming Response:")
    for chunk in hf_chat.stream_response(prompt):
        print(chunk, end='', flush=True)

    # Generate responses for a batch of prompts
    print("\n\nBatch Responses:")
    batch_responses = hf_chat.generate_batch(batch_prompts)
    for i, res in enumerate(batch_responses):
        print(f"Response {i + 1}: {res}\n")