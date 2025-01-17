import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class DeepSeek:
    def __init__(
        self,
        model_name="deepseek-ai/deepseek-coder-7b-instruct-v1.5",
        device=None,
        max_memory=None,
        torch_dtype=torch.float16,  # Use mixed precision for better performance
    ):
        """
        Initialize the DeepSeek class with the specified model and device.

        Args:
            model_name (str): Name of the DeepSeek model on Hugging Face Hub.
            device (str): Device to use ('cuda' or 'cpu'). Defaults to auto-detection.
            max_memory (dict): Memory allocation for devices.
            torch_dtype (torch.dtype): Torch data type (float16 or float32).
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create offload directory if it doesn't exist
        offload_dir = "model_offload"
        os.makedirs(offload_dir, exist_ok=True)
        
        try:
            logging.info(f"Loading DeepSeek model '{model_name}' on {self.device}...")
            
            # Configure model loading parameters
            model_kwargs = {
                "torch_dtype": torch_dtype,
                "device_map": "auto",
                "use_cache": True,
                "low_cpu_mem_usage": True,
                "trust_remote_code": True,
                "offload_folder": offload_dir,
                "offload_state_dict": True,
                "offload_buffers": True,
            }
            
            # Add max_memory if provided
            if max_memory:
                model_kwargs["max_memory"] = max_memory

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True
            )
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                **model_kwargs
            )
            
            # Ensure model is in evaluation mode
            self.model.eval()
            
            # Set pad_token if missing
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            logging.info("Model loaded successfully.")
            
        except Exception as e:
            logging.error(f"Error during model initialization: {str(e)}")
            raise RuntimeError(f"Error loading model '{model_name}': {e}")

    def generate_text(
        self,
        prompt: str,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_new_tokens: int = 128,
        repetition_penalty: float = 1.05,
        do_sample: bool = True,
        debug: bool = False,
    ) -> str:
        """
        Generate text based on the given prompt.

        Args:
            prompt (str): Input prompt.
            temperature (float): Sampling temperature (default=0.7).
            top_p (float): Top-p sampling value (default=0.9).
            max_new_tokens (int): Maximum number of new tokens to generate (default=128).
            repetition_penalty (float): Penalty for repeated tokens (default=1.05).
            do_sample (bool): Whether to use sampling (default=True).
            debug (bool): Enable detailed logging for debugging.

        Returns:
            str: Generated text.
        """
        if debug:
            logging.getLogger().setLevel(logging.DEBUG)

        if not prompt.strip():
            raise ValueError("Prompt cannot be empty or whitespace.")

        try:
            # Refine the prompt to instruct the model to only output the SQL query
            refined_prompt = (
                "Generate an SQL query for the following question. "
                "Only output the SQL query, without any explanations or additional text.\n\n"
                f"Question: {prompt}"
            )
            # Add safety checks for parameters
            temperature = max(1e-6, min(temperature, 2.0))
            
            # Tokenize input
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            
            # Move inputs to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Generate with error handling
            with torch.no_grad():
                try:
                    outputs = self.model.generate(
                        inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        repetition_penalty=repetition_penalty,
                        do_sample=do_sample,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        num_return_sequences=1,
                        no_repeat_ngram_size=3,
                        early_stopping=True,
                    )
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        torch.cuda.empty_cache()
                        logging.warning("Encountered CUDA OOM. Cleared cache and retrying with smaller parameters...")
                        # Retry with more conservative parameters
                        outputs = self.model.generate(
                            inputs["input_ids"],
                            attention_mask=inputs["attention_mask"],
                            max_new_tokens=64,
                            temperature=0.8,
                            top_p=0.95,
                            do_sample=True,
                            num_return_sequences=1,
                            early_stopping=True,
                        )
                    else:
                        raise e

            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            if debug:
                logging.debug(f"Generated response: {response}")
            return response
            
        except Exception as e:
            logging.error(f"Generation error details: {str(e)}")
            raise RuntimeError(f"Error during text generation: {e}")

    def batch_generate(
        self,
        prompts: list,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_new_tokens: int = 128,
        repetition_penalty: float = 1.05,
        do_sample: bool = True,
        debug: bool = False,
        batch_size: int = 4,  # Process prompts in batches
        **kwargs
    ) -> list:
        """
        Generate responses for a batch of prompts.

        Args:
            prompts (list): List of input prompts.
            temperature (float): Sampling temperature (default=0.7).
            top_p (float): Top-p sampling value (default=0.9).
            max_new_tokens (int): Maximum number of new tokens to generate (default=128).
            repetition_penalty (float): Penalty for repeated tokens (default=1.05).
            do_sample (bool): Whether to use sampling (default=True).
            debug (bool): Enable detailed logging for debugging.
            batch_size (int): Number of prompts to process in a single batch.
            **kwargs: Additional parameters for generation.

        Returns:
            list: List of generated responses.
        """
        responses = []
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            try:
                # Tokenize the batch
                inputs = self.tokenizer(
                    batch_prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512,
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                # Generate responses for the batch
                with torch.no_grad():
                    outputs = self.model.generate(
                        inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        repetition_penalty=repetition_penalty,
                        do_sample=do_sample,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        num_return_sequences=1,
                        no_repeat_ngram_size=3,
                        early_stopping=True,
                    )

                # Decode and store responses
                batch_responses = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
                responses.extend(batch_responses)

                if debug:
                    for j, response in enumerate(batch_responses):
                        logging.debug(f"Prompt: {batch_prompts[j]}")
                        logging.debug(f"Response: {response}")

            except Exception as e:
                error_msg = f"Error: {str(e)}"
                logging.error(error_msg)
                responses.extend([error_msg] * len(batch_prompts))

        return responses

# Example usage
if __name__ == "__main__":
    try:
        print("Initializing the DeepSeek class...")
        
        # Calculate available CUDA memory and format it properly
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory
            # Convert to GiB and format properly
            gpu_mem_gb = int((gpu_memory * 0.8) / (1024 * 1024 * 1024))  # Use 80% of available GPU memory
            gpu_mem_str = f"{gpu_mem_gb}GiB"
        else:
            gpu_mem_str = "24GiB"

        deepseek = DeepSeek(
            model_name="deepseek-ai/deepseek-coder-33b-instruct",
            max_memory={
                "cpu": "24GiB",
                0: gpu_mem_str
            },
            torch_dtype=torch.float16,  # Use mixed precision
        )

        # Batch prompts example
        print("Generating batch responses...")
        # prompts = [
        #     "What is the role of AI in healthcare? Please provide a brief answer.",
        #     "Explain the concept of machine learning in simple terms.",
        #     "What are the main applications of natural language processing?"
        # ]
        prompts = [
            """
            ### Some example pairs of question and corresponding SQL query are provided based on similar problems:

### How many faculty do we have?
SELECT count(*) FROM Faculty

### How many aircrafts do we have?
SELECT count(*) FROM Aircraft

### How many employees do we have?
SELECT count(*) FROM Employee

### How many flights do we have?
SELECT count(*) FROM Flight

### How many accounts do we have?
SELECT count(*) FROM Accounts

### How many customers do we have?
SELECT count(*) FROM Customers

### How many tracks do we have?
SELECT count(*) FROM track

### How many transactions do we have?
SELECT count(*) FROM Financial_transactions

### How many artists do we have?
SELECT count(*) FROM artist

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
### Final Question: How many singers do we have?
### SQL:
            
            """,
            """
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
        ]
        
        batch_responses = deepseek.batch_generate(
            prompts,
            temperature=0.8,
            top_p=0.95,
            max_new_tokens=256,
            debug=True,
            batch_size=2,  # Process 2 prompts at a time
        )
        
        print("\nBatch Responses:")
        for i, res in enumerate(batch_responses, 1):
            print(f"\n{i}. {res}")

    except Exception as e:
        logging.error(f"An error occurred: {e}")