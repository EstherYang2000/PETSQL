import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
import re

class CodeLlama:
    def __init__(
        self,
        model_name="meta-llama/CodeLlama-7b-hf",
        device=None,
        max_memory=None,
        torch_dtype=torch.float16,
        # 你也可以在這裡增加更多參數，如 'device_map' 或 'revision'
    ):
        """
        Initialize the CodeLlama class with a specified model and device.

        Args:
            model_name (str): Name of the CodeLlama model on Hugging Face Hub.
            device (str): Device to use ('cuda' or 'cpu'). Defaults to auto-detection.
            max_memory (dict): Memory allocation for devices, e.g. {"cpu": "4GiB", 0: "22GiB"}.
            torch_dtype (torch.dtype): Torch data type (float16, float32, etc.).
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        try:
            logging.info(f"Loading CodeLlama model '{model_name}' on {self.device}...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, legacy=False)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                device_map="auto",
                max_memory=max_memory,
                use_cache=False,  # Disable KV cache
                low_cpu_mem_usage=True,
            )
            # Set pad_token if missing
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            logging.info("Model loaded successfully.")
        except Exception as e:
            raise RuntimeError(f"Error loading model '{model_name}': {e}")

    def __call__(
        self,
        prompt: str,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_new_tokens: int = 256,
        repetition_penalty: float = 1.05,
        do_sample: bool = True,
        num_beams: int = 1,
        no_repeat_ngram_size: int = 0,
        early_stopping: bool = False,
        length_penalty: float = 1.0,
        pad_token_id: int = None,
    ) -> str:
        """
        Generate a text response based on the given prompt.

        Args:
            prompt (str): Input prompt for CodeLlama.
            temperature (float): Sampling temperature for randomness (default=0.7).
            top_p (float): Top-p sampling value (default=0.9).
            max_new_tokens (int): Maximum number of new tokens to generate (default=256).
            repetition_penalty (float): Penalty for repeated tokens (default=1.05).
            do_sample (bool): Whether to use sampling; if False, uses greedy/beam search (default=True).
            num_beams (int): Number of beams for beam search (default=1).
            no_repeat_ngram_size (int): Avoid repeating n-grams of this size (default=0).
            early_stopping (bool): Stop when all beams reach the end of generated sequences (default=False).
            length_penalty (float): Exponential penalty to sequence length (default=1.0).
            pad_token_id (int): Token ID to use for padding; if None, will use model's pad_token_id.

        Returns:
            str: Generated text response.
        """
        if not prompt.strip():
            raise ValueError("Prompt cannot be empty or whitespace.")

        # logging.info(f"Generating text for prompt: {prompt}")
        try:
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1024,  # 確保輸入長度不超過模型限制
            ).to(self.device)

            # 若使用者沒有傳入 pad_token_id，就用 tokenizer 預設
            if pad_token_id is None:
                pad_token_id = self.tokenizer.pad_token_id

            outputs = self.model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                do_sample=do_sample,
                num_beams=num_beams,
                no_repeat_ngram_size=no_repeat_ngram_size,
                early_stopping=early_stopping,
                length_penalty=length_penalty,
                pad_token_id=pad_token_id,
            )
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            logging.info(f"Generated response: {response}")
            return response
        except Exception as e:
            raise RuntimeError(f"Error during text generation: {e}")

    def generate_batch(
        self,
        prompts: list,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_new_tokens: int = 256,
        repetition_penalty: float = 1.05,
        do_sample: bool = True,
        num_beams: int = 1,
        no_repeat_ngram_size: int = 0,
        early_stopping: bool = False,
        length_penalty: float = 1.0,
        pad_token_id: int = None,
    ) -> list:
        """
        Generate responses for a batch of prompts.
        Args:
            prompts (list): List of input prompts.
            temperature (float): Sampling temperature for randomness (default=0.7).
            top_p (float): Top-p sampling value (default=0.9).
            max_new_tokens (int): Maximum number of new tokens to generate (default=256).
            repetition_penalty (float): Penalty for repeated tokens (default=1.05).
            do_sample (bool): Whether to use sampling; if False, uses greedy/beam search (default=True).
            num_beams (int): Number of beams for beam search (default=1).
            no_repeat_ngram_size (int): Avoid repeating n-grams of this size (default=0).
            early_stopping (bool): Stop when all beams reach the end of generated sequences (default=False).
            length_penalty (float): Exponential penalty to sequence length (default=1.0).
            pad_token_id (int): Token ID to use for padding; if None, will use model's pad_token_id.

        Returns:
            list: List of generated responses (str).
        """
        if not isinstance(prompts, list) or not all(isinstance(p, str) for p in prompts):
            raise ValueError("Prompts must be a list of strings.")

        logging.info("Generating responses for a batch of prompts...")
        try:
            inputs = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1024,
            ).to(self.device)

            # 若使用者沒有傳入 pad_token_id，就用 tokenizer 預設
            if pad_token_id is None:
                pad_token_id = self.tokenizer.pad_token_id

            outputs = self.model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                do_sample=do_sample,
                num_beams=num_beams,
                no_repeat_ngram_size=no_repeat_ngram_size,
                early_stopping=early_stopping,
                length_penalty=length_penalty,
                pad_token_id=pad_token_id,
            )

            # 依照每個 prompt 的維度進行decode
            # 如果使用 beam search，outputs 的 shape 可能與 prompts 數量不同，需要多一層處理
            # 這裡預設 num_beams=1，故 outputs.shape[0] = len(prompts)
            responses = [
                self.tokenizer.decode(output, skip_special_tokens=True)
                for output in outputs
            ]
            # Extract SQL queries for each response
            final_sql_queries = []
            for response in responses:
                print(response)
                sql_query_match = re.search(r"### SQL:\s*\n(SELECT .*?)", response, re.DOTALL)
                if sql_query_match:
                    final_sql = sql_query_match.group(1).strip()
                    final_sql_queries.append(final_sql)
                else:
                    logging.warning("No SQL query found in response.")
                    final_sql_queries.append("")

            logging.info(f"Generated batch responses: {final_sql_queries}")
            return final_sql_queries
            
        except Exception as e:
            raise RuntimeError(f"Error during batch text generation: {e}")

    @staticmethod
    def download_model(model_name, local_dir):
        """
        Download and save the CodeLlama model locally.
        Args:
            model_name (str): Name of the model to download.
            local_dir (str): Directory to save the model.
        Returns:
            str: Local directory where the model is saved.
        """
        try:
            logging.info(f"Downloading model '{model_name}' to '{local_dir}'...")
            tokenizer = AutoTokenizer.from_pretrained(model_name, legacy=False)
            tokenizer.save_pretrained(local_dir)
            model = AutoModelForCausalLM.from_pretrained(model_name)
            model.save_pretrained(local_dir)
            logging.info("Model downloaded successfully.")
            return local_dir
        except Exception as e:
            raise RuntimeError(f"Error downloading model '{model_name}': {e}")


# Example usage
if __name__ == "__main__":
    try:
        print("Initialize the CodeLlama class")
        codellama = CodeLlama(
            # beneyal/code-llama-7b-spider-qpl-lora
            model_name="codellama/CodeLlama-34b-Instruct-hf", 
            max_memory={"cpu": "4GiB", 0: "22GiB"},  # Example memory allocation
        )

        # print("Single prompt")
        # # Single prompt example
        # prompt = """"""
        # response = codellama(
        #     prompt,
        #     temperature=0.3,
        #     top_p=0.85,
        #     max_new_tokens=2048,
        #     repetition_penalty=1.1,
        #     do_sample=True,
        #     num_beams=1
        # )
        # print(f"Response: {response}")

        print("Batch of prompts")
        batch_prompts = [
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
        batch_responses = codellama.generate_batch(
            batch_prompts,
            temperature=0.4,
            top_p=0.9,
            max_new_tokens=2056,
            repetition_penalty=1.05,
            do_sample=True,
            num_beams=1
        )
        print("Batch Responses:")
        for i, res in enumerate(batch_responses):
            print(f"{i + 1}. {res}")
    except Exception as e:
        logging.error(f"An error occurred: {e}")
