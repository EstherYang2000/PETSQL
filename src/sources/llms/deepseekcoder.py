import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class DeepSeekCoder:
    def __init__(
        self,
        model_name="TheBloke/deepseek-coder-6.7B-instruct-GGUF",
        device=None,
        max_memory=None,
        torch_dtype=torch.float16,
    ):
        """
        Initialize the DeepSeekCoder class with a specified model and device.

        Args:
            model_name (str): Name of the model on Hugging Face Hub.
            device (str): Device to use ('cuda' or 'cpu'). Defaults to auto-detection.
            max_memory (dict): Memory allocation for devices, e.g., {"cpu": "4GiB", 0: "22GiB"}.
            torch_dtype (torch.dtype): Torch data type (float16, float32, etc.).
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        try:
            logging.info(f"Loading DeepSeekCoder model '{model_name}' on {self.device}...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, legacy=False)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                device_map="auto",
                max_memory=max_memory,
                use_cache=False,
                low_cpu_mem_usage=True,
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            logging.info("Model loaded successfully.")
        except Exception as e:
            raise RuntimeError(f"Error loading model '{model_name}': {e}")

    def generate_response(
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
        Generate a response based on the provided prompt.

        Args:
            prompt (str): Input prompt for the model.
            temperature (float): Sampling temperature for randomness (default=0.7).
            top_p (float): Top-p sampling value (default=0.9).
            max_new_tokens (int): Maximum new tokens to generate (default=256).
            repetition_penalty (float): Penalty for repeated tokens (default=1.05).
            do_sample (bool): Use sampling; if False, uses greedy/beam search (default=True).
            num_beams (int): Number of beams for beam search (default=1).
            no_repeat_ngram_size (int): Prevent repeating n-grams of this size (default=0).
            early_stopping (bool): Stop when all beams end generated sequences (default=False).
            length_penalty (float): Penalty for sequence length (default=1.0).
            pad_token_id (int): Padding token ID; defaults to tokenizer's pad_token_id.

        Returns:
            str: Generated text response.
        """
        if not prompt.strip():
            raise ValueError("Prompt cannot be empty or whitespace.")

        try:
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1024,
            ).to(self.device)

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

    def generate_batch_responses(self, prompts: list, **kwargs) -> list:
        """
        Generate responses for a batch of prompts.

        Args:
            prompts (list): List of prompts.
            **kwargs: Additional parameters for text generation.

        Returns:
            list: List of generated text responses.
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

            pad_token_id = kwargs.get("pad_token_id", self.tokenizer.pad_token_id)

            outputs = self.model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                **kwargs,
            )

            responses = [
                self.tokenizer.decode(output, skip_special_tokens=True)
                for output in outputs
            ]

            # Extract SQL queries if present in the responses
            final_sql_queries = []
            for response in responses:
                sql_query_match = re.search(r"### SQL:\s*\n(SELECT .*?)", response, re.DOTALL)
                final_sql_queries.append(sql_query_match.group(1).strip() if sql_query_match else "")

            logging.info(f"Generated batch responses: {final_sql_queries}")
            return final_sql_queries
        except Exception as e:
            raise RuntimeError(f"Error during batch text generation: {e}")

    @staticmethod
    def download_model(model_name, local_dir):
        """
        Download and save the model locally.

        Args:
            model_name (str): Name of the model to download.
            local_dir (str): Directory to save the model.

        Returns:
            str: Path to the saved model directory.
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
        deepseek = DeepSeekCoder(
            model_name="TheBloke/deepseek-coder-6.7B-instruct-GGUF",
            max_memory={"cpu": "4GiB", 0: "22GiB"},
        )

        # Single prompt
        prompt = "Write a Python function to calculate the Fibonacci sequence."
        response = deepseek.generate_response(
            prompt,
            temperature=0.3,
            top_p=0.85,
            max_new_tokens=128,
        )
        print(f"Response: {response}")

        # Batch prompts
        batch_prompts = [
            "What is machine learning?",
            "Explain the concept of recursion.",
            "Write a SQL query to find duplicate records in a table."
        ]
        batch_responses = deepseek.generate_batch_responses(
            batch_prompts,
            temperature=0.4,
            top_p=0.9,
            max_new_tokens=128,
        )
        print("Batch Responses:", batch_responses)
    except Exception as e:
        logging.error(f"An error occurred: {e}")
