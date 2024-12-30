import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


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

        logging.info(f"Generating text for prompt: {prompt}")
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

            logging.info(f"Generated batch responses: {responses}")
            return responses
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
            model_name="meta-llama/CodeLlama-7b-hf",
            max_memory={"cpu": "4GiB", 0: "22GiB"},  # Example memory allocation
        )

        print("Single prompt")
        # Single prompt example
        prompt = "Write a Python function to calculate the Fibonacci sequence."
        response = codellama(
            prompt,
            temperature=0.3,
            top_p=0.85,
            max_new_tokens=128,
            repetition_penalty=1.1,
            do_sample=True,
            num_beams=1
        )
        print(f"Response: {response}")

        print("Batch of prompts")
        batch_prompts = [
            "What is machine learning?",
            "Explain the concept of recursion.",
            "Write a SQL query to find duplicate records in a table."
        ]
        batch_responses = codellama.generate_batch(
            batch_prompts,
            temperature=0.4,
            top_p=0.9,
            max_new_tokens=128,
            repetition_penalty=1.05,
            do_sample=True,
            num_beams=1
        )
        print("Batch Responses:")
        for i, res in enumerate(batch_responses):
            print(f"{i + 1}. {res}")
    except Exception as e:
        logging.error(f"An error occurred: {e}")
