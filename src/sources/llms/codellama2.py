from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
import torch

class CodeLlama2:
    def __init__(
        self,
        model_name="beneyal/code-llama-7b-spider-qpl-lora",
        device=None,
        max_memory=None,
        torch_dtype=torch.float16,
    ):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        try:
            logging.info(f"Loading CodeLlama model '{model_name}' on {self.device}...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, legacy=False)

            # Load the base model and apply PEFT
            base_model = AutoModelForCausalLM.from_pretrained(
                "codellama/CodeLlama-7b-hf",
                torch_dtype=torch_dtype,
                device_map="auto",
                max_memory=max_memory,
            )
            self.model = PeftModel.from_pretrained(base_model, model_name)
            self.model.to(self.device)

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            logging.info("Model loaded successfully.")
        except Exception as e:
            raise RuntimeError(f"Error loading model '{model_name}': {e}")

    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_new_tokens: int = 256,
        repetition_penalty: float = 1.05,
        do_sample: bool = True,
        num_beams: int = 1,
        no_repeat_ngram_size: int = 0,
    ) -> str:
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
            )
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
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
    ) -> list:
        if not isinstance(prompts, list) or not all(isinstance(p, str) for p in prompts):
            raise ValueError("Prompts must be a list of strings.")

        try:
            inputs = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1024,
            ).to(self.device)

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
            )
            responses = [
                self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs
            ]
            return responses
        except Exception as e:
            raise RuntimeError(f"Error during batch text generation: {e}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    try:
        codellama = CodeLlama2(
            model_name="beneyal/code-llama-7b-spider-qpl-lora",
            max_memory={"cpu": "4GiB", 0: "22GiB"},
        )

        # Single prompt
        prompt = "Generate a SQL query to count all rows in a table called 'employees'."
        response = codellama.generate(
            prompt,
            temperature=0.7,
            top_p=0.85,
            max_new_tokens=128,
            repetition_penalty=1.1,
            do_sample=True,
            num_beams=1,
        )
        print(f"SQL Response: {response}")

        # Batch prompts
        prompts = [
            "Find the average salary of employees grouped by department.",
            "List all customers from New York.",
            "Show the names of employees hired in 2022.",
        ]
        batch_responses = codellama.generate_batch(
            prompts,
            temperature=0.6,
            top_p=0.9,
            max_new_tokens=128,
            repetition_penalty=1.05,
            do_sample=True,
            num_beams=1,
        )
        print("Batch SQL Responses:")
        for i, sql in enumerate(batch_responses):
            print(f"{i + 1}: {sql}")
    except Exception as e:
        logging.error(f"An error occurred: {e}")
