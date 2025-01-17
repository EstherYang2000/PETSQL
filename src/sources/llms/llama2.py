import torch
from transformers import AutoTokenizer, AutoModelForCausalLM,AutoModelForSeq2SeqLM
import logging
import re
from transformers import StoppingCriteriaList, StoppingCriteria

logging.basicConfig(level=logging.INFO)

def remove_verbatim_prompt(answer_text: str, prompt_text: str) -> str:
    """
    如果生成的文字以 prompt_text 開頭，就去除該部分，
    以免模型重複輸出 prompt。
    """
    stripped_answer = answer_text.strip()
    stripped_prompt = prompt_text.strip()
    if stripped_answer.startswith(stripped_prompt):
        return stripped_answer[len(stripped_prompt):].lstrip()
    return stripped_answer

def cut_after_first_sql(answer_text: str) -> str:
    """
    在文字中尋找第一個從 'SELECT' 開始到分號 ';' 結束的段落。
    若找到，回傳該 SQL；否則回傳原文字。
    這樣能截斷多次重複的 SQL。
    """
    # 忽略大小寫搜尋
    match = re.search(r'(SELECT.*?;)', answer_text, flags=re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        return answer_text.strip()

class StopOnStrCriteria(StoppingCriteria):
    """
    偵測生成出的文字中是否包含指定的字串，一旦出現就停止。
    """
    def __init__(self, stop_string: str, tokenizer):
        super().__init__()
        self.stop_string = stop_string
        self.tokenizer = tokenizer

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # 1) decode目前已生成的 tokens
        text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        # 2) 如果 text 中包含停用字串，就回傳 True 以終止生成
        if self.stop_string in text:
            return True
        return False

class Llama2:
    def __init__(self, model_name="meta-llama/Llama-3.2-1B", device=None, max_memory=None):
        """
        Initialize the Llama2 class with a specified model and device.

        Args:
            model_name (str): Name of the model to load.
            device (str): Device to use ('cuda' or 'cpu'). Defaults to auto-detection.
            max_memory (dict): Memory allocation for devices (e.g., {"cpu": "4GiB", 0: "23GiB"}).
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        try:
            logging.info(f"Loading model '{model_name}' on {self.device}...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, legacy=False)
            # self.model = AutoModelForSeq2SeqLM.from_pretrained(
            #     model_name,
            #     torch_dtype=torch.float16,
            #     device_map="auto",
            #     max_memory=max_memory,
            #     )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                max_memory=max_memory,
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            logging.info("Model loaded successfully.")
        except Exception as e:
            raise RuntimeError(f"Error loading model '{model_name}': {e}")

    def __call__(self, prompt: str,stop_string=None, **generate_params) -> str:
        """
        Generate text for a single prompt.

        Args:
            prompt (str): The input prompt/string for the model.
            **generate_params: Arbitrary keyword arguments that will be passed to `self.model.generate`.
                Common options include:
                - temperature (float)
                - top_p (float)
                - max_new_tokens (int)
                - repetition_penalty (float)
                - do_sample (bool)
                - etc.

        Returns:
            str: The generated text from the model.
        """
        if not prompt.strip():
            raise ValueError("Prompt cannot be empty or whitespace.")

        logging.info(f"Generating text for prompt: {prompt}")
        try:
            # 1) Tokenize
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048,
                
            ).to(self.device)
            # 如果使用者有提供 stop_string，就建立 stopping_criteria
            stopping_criteria = None
            if stop_string is not None:
                stopping_criteria = StoppingCriteriaList([
                    StopOnStrCriteria(stop_string, self.tokenizer)
                ])
            # 2) Generate with arbitrary parameters from **generate_params
            outputs = self.model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                pad_token_id=self.tokenizer.pad_token_id,
                **generate_params  # Pass all generation params here
            )

            # 3) Decode
            # We remove the original prompt part by slicing the output
            prompt_length = inputs.input_ids.shape[1]
            generated_tokens = outputs[0][prompt_length:]
            raw_answer = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

            logging.info(f"Generated final answer: {raw_answer}")
            return raw_answer
        except Exception as e:
            raise RuntimeError(f"Error during text generation: {e}")

    def generate_batch(self, prompts,stop_string=None, **generate_params):
        """
        Generate responses for a batch of prompts.

        Args:
            prompts (list): List of prompt strings.
            **generate_params: Arbitrary keyword arguments that will be passed to `self.model.generate`.

        Returns:
            list: A list of generated responses (strings).
        """
        if not isinstance(prompts, list) or not all(isinstance(p, str) for p in prompts):
            raise ValueError("Prompts must be a list of strings.")

        logging.info("Generating responses for a batch of prompts...")
        try:
            # 1) Tokenize all prompts in a batch
            inputs = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048,
            ).to(self.device)
            stopping_criteria = None
            if stop_string is not None:
                stopping_criteria = StoppingCriteriaList([
                    StopOnStrCriteria(stop_string, self.tokenizer)
                ])
            # 2) Generate
            outputs = self.model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                pad_token_id=self.tokenizer.pad_token_id,
                stopping_criteria=stopping_criteria,
                **generate_params
            )

            # 3) Decode each output
            prompt_lengths = []
            for p in prompts:
                enc = self.tokenizer(
                    p,
                    return_tensors="pt",
                    max_length=2048,
                    truncation=True
                )
                prompt_lengths.append(enc.input_ids.shape[1])

            responses = []
            for i, output_ids in enumerate(outputs):
                start_idx = prompt_lengths[i]
                answer_ids = output_ids[start_idx:]
                raw_answer = self.tokenizer.decode(answer_ids, skip_special_tokens=True)
                # raw_answer = cut_after_first_sql(raw_answer)
                responses.append(raw_answer)

            logging.info(f"Generated batch responses: {responses}")
            return responses
        except Exception as e:
            raise RuntimeError(f"Error during batch text generation: {e}")


# Example usage
if __name__ == "__main__":
    try:
        # Initialize the Llama2 class
        llama = Llama2(
            model_name="TheBloke/deepseek-coder-6.7B-instruct-GGUF",
            max_memory={"cpu": "4GiB", 0: "22GiB"},
        )

        # Single prompt with custom generation parameters
        prompt = "Explain gravity briefly."
        response = llama(
            prompt,
            temperature=0.8,
            top_p=0.95,
            max_new_tokens=200,
            repetition_penalty=1.1,
            do_sample=True,
            stop_string="### Explanation"
        )
        print(f"Single Response:\n{response}")

        # Batch prompts with custom generation parameters
        batch_prompts = ["What is AI?", "How many planets are in the solar system?"]
        batch_responses = llama.generate_batch(
            batch_prompts,
            temperature=0.7,
            top_p=0.9,
            max_new_tokens=150,
            repetition_penalty=1.2,
            do_sample=True,
            stop_string="### Explanation"
        )
        print("Batch Responses:")
        for i, res in enumerate(batch_responses):
            print(f"{i + 1}. {res}")

    except Exception as e:
        logging.error(f"An error occurred: {e}")
