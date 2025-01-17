# import openai
import os
from openai import OpenAI

# Configure OpenAI API
# openai.api_key = os.getenv("OPEN_API_KEY")  # Replace with your actual OpenAI API key
# openai.organization = os.getenv("OPEN_GROUP_ID")  # Set the OpenAI organization ID if needed


class GPT:
    def __init__(self, model="o1-preview"):
        """
        Initialize the GPT class.

        Args:
            model (str): Default model name for the OpenAI API. For example: "gpt-3.5-turbo".
        """
        self.model = model
        # 初始化 OpenAI 客戶端
        self.client = OpenAI(
            api_key=os.environ.get("OPEN_API_KEY"),  # or your actual API key
        )

    def __call__(
        self,
        prompt: str,
        model: str = None,  # <-- 新增 model 參數，預設 None
        temperature: float = 0.7,
        max_tokens: int = 200,
        **kwargs
    ) -> str:
        """
        Generate a response from the OpenAI API using the ChatCompletion endpoint.

        Args:
            prompt (str): The input prompt for the model.
            model (str): Which model to use. If None, defaults to self.model.
            temperature (float): Sampling temperature for randomness.
            max_tokens (int): Maximum number of tokens to generate.
            **kwargs: Additional parameters for the OpenAI API.

        Returns:
            str: The generated response.
        """
        if not prompt or prompt.strip() == "":
            raise ValueError("Prompt cannot be None or empty")
        use_model = model if model else self.model
        client = self.client
        
        try:
            # Call ChatCompletion API
            response = client.chat.completions.create(
                model=use_model,
                messages=[
                    {"role": "user", "content": str(prompt)}  # Ensure prompt is string
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
            print(f"Error calling OpenAI API: {str(e)}")
            raise

    def batch_generate(
        self,
        prompts: list,
        model: str = None,  # <-- 同樣在 batch_generate 加上 model 參數
        temperature: float = 0.7,
        max_tokens: int = 200,
        **kwargs
    ) -> list:
        """
        Generate responses for a batch of prompts.

        Args:
            prompts (list of str): List of input prompts.
            model (str): Which model to use. If None, defaults to self.model.
            temperature (float): Sampling temperature for randomness.
            max_tokens (int): Maximum number of tokens to generate.
            **kwargs: Additional parameters for the OpenAI API.

        Returns:
            list of str: List of generated responses.
        """
        responses = []
        for prompt in prompts:
            try:
                # 每條 prompt 都可以帶入同樣的 model。若要單獨針對不同 prompt 使用不同 model，請自行修改。
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
    llm = GPT(model="o1-preview")  # 也可以改成 "gpt-3.5-turbo" 或其他
    
    # Single prompt example
    prompt = "请用一句话解释万有引力"
    response = llm(prompt)  # 不傳 model 時，預設使用 self.model
    print(f"Response: {response}")

    # 若要在呼叫時另外指定模型，可這樣寫
    # response = llm(prompt, model="gpt-3.5-turbo")

    # Batch prompts example
    prompts = [
        "Explain gravity in one sentence.",
        "What is AI?",
        "How do airplanes fly?"
    ]
    batch_responses = llm.batch_generate(prompts)
    print("Batch Responses:")
    for i, res in enumerate(batch_responses):
        print(f"{i + 1}. {res}")
