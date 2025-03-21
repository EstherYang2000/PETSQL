from openai import OpenAI

class GrokChat:
    """
    A class to encapsulate interaction with the Grok AI model using OpenAI-compatible API.
    """
    def __init__(self, model: str = "grok-1", api_key: str = "$GROK_API_KEY", base_url: str = "https://api.x.ai/v1"):
        """
        Initialize the GrokChat class with a specified model.

        Args:
            model (str): The name of the model to interact with.
            api_key (str): The API key for authentication.
            base_url (str): The base URL of the Grok API.
        """
        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url)
    
    def generate_response(self, prompt: str) -> str:
        """
        Sends a single prompt to the specified model and retrieves the response.

        Args:
            prompt (str): The input prompt to send to the model.

        Returns:
            str: The content of the model's response.
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"An error occurred: {e}")
            return ""
    
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
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                )
                responses.append(response.choices[0].message.content)
            except Exception as e:
                print(f"An error occurred for prompt '{prompt}': {e}")
                responses.append("")
        return responses
    
# Example usage
if __name__ == "__main__":
    # Define a single prompt
    prompt = "Hello"
    
    # Initialize the class
    grok_chat = GrokChat()
    
    # Generate a single response
    print("Single Response:")
    response = grok_chat.generate_response(prompt)
    print(response)
    
    # Generate responses for a batch of prompts
    # batch_prompts = ["Hello", "How are you?", "What is AI?"]
    # print("\n\nBatch Responses:")
    # batch_responses = grok_chat.generate_batch(batch_prompts)
    # for i, res in enumerate(batch_responses):
    #     print(f"Response {i + 1}: {res}\n")
