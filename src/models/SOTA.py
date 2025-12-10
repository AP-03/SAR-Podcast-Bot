import os
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI client with API key from environment
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Default system prompt for surgical robotics and dialog tasks
DEFAULT_SYSTEM_PROMPT = """You are a helpful assistant that provides clear, concise responses.

For conversational queries:
- Respond naturally and directly to the question asked
- Keep responses focused and relevant
- Maintain appropriate brevity while being informative

For surgical robotics topics:
- Answer questions about robotic surgery concepts clearly
- Explain algorithms, techniques, and safety considerations when asked
- Provide direct answers without unnecessary formatting or structure"""


def query_gpt(
    prompt: str, 
    model: str = "gpt-4o-mini",
    system_prompt: str = None,
    temperature: float = 0.7,
    max_tokens: int = 500
) -> str:
    """
    Query OpenAI's GPT model with a prompt.
    
    Args:
        prompt: The input prompt (user message)
        model: The model to use (default: gpt-4o - latest and most capable)
        system_prompt: Custom system prompt (uses DEFAULT_SYSTEM_PROMPT if None)
        temperature: Sampling temperature (0.0-2.0)
        max_tokens: Maximum tokens in response
    
    Returns:
        The model's response text
    """
    if system_prompt is None:
        system_prompt = DEFAULT_SYSTEM_PROMPT
    
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        temperature=temperature,
        max_tokens=max_tokens
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    # Test with dialog example
    result = query_gpt("Context: How are you doing today? Response:")
    print("Dialog example:")
    print(result)
    print()
    
    # Test with robotics example
    robotics_result = query_gpt("Context: Preparation Phase Response:")
    print("Robotics example:")
    print(robotics_result)