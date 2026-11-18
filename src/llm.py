from openai import OpenAI

from src import config


def ask(prompt):
    client = OpenAI(
        base_url=config.AZURE_OPENAI_ENDPOINT,
        api_key=config.AZURE_OPENAI_API_KEY,
    )
    # No temperature: GPT-6 Sol is a reasoning model.
    response = client.responses.create(
        model=config.AZURE_OPENAI_DEPLOYMENT,
        input=prompt,
        max_output_tokens=2000,
    )
    return response.output_text
