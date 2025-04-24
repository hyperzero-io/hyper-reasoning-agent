import os
from dotenv import load_dotenv
from pathlib import Path

# Load .env file if it exists
load_dotenv(dotenv_path=Path('.') / '.env', override=True)

class Config:
    # Example: Add all config keys you want to use here
    # You can add more as needed
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
    # Add other keys as needed, e.g.:
    # OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    # AZURE_API_KEY = os.getenv('AZURE_API_KEY')
    # ...
    # OpenAI and Agent model config
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    OPENAI_BASE_URL = os.getenv('OPENAI_BASE_URL', 'https://api.openai.com/v1')
    OPENAI_MODEL_GPT4 = os.getenv('OPENAI_MODEL_GPT4', 'gpt-4.1-2025-04-14')
    OPENAI_MODEL_GPT45 = os.getenv('OPENAI_MODEL_GPT45', 'gpt-4.5-preview-2025-02-27')
    OPENAI_MODEL_O4MINI = os.getenv('OPENAI_MODEL_O4MINI', 'o4-mini-2025-04-16')
    # Add more as needed

config = Config() 