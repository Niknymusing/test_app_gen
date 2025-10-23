from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI  # Already imported, but ensure

from pydantic import SecretStr

class LLMInstanceConfig(TypedDict):
