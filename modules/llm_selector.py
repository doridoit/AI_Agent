from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
import os

class LLMManager:
    def __init__(self):
        self.llm_instance: BaseChatModel | None = None
        self.model_id: str | None = None
        self.api_error: str | None = None

    def init(self, provider_id: str) -> str:
        self.model_id = provider_id

        try:
            if provider_id.startswith("google:"):
                model = "gemini-1.5-flash"
                api_key = os.getenv("GOOGLE_API_KEY")
                print("✅ Loaded API Key:", api_key)
                if not api_key:
                    self.api_error = "Missing GOOGLE_API_KEY"
                    return "error"
                self.llm_instance = ChatGoogleGenerativeAI(
                    model=model,
                    temperature=0.4,
                    api_key=api_key
                )
            elif provider_id.startswith("openai:"):
                model = provider_id.split(":")[1]
                api_key = os.getenv("OPENAI_API_KEY")
                print("✅ Loaded OpenAI API Key:", api_key)
                if not api_key:
                    self.api_error = "Missing OPENAI_API_KEY"
                    return "error"
                self.llm_instance = ChatOpenAI(
                    model=model,
                    temperature=0.4,
                    api_key=api_key
                )
            else:
                self.api_error = f"Unknown provider_id: {provider_id}"
                return "error"
        except Exception as e:
            self.api_error = str(e)
            return "error"

        return "ok"

    def get_llm(self) -> BaseChatModel | None:
        return self.llm_instance

    def get_error(self) -> str | None:
        return self.api_error