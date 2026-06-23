from ..LLMInterface import LLMInterface
from ..LLMEnums import DocumentTypeEnum
import google.generativeai as genai
import logging

class GeminiProvider(LLMInterface):

    def __init__(self, api_key: str,
                       default_input_max_characters: int=1000,
                       default_generation_max_output_tokens: int=1000,
                       default_generation_temperature: float=0.1):

        self.api_key = api_key
        self.default_input_max_characters = default_input_max_characters
        self.default_generation_max_output_tokens = default_generation_max_output_tokens
        self.default_generation_temperature = default_generation_temperature

        self.generation_model_id = None
        self.embedding_model_id = None
        self.embedding_size = 768

        genai.configure(api_key=self.api_key)
        self.client = genai

        self.logger = logging.getLogger(__name__)

    def set_generation_model(self, model_id: str):
        self.generation_model_id = model_id

    def set_embedding_model(self, model_id: str, embedding_size: int):
        self.embedding_model_id = model_id
        self.embedding_size = embedding_size or 768

    def process_text(self, text: str):
        return text[:self.default_input_max_characters].strip()

    def generate_text(self, prompt: str, chat_history: list = [], max_output_tokens: int = None,
                      temperature: float = None):

        if not self.generation_model_id:
            self.logger.error("Generation model for Gemini was not set")
            return None

        max_output_tokens = max_output_tokens or self.default_generation_max_output_tokens
        temperature = temperature or self.default_generation_temperature

        model = self.client.GenerativeModel(self.generation_model_id)
        response = model.generate_content(
            self.process_text(prompt),
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=max_output_tokens,
                temperature=temperature,
            )
        )

        if not response or not response.text:
            self.logger.error("Error while generating text with Gemini")
            return None

        return response.text

    def embed_text(self, text: str, document_type: str = None):

        if not self.embedding_model_id:
            self.logger.error("Embedding model for Gemini was not set")
            return None

        task_type = "retrieval_document"
        if document_type == DocumentTypeEnum.QUERY:
            task_type = "retrieval_query"

        response = self.client.embed_content(
            model=self.embedding_model_id,
            content=self.process_text(text),
            task_type=task_type,
        )

        if not response or "embedding" not in response:
            self.logger.error("Error while embedding text with Gemini")
            return None

        return response["embedding"]

    def construct_prompt(self, prompt: str, role: str):
        return {
            "role": role,
            "parts": [self.process_text(prompt)]
        }