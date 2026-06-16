from ..LLMInterface import LLMInterface
from ..LLMEnums import OpenAIEnums  # سنستخدم نفس قيم الـ Roles (USER, SYSTEM.. إلخ) المتوافقة مع Groq
from groq import Groq
import logging

class GroqProvider(LLMInterface):

    def __init__(self, api_key: str, api_url: str = None,
                 default_input_max_characters: int = 1000,
                 default_generation_max_output_tokens: int = 1000,
                 default_generation_temperature: float = 0.1):
        
        self.api_key = api_key
        self.api_url = api_url

        self.default_input_max_characters = default_input_max_characters
        self.default_generation_max_output_tokens = default_generation_max_output_tokens
        self.default_generation_temperature = default_generation_temperature

        self.generation_model_id = None

        self.embedding_model_id = None
        self.embedding_size = None

        # إنشاء الـ Client الخاص بـ Groq
        # يدعم base_url إذا أردتِ توجيهه لـ Proxy أو سيرفر معين، وإلا يتوجه للسيرفر السحابي الرسمي
        self.client = Groq(
            api_key = self.api_key,
            base_url = self.api_url if self.api_url and len(self.api_url) else None
        )

        self.enums = OpenAIEnums
        self.logger = logging.getLogger(__name__)

    def set_generation_model(self, model_id: str):
        self.generation_model_id = model_id

    def set_embedding_model(self, model_id: str, embedding_size: int):
        """
        منصة Groq مخصصة للتوليد السريع فقط، لا تدعم الـ Embeddings مجاناً للملفات بشكل رسمي.
        """
        self.embedding_model_id = model_id
        self.embedding_size = embedding_size

    def process_text(self, text: str):
        return text[:self.default_input_max_characters].strip()

    def generate_text(self, prompt: str, chat_history: list = [], max_output_tokens: int = None,
                      temperature: float = None):
        
        if not self.client:
            self.logger.error("Groq client was not set")
            return None

        if not self.generation_model_id:
            self.logger.error("Generation model for Groq was not set")
            return None
        
        max_output_tokens = max_output_tokens if max_output_tokens else self.default_generation_max_output_tokens
        temperature = temperature if temperature else self.default_generation_temperature

        # بناء الـ Prompt وإضافته لتاريخ المحادثة بناءً على الاستراكشتر الخاص بكِ
        chat_history.append(
            self.construct_prompt(prompt=prompt, role=OpenAIEnums.USER.value)
        )

        try:
            response = self.client.chat.completions.create(
                model = self.generation_model_id,
                messages = chat_history,
                max_tokens = max_output_tokens,
                temperature = temperature
            )

            if not response or not response.choices or len(response.choices) == 0 or not response.choices[0].message:
                self.logger.error("Error while generating text with Groq")
                return None

            return response.choices[0].message.content

        except Exception as e:
            self.logger.error(f"Exception during Groq generation: {str(e)}")
            return None

    def embed_text(self, text: str, document_type: str = None):
        """
        دالة الـ Embedding يتم تركها فارغة لـ Groq لأن التخزين والـ Vectors يعتمدان على Cohere في مشروعك.
        """
        self.logger.warning("GroqProvider does not support embedding text natively. Use CoHereProvider instead.")
        return None

    def construct_prompt(self, prompt: str, role: str):
        return {
            "role": role,
            "content": self.process_text(prompt)
        }