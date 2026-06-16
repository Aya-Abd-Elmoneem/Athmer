from sentence_transformers import SentenceTransformer
import logging

class SentenceTransformerProvider:

    def __init__(self,
                 model_name="BAAI/bge-m3"):

        self.model = SentenceTransformer(model_name)

        self.embedding_model_id = model_name

        self.embedding_size = (
            self.model.get_sentence_embedding_dimension()
        )

        self.logger = logging.getLogger(__name__)

    def embed_text(self, text: str, document_type=None):

        vector = self.model.encode(
            text,
            normalize_embeddings=True
        )

        return vector.tolist()

    def embed_texts(self, texts: list, document_type=None):

        vectors = self.model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=32,
            show_progress_bar=True
        )

        return vectors.tolist()

    def set_embedding_model(self, *args, **kwargs):
        pass