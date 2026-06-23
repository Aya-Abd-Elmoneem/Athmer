# from sentence_transformers import SentenceTransformer
# import logging

# class SentenceTransformerProvider:

#     def __init__(self, model_name="BAAI/bge-m3"):
#         self.embedding_model_id = model_name
#         self.embedding_size = None
#         self._model = None  # مش بنحمل هنا

#         self.logger = logging.getLogger(__name__)

#     def _load_model(self):
#         if self._model is None:
#             self.logger.info(f"Loading embedding model: {self.embedding_model_id}")
#             self._model = SentenceTransformer(self.embedding_model_id)
#             self.embedding_size = self._model.get_sentence_embedding_dimension()
#         return self._model

#     def embed_text(self, text: str, document_type=None):
#         model = self._load_model()
#         vector = model.encode(text, normalize_embeddings=True)
#         return vector.tolist()

#     def embed_texts(self, texts: list, document_type=None):
#         model = self._load_model()
#         vectors = model.encode(
#             texts,
#             normalize_embeddings=True,
#             batch_size=32,
#             show_progress_bar=True
#         )
#         return vectors.tolist()

#     def set_embedding_model(self, *args, **kwargs):
#         pass