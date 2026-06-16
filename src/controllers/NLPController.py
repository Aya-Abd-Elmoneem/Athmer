from .BaseController import BaseController
from models.db_schemes import Project, DataChunk
from stores.llm.LLMEnums import DocumentTypeEnum
from typing import List
import json

class NLPController(BaseController):

    def __init__(self, vectordb_client, generation_client, 
                 embedding_client, template_parser):
        super().__init__()

        self.vectordb_client = vectordb_client
        self.generation_client = generation_client
        self.embedding_client = embedding_client
        self.template_parser = template_parser

    def create_collection_name(self, project_id: str):
        return f"collection_{project_id}".strip()
    
    def reset_vector_db_collection(self, project: Project):
        collection_name = self.create_collection_name(project_id=project.project_id)
        return self.vectordb_client.delete_collection(collection_name=collection_name)
    
    def get_vector_db_collection_info(self, project: Project):
        collection_name = self.create_collection_name(project_id=project.project_id)
        collection_info = self.vectordb_client.get_collection_info(collection_name=collection_name)

        return json.loads(
            json.dumps(collection_info, default=lambda x: x.__dict__)
        )
    
    def index_into_vector_db(self, project: Project, chunks: List[DataChunk],
                                   chunks_ids: List[int], 
                                   do_reset: bool = False):
        
        # step1: get collection name
        collection_name = self.create_collection_name(project_id=project.project_id)

        # step2: manage items
        texts = [ c.chunk_text for c in chunks ]
        metadata = [ c.chunk_metadata for c in  chunks]
        vectors = [
            self.embedding_client.embed_text(text=text, 
                                             document_type=DocumentTypeEnum.DOCUMENT.value)
            for text in texts
        ]

        # step3: create collection if not exists
        _ = self.vectordb_client.create_collection(
            collection_name=collection_name,
            embedding_size=self.embedding_client.embedding_size,
            do_reset=do_reset,
        )

        # step4: insert into vector db
        _ = self.vectordb_client.insert_many(
            collection_name=collection_name,
            texts=texts,
            metadata=metadata,
            vectors=vectors,
            record_ids=chunks_ids,
        )

        return True

    def search_vector_db_collection(self, project: Project, text: str, limit: int = 10):

        # step1: get collection name
        collection_name = self.create_collection_name(project_id=project.project_id)

        # step2: get text embedding vector
        vector = self.embedding_client.embed_text(text=text, 
                                                 document_type=DocumentTypeEnum.QUERY.value)

        if not vector or len(vector) == 0:
            return False

        # step3: do semantic search
        results = self.vectordb_client.search_by_vector(
            collection_name=collection_name,
            vector=vector,
            limit=limit
        )

        if not results:
            return False

        return results
    
    def rewrite_query(self, query: str):
        """تحويل العامية المصرية لفصحى مناسبة للـ retrieval"""

        rewrite_prompt = "\n".join([
            "أنت مساعد متخصص في تحويل الأسئلة العامية المصرية إلى فصحى واضحة.",
            "حوّل السؤال التالي لفصحى مناسبة للبحث في وثائق البنك الزراعي.",
            "أرجع الفصحى فقط بدون أي كلام إضافي.",
            "",
            f"السؤال: {query}",
            "الفصحى:",
        ])

        formal = self.generation_client.generate_text(
            prompt=rewrite_prompt,
            chat_history=[],
            max_output_tokens=100,
        )

        # لو الـ rewrite فشل، استخدمي السؤال الأصلي
        return formal.strip() if formal else query


    def detect_intent(self, query: str):
        """تحديد نية المستخدم قبل الـ RAG"""

        query_lower = query.strip().lower()

        INTENTS = {
            "greeting":  ["اهلا", "مرحبا", "السلام", "صباح", "مساء", "هاي", "hello", "hi"],
            "identity":  ["من انت", "ما هو دورك", "دورك", "انت مين", "عرف بنفسك"],
            "loan":      ["قرض", "تمويل", "اقتراض", "ائتمان", "سلفة"],
            "insurance": ["تأمين", "حماية", "ضمان"],
            "card":      ["بطاقة", "كارت", "فلاح"],
            "savings":   ["ادخار", "توفير", "حساب"],
            "out_scope": ["كورة", "طقس", "اخبار", "سياسة", "رياضة"],
        }

        RESPONSES = {
            "greeting": "أهلاً بك في أثمر! 🌾 أنا هنا لمساعدتك في خدمات البنك الزراعي المصري. اسألني عن القروض، التأمين، بطاقة الفلاح، أو الادخار.",
            "identity": "أنا أثمر، المساعد الذكي للبنك الزراعي المصري. 🌱 بساعدك تعرف كل حاجة عن خدمات البنك — قروض، تأمين زراعي، بطاقة الفلاح، والادخار.",
            "out_scope": "أنا متخصص في خدمات البنك الزراعي المصري بس. للمساعدة في موضوع تاني، تواصل مع البنك على 19929. 📞",
        }

        for intent, keywords in INTENTS.items():
            if any(kw in query_lower for kw in keywords):
                if intent in RESPONSES:
                    return intent, RESPONSES[intent]  # رد جاهز بدون RAG
                return intent, None  # يدخل الـ RAG لكن بنعرف الـ intent

        return "general", None  # سؤال عام → RAG
    def expand_query(self, query: str):
        """توليد صياغات بديلة للسؤال"""

        expansion_prompt = "\n".join([
            "اكتب 3 طرق مختلفة للسؤال التالي باستخدام مرادفات مختلفة.",
            "كل سؤال في سطر منفصل. بدون ترقيم أو شرح.",
            "",
            f"السؤال: {query}",
        ])

        expanded_text = self.generation_client.generate_text(
            prompt=expansion_prompt,
            chat_history=[],
            max_output_tokens=150,
        )

        if not expanded_text:
            return [query]

        variants = [q.strip() for q in expanded_text.strip().split("\n") if q.strip()]
        return [query] + variants[:3]  # الأصلي + 3 variants
    
    def answer_rag_question(self, project: Project, query: str, limit: int = 10):
        
        answer, full_prompt, chat_history = None, None, None

        # ── 1. Intent Detection ──────────────────────────
        intent, quick_response = self.detect_intent(query)
        if quick_response:
            return quick_response, None, None

        # ── 2. Query Rewriting (عامية → فصحى) ───────────
        formal_query = self.rewrite_query(query)

        # ── 3. Query Expansion ───────────────────────────
        query_variants = self.expand_query(formal_query)

        all_results = []
        seen_ids = set()
        for variant in query_variants:
            results = self.search_vector_db_collection(
                project=project, text=variant, limit=5,
            )
            if results:
                for r in results:
                    if r.id not in seen_ids:
                        seen_ids.add(r.id)
                        all_results.append(r)

        retrieved_documents = sorted(all_results, key=lambda x: x.score, reverse=True)[:limit]

        if not retrieved_documents:
            return answer, full_prompt, chat_history
        
        # step2: Construct LLM prompt
        system_prompt = self.template_parser.get("rag", "system_prompt")

        documents_prompts = "\n".join([
            self.template_parser.get("rag", "document_prompt", {
                    "doc_num": idx + 1,
                    "chunk_text": doc.text,
            })
            for idx, doc in enumerate(retrieved_documents)
        ])

        footer_prompt = self.template_parser.get("rag", "footer_prompt", {
            "query": query
        })

        # step3: Construct Generation Client Prompts
        chat_history = [
            self.generation_client.construct_prompt(
                prompt=system_prompt,
                role=self.generation_client.enums.SYSTEM.value,
            )
        ]

        full_prompt = "\n\n".join([ documents_prompts,  footer_prompt])

        # step4: Retrieve the Answer
        answer = self.generation_client.generate_text(
            prompt=full_prompt,
            chat_history=chat_history
        )

        return answer, full_prompt, chat_history