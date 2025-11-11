# rag_assistant.py
import os
from dotenv import load_dotenv
from typing import Optional

load_dotenv()

# LLM clients are optional: use whichever keys/libs you have installed.
try:
    from langchain_groq import ChatGroq
except Exception:
    ChatGroq = None
try:
    from langchain_openai import ChatOpenAI
except Exception:
    ChatOpenAI = None
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except Exception:
    ChatGoogleGenerativeAI = None

# Minimal prompt composition without relying heavily on LangChain glue (keeps portable)
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser

from vectordb import VectorDB

class RAGAssistant:
    """
    RAG assistant supporting 'strict' (retrieval-only) and 'creative' (RAG+LLM) modes.
    """

    def __init__(self, mode: str = "strict"):
        self.mode = (mode or "strict").lower()
        self.vector_db = VectorDB()
        self.llm = self._get_llm()  # may raise if none configured and creative mode used
        self.chain = self._build_chain()
        print(f"✅ Chemistry Assistant ready in {self.mode.title()} Mode.")

    def _get_llm(self):
        """Return an available LLM wrapper based on env keys. If none, return None."""
        if os.getenv("GROQ_API_KEY") and ChatGroq is not None:
            return ChatGroq(api_key=os.getenv("GROQ_API_KEY"), model="llama-3.1-8b-instant")
        if os.getenv("OPENAI_API_KEY") and ChatOpenAI is not None:
            return ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o-mini")
        if os.getenv("GOOGLE_API_KEY") and ChatGoogleGenerativeAI is not None:
            return ChatGoogleGenerativeAI(google_api_key=os.getenv("GOOGLE_API_KEY"), model="gemini-pro")
        # No LLM configured — that's fine for strict mode (retrieval only)
        return None

    def _build_chain(self):
        """Create prompt template (LangChain core)."""
        strict_prompt = (
            "You are ChemGPT (STRICT). Answer ONLY using the provided CONTEXT. "
            "If context lacks the answer, respond EXACTLY: 'No known reaction found in current knowledge base.' "
            "Do NOT hypothesize."
        )
        creative_prompt = (
            "You are ChemGPT (CREATIVE). Use CONTEXT if present. If missing, you MAY infer plausible chemistry "
            "but label inferences clearly and include safety notes."
        )
        system_prompt = strict_prompt if self.mode == "strict" else creative_prompt

        user_template = "Context:\n{context}\n\nUser Question:\n{question}\n\nAnswer:"
        system_msg = SystemMessagePromptTemplate.from_template(system_prompt)
        user_msg = ChatPromptTemplate.from_template(user_template)
        return ChatPromptTemplate.from_messages([system_msg, user_msg]) | (self.llm or StrOutputParser()) | StrOutputParser()

    def add_documents(self, docs):
        """Add docs to VectorDB. docs: list[{"content": path_or_raw_text, "metadata": {...}}]"""
        self.vector_db.add_documents(docs)

    def invoke(self, question: str) -> str:
        """
        1) Retrieve top docs from VectorDB.
        2) If strict and no context -> return explicit 'No known reaction...' message.
        3) If creative or context present, call LLM chain (if configured). If no LLM configured and creative requested,
           raise an informative error.
        """
        results = self.vector_db.search(question, n_results=3)
        docs = results.get("documents", [])
        context = "\n\n".join([d for d in docs if len(d.strip()) > 20])

        if self.mode == "strict" and not context:
            return "No known reaction found in current knowledge base."

        # For strict mode with context: return the context (retrieval-only) — no LLM hallucination
        if self.mode == "strict":
            # return joined context as authoritative answer
            return context.strip() or "No known reaction found in current knowledge base."

        # For creative: require an LLM
        if not self.llm:
            raise ValueError("Creative mode requested but no LLM configured (set OPENAI_API_KEY/GROQ_API_KEY/GOOGLE_API_KEY).")

        # Build final prompt and call chain
        answer = self.chain.invoke({"context": context or "No context available", "question": question})
        # simple hallucination guard: if strict keywords appear in output, user may want strictly retrieved content
        return answer.strip()
